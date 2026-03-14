import os
import json

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from tqdm import tqdm
from geopy import distance
from torch.utils.data import random_split, DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils.ListDataset import ListDataset
from utils.utils import get_logger
from utils.data_util import encode_time
from utils.evaluate_funcs import edit_distance, hausdorff_metric, dtw_metric
from generator.generator_v4 import GeneratorV4
from discriminator.discriminator_v1 import DiscriminatorV1
from search import Searcher
from rollout import Rollout
from loss import gan_loss

# Adapted for thesis documentation purposes.
# Simplified for the Porto_Taxi thesis experiment only.
# This script performs adversarial / reinforcement learning
# for the road-level generator, starting from pretrained components.
# Replace the paths below with your own local paths before running.

# =========================================================
# USER CONFIGURATION
# =========================================================
device = "cpu"
exp_id = 1

learning_rate = 0.0005
weight_decay = 0.0001
dis_train_rate = 0.8
batch_size = 32
pretrain_discriminator = True
debug = False
clip = 5.0

if debug:
    total_epoch = 1
    pretrain_dis_epoch = 1
    dis_sample_num = 10
    gen_sample_num = 1
    rollout_times = 1
else:
    total_epoch = 8
    pretrain_dis_epoch = 5
    dis_sample_num = 2500
    gen_sample_num = 1000
    rollout_times = 4

# Path to the Porto pretrain_data folder
data_root = r"PASTE_YOUR_PRETRAIN_DATA_PATH_HERE"

# Path to the Porto save folder
save_root = r"PASTE_YOUR_MODEL_SAVE_PATH_HERE"

save_folder = os.path.join(save_root, "our_gan")
pretrain_gan_file = os.path.join(save_root, "function_g_fc.pt")
pretrain_gat_file = os.path.join(save_root, "gat_fc.pt")
trajectory_file = os.path.join(data_root, "Porto_2026.csv")

# =========================================================
# GENERATOR / DISCRIMINATOR CONFIGURATION
# =========================================================
gen_config = {
    "function_g": {
        "road_emb_size": 256,
        "time_emb_size": 50,
        "hidden_size": 256,
        "dropout_p": 0.6,
        "lstm_layer_num": 2,
        "pretrain_road_rep": None,
        "dis_weight": 0.5,
        "device": device
    },
    "function_h": {
        "embed_dim": 256,
        "gps_emb_dim": 10,
        "num_of_heads": 5,
        "concat": False,
        "device": device,
        "distance_mode": "l2"
    },
    "dis_weight": 0.45
}

dis_config = {
    "road_emb_size": 256,
    "hidden_size": 256,
    "dropout_p": 0.6,
    "lstm_layer_num": 2,
    "pretrain_road_rep": None,
    "device": device
}

# =========================================================
# LOAD ROAD FEATURES / GRAPH
# =========================================================
node_feature_file = os.path.join(data_root, "porto_node_feature.pt")
node_features = torch.load(node_feature_file, map_location=device).to(device)

adjacent_np_file = os.path.join(data_root, "porto_adjacent_mx.npz")
adj_mx = sp.load_npz(adjacent_np_file)

# =========================================================
# LOAD SUPPORTING DATA
# =========================================================
logger = get_logger()

with open(os.path.join(data_root, "od_distinct_route.json"), "r", encoding="utf-8") as f:
    od_distinct_route = json.load(f)

with open(os.path.join(data_root, "adjacent_list.json"), "r", encoding="utf-8") as f:
    adjacent_list = json.load(f)

with open(os.path.join(data_root, "porto_rid_gps.json"), "r", encoding="utf-8") as f:
    rid_gps = json.load(f)

with open(os.path.join(data_root, "road_length.json"), "r", encoding="utf-8") as f:
    road_length = json.load(f)

road_time_distribution = np.load(os.path.join(data_root, "road_time_distribution.npy"))

# =========================================================
# DATA FEATURE DESCRIPTION
# =========================================================
road_num = 11095
time_size = 2880
loc_pad = road_num
time_pad = time_size

road2grid_path = os.path.join(data_root, "porto_road2grid.json")
with open(road2grid_path, "r", encoding="utf-8") as f:
    road2grid = json.load(f)

rows, cols = [], []
for _, value in road2grid.items():
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        row_idx, col_idx = int(value[0]), int(value[1])
    elif isinstance(value, dict) and "row" in value and "col" in value:
        row_idx, col_idx = int(value["row"]), int(value["col"])
    else:
        continue
    rows.append(row_idx)
    cols.append(col_idx)

img_height = max(rows) + 1
img_width = max(cols) + 1

# Keep the swap because it was needed to match the saved gat_fc.pt
img_height, img_width = img_width, img_height

print("img_height, img_width =", img_height, img_width)

data_feature = {
    "road_num": road_num + 1,
    "time_size": time_size + 1,
    "road_pad": loc_pad,
    "time_pad": time_pad,
    "adj_mx": adj_mx,
    "node_features": node_features,
    "img_height": img_height,
    "img_width": img_width
}


def collate_fn(indices):
    """
    Custom DataLoader collation function for discriminator training.
    """
    trace_loc = []
    trace_tim = []
    label = []

    for item in indices:
        trace_loc.append(torch.tensor(item[0]))
        trace_tim.append(torch.tensor(item[1]))
        label.append(item[2])

    trace_loc = pad_sequence(trace_loc, batch_first=True, padding_value=loc_pad)
    trace_tim = pad_sequence(trace_tim, batch_first=True, padding_value=time_pad)
    label = torch.tensor(label)
    trace_mask = ~(trace_loc == loc_pad)

    return [
        trace_loc.to(device),
        trace_tim.to(device),
        label.to(device),
        trace_mask.to(device)
    ]


def generate_discriminator_data(pos, gen_model):
    """
    Build discriminator training data from:
    - positive samples: real trajectories
    - negative samples: trajectories generated by the current generator
    """
    data = []

    for _, row in tqdm(pos.iterrows(), total=pos.shape[0], desc="Generating discriminator data"):
        trace_loc = list(map(int, row["rid_list"].split(",")))
        trace_tim = list(map(encode_time, row["time_list"].split(",")))

        data.append([trace_loc, trace_tim, 1])

        neg_trace_loc, neg_trace_tim = searcher.road_random_sample(
            gen_model=gen_model,
            trace_loc=[trace_loc[0]],
            trace_tim=[trace_tim[0]],
            des=trace_loc[-1],
            default_len=len(trace_loc)
        )
        data.append([neg_trace_loc, neg_trace_tim, 0])

    dataset = ListDataset(data)

    train_num = int(len(dataset) * dis_train_rate)
    eval_num = len(dataset) - train_num
    train_dataset, eval_dataset = random_split(dataset, [train_num, eval_num])

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
        DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
    )


def train_discriminator(max_epoch):
    """
    Pretrain or update the discriminator using real and generated road trajectories.
    """
    dis_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    generator.train(False)
    discriminator.train(True)

    for epoch in range(max_epoch):
        pos_sample_index = np.random.randint(0, total_trace, size=dis_sample_num)
        pos_sample = trace.iloc[pos_sample_index]

        train_loader, eval_loader = generate_discriminator_data(
            gen_model=generator,
            pos=pos_sample
        )

        train_total_loss = 0
        discriminator.train(True)

        for batch in tqdm(train_loader, desc="Training discriminator"):
            dis_optimizer.zero_grad()

            score = discriminator.forward(
                trace_loc=batch[0],
                trace_time=batch[1],
                trace_mask=batch[3]
            )
            loss = discriminator.loss_func(score, batch[2])

            loss.backward()
            train_total_loss += loss.item()
            dis_optimizer.step()

        discriminator.train(False)
        eval_hit = 0
        eval_total_cnt = len(eval_loader.dataset)

        for batch in tqdm(eval_loader, desc="Evaluating discriminator"):
            score = discriminator.forward(
                trace_loc=batch[0],
                trace_time=batch[1],
                trace_mask=batch[3]
            )

            truth = batch[2]
            _, index = torch.topk(score, 1, dim=1)

            for i, pred in enumerate(index):
                if truth[i] in pred:
                    eval_hit += 1

        avg_ac = eval_hit / eval_total_cnt
        logger.info(
            "Discriminator epoch {}: loss {:.6f}, top1 acc {:.6f}".format(
                epoch, train_total_loss, avg_ac
            )
        )


def train_generator(stage):
    """
    Update the generator via adversarial / reinforcement learning.
    """
    gen_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    discriminator.train(False)

    pos_sample_index = np.random.randint(0, total_trace, size=gen_sample_num)
    pos_sample = trace.iloc[pos_sample_index]

    total_edit_distance = 0
    total_hausdorff = 0
    total_dtw = 0
    total_cnt = 0

    for _, row in tqdm(pos_sample.iterrows(), total=pos_sample.shape[0], desc="Training generator"):
        trace_loc = list(map(int, row["rid_list"].split(",")))
        trace_tim = list(map(encode_time, row["time_list"].split(",")))

        generator.train(False)
        neg_trace_loc, neg_trace_tim = searcher.road_random_sample(
            gen_model=generator,
            trace_loc=[trace_loc[0]],
            trace_tim=[trace_tim[0]],
            des=trace_loc[-1],
            default_len=len(trace_loc)
        )

        reward, yaw_distance = rollout.get_road_reward(
            generate_trace=(neg_trace_loc, neg_trace_tim),
            des=trace_loc[-1],
            rollout_times=rollout_times,
            discriminator=discriminator
        )

        generator.train(True)

        seq_len = len(neg_trace_loc)
        if seq_len <= 1:
            continue

        des_center_gps = rid_gps[str(trace_loc[-1])]
        candidate_prob_list = []
        gen_candidate = []

        for i in range(1, seq_len):
            des_tensor = torch.tensor([trace_loc[-1]]).to(device)

            input_trace_loc = neg_trace_loc[:i]
            input_trace_tim = neg_trace_tim[:i]
            current_rid = input_trace_loc[-1]

            candidate_set = adjacent_list[str(current_rid)]
            candidate_dis = []

            for candidate in candidate_set:
                candidate_gps = rid_gps[str(candidate)]
                dist_km_scaled = distance.distance(
                    (des_center_gps[1], des_center_gps[0]),
                    (candidate_gps[1], candidate_gps[0])
                ).kilometers * 10
                candidate_dis.append(dist_km_scaled)

            trace_loc_tensor = torch.LongTensor(input_trace_loc).to(device).unsqueeze(0)
            trace_tim_tensor = torch.LongTensor(input_trace_tim).to(device).unsqueeze(0)
            candidate_set_tensor = torch.LongTensor(candidate_set).to(device).unsqueeze(0)
            candidate_dis_tensor = torch.FloatTensor(candidate_dis).to(device).unsqueeze(0)

            candidate_prob = generator.predict(
                trace_loc=trace_loc_tensor,
                trace_time=trace_tim_tensor,
                des=des_tensor,
                candidate_set=candidate_set_tensor,
                candidate_dis=candidate_dis_tensor
            )
            candidate_prob_list.append(candidate_prob.squeeze(0))

            choose_index = candidate_set.index(neg_trace_loc[i])
            gen_candidate.append(choose_index)

        reward = torch.tensor(reward).to(device)
        yaw_distance = torch.tensor(yaw_distance).to(device)
        gen_candidate = torch.tensor(gen_candidate).to(device)

        loss = gan_loss(
            candidate_prob=candidate_prob_list,
            gen_candidate=gen_candidate,
            reward=reward,
            yaw_loss=yaw_distance
        )

        torch.nn.utils.clip_grad_norm_(generator.parameters(), clip)
        gen_optimizer.zero_grad()
        loss.backward()
        gen_optimizer.step()

        generator.function_h.update_node_emb()

        total_edit_distance += edit_distance(neg_trace_loc, trace_loc)

        generated_gps_list = []
        for road_id in neg_trace_loc:
            gps = rid_gps[str(road_id)]
            generated_gps_list.append([gps[1], gps[0]])

        true_gps_list = []
        for road_id in trace_loc:
            gps = rid_gps[str(road_id)]
            true_gps_list.append([gps[1], gps[0]])

        true_gps_array = np.array(true_gps_list)
        generated_gps_array = np.array(generated_gps_list)

        total_hausdorff += hausdorff_metric(true_gps_array, generated_gps_array)
        total_dtw += dtw_metric(true_gps_array, generated_gps_array)
        total_cnt += 1

    logger.info("Generator evaluation:")
    logger.info(
        "avg EDT {}, avg Hausdorff {}, avg DTW {}".format(
            total_edit_distance / total_cnt,
            total_hausdorff / total_cnt,
            total_dtw / total_cnt
        )
    )


if __name__ == "__main__":
    logger.info("Loading true trajectories.")
    trace = pd.read_csv(trajectory_file)
    total_trace = trace.shape[0]

    searcher = Searcher(
        device=device,
        adjacent_list=adjacent_list,
        road_center_gps=rid_gps,
        road_length=road_length,
        road_time_distribution=road_time_distribution
    )

    logger.info(f"Loading pretrained generator from {pretrain_gan_file} and {pretrain_gat_file}")

    generator = GeneratorV4(config=gen_config, data_feature=data_feature).to(device)

    generatorv1_state = torch.load(pretrain_gan_file, map_location=device)
    generator.function_g.load_state_dict(generatorv1_state)

    gat_state = torch.load(pretrain_gat_file, map_location=device)
    generator.function_h.load_state_dict(gat_state)

    rollout = Rollout(
        searcher=searcher,
        generator=generator,
        device=device,
        od_distinct_route=od_distinct_route,
        road_gps=rid_gps
    )

    discriminator = DiscriminatorV1(config=dis_config, data_feature=data_feature).to(device)

    if pretrain_discriminator:
        logger.info("Starting discriminator pretraining.")
        train_discriminator(max_epoch=pretrain_dis_epoch)
    else:
        logger.info("Loading discriminator checkpoint.")
        discriminator_state = torch.load(
            os.path.join(save_folder, "adversarial_discriminator.pt"),
            map_location=device
        )
        discriminator.load_state_dict(discriminator_state)

    for epoch in range(total_epoch):
        logger.info(f"Starting generator training at epoch {epoch}")
        train_generator(stage=1)

        logger.info(f"Starting discriminator training at epoch {epoch}")
        train_discriminator(max_epoch=1)

        rollout.update_params(generator)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    torch.save(generator.state_dict(), os.path.join(save_folder, f"adversarial_3_generator_{exp_id}.pt"))
    torch.save(discriminator.state_dict(), os.path.join(save_folder, "adversarial_discriminator.pt"))