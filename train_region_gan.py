import os
import json

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils.ListDataset import ListDataset
from utils.utils import get_logger
from utils.data_util import encode_time
from utils.evaluate_funcs import edit_distance, hausdorff_metric, dtw_metric
from generator.generator_v4 import GeneratorV4
from discriminator.discriminator_v1 import DiscriminatorV1
from search import DoubleLayerSearcher
from rollout import Rollout
from loss import gan_loss

# Adapted for thesis documentation purposes.
# Simplified for the Porto_Taxi thesis experiment only.
# This script performs adversarial / reinforcement learning
# for the region-level generator, starting from a pretrained model.
# Replace the paths below with your own local paths before running.

# =========================================================
# USER CONFIGURATION
# =========================================================
device = "cpu"

learning_rate = 0.0005
weight_decay = 0.0001
dis_train_rate = 0.8
batch_size = 64
clip = 5.0
pretrain_discriminator = True
debug = False

if debug:
    total_epoch = 1
    pretrain_dis_epoch = 1
    dis_sample_num = 10
    gen_sample_num = 1
    rollout_times = 1
else:
    total_epoch = 10
    pretrain_dis_epoch = 5
    dis_sample_num = 5000
    gen_sample_num = 2000
    rollout_times = 8

# Path to the Porto pretrain_data folder
data_root = r"PASTE_YOUR_PRETRAIN_DATA_PATH_HERE"

# Path to the Porto save folder
save_root = r"PASTE_YOUR_MODEL_SAVE_PATH_HERE"

save_folder = os.path.join(save_root, "our_region_gan")
pretrained_region_generator_file = os.path.join(save_root, "my_region_generator.pt")
trajectory_file = os.path.join(data_root, "porto_taxi_mm_region_train.csv")

# =========================================================
# MODEL CONFIGURATION
# =========================================================
region_gen_config = {
    "function_g": {
        "road_emb_size": 128,
        "time_emb_size": 32,
        "hidden_size": 128,
        "dropout_p": 0.6,
        "lstm_layer_num": 2,
        "pretrain_road_rep": None,
        "dis_weight": 0.5,
        "device": device
    },
    "function_h": {
        "embed_dim": 128,
        "gps_emb_dim": 5,
        "num_of_heads": 5,
        "concat": False,
        "device": device,
        "distance_mode": "l2",
        "no_gps_emb": True
    }
}

region_dis_config = {
    "road_emb_size": 64,
    "hidden_size": 64,
    "dropout_p": 0.6,
    "lstm_layer_num": 2,
    "pretrain_road_rep": None,
    "device": device
}

# =========================================================
# LOAD REGION FEATURES / GRAPH
# =========================================================
region_adjacent_np_file = os.path.join(data_root, "region_adj_mx.npz")
region_adj_mx = sp.load_npz(region_adjacent_np_file)

region_feature_file = os.path.join(data_root, "region_feature.pt")
region_features = torch.load(region_feature_file, map_location=device)

logger = get_logger()

# =========================================================
# LOAD SUPPORTING DATA
# =========================================================
with open(os.path.join(data_root, "adjacent_list.json"), "r", encoding="utf-8") as f:
    adjacent_list = json.load(f)

with open(os.path.join(data_root, "porto_rid_gps.json"), "r", encoding="utf-8") as f:
    rid_gps = json.load(f)

with open(os.path.join(data_root, "road_length.json"), "r", encoding="utf-8") as f:
    road_length = json.load(f)

with open(os.path.join(data_root, "region_adjacent_list.json"), "r", encoding="utf-8") as f:
    region_adjacent_list = json.load(f)

region_dist = np.load(os.path.join(data_root, "region_count_dist.npy"))

with open(os.path.join(data_root, "region_transfer_prob.json"), "r", encoding="utf-8") as f:
    region_transfer_freq = json.load(f)

with open(os.path.join(data_root, "rid2region.json"), "r", encoding="utf-8") as f:
    rid2region = json.load(f)

with open(os.path.join(data_root, "region2rid.json"), "r", encoding="utf-8") as f:
    region2rid = json.load(f)

with open(os.path.join(data_root, "porto_region_gps.json"), "r", encoding="utf-8") as f:
    region_gps = json.load(f)

with open(os.path.join(data_root, "region_od_distinct_route.json"), "r", encoding="utf-8") as f:
    od_distinct_route = json.load(f)

road_time_distribution = np.load(os.path.join(data_root, "road_time_distribution.npy"))
region_time_distribution = np.load(os.path.join(data_root, "region_time_distribution.npy"))

# =========================================================
# DATA FEATURE DESCRIPTION
# =========================================================
time_size = 2880
time_pad = time_size

region_num = len(region2rid)
loc_pad = region_num

region_data_feature = {
    "road_num": region_num + 1,
    "time_size": time_size + 1,
    "road_pad": loc_pad,
    "time_pad": time_pad,
    "adj_mx": region_adj_mx,
    "node_features": region_features
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
    label = torch.LongTensor(label)
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
        trace_loc = list(map(int, row["region_list"].split(",")))
        trace_tim = list(map(encode_time, row["time_list"].split(",")))

        data.append([trace_loc, trace_tim, 1.0])

        neg_trace_loc, neg_trace_tim = searcher.region_random_sample(
            region_model=gen_model,
            trace_loc=[trace_loc[0]],
            trace_tim=[trace_tim[0]],
            des=trace_loc[-1],
            default_len=len(trace_loc)
        )
        data.append([neg_trace_loc, neg_trace_tim, 0.0])

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
    Pretrain or update the discriminator using real and generated region trajectories.
    """
    dis_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    region_generator.train(False)
    discriminator.train(True)

    for epoch in range(max_epoch):
        pos_sample_index = np.random.randint(0, total_trace, size=dis_sample_num)
        pos_sample = trace.iloc[pos_sample_index]

        train_loader, eval_loader = generate_discriminator_data(
            gen_model=region_generator,
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
            truth = batch[2].tolist()

            for index, val in enumerate(truth):
                if val == 1 and score[index][1].item() > 0.5:
                    eval_hit += 1
                elif val == 0 and score[index][1].item() < 0.5:
                    eval_hit += 1

        avg_ac = eval_hit / eval_total_cnt
        logger.info(f"Discriminator epoch {epoch}: loss {train_total_loss:.6f}, accuracy {avg_ac:.6f}")


def train_generator(stage):
    """
    Update the generator via adversarial / reinforcement learning.
    """
    gen_optimizer = torch.optim.Adam(
        region_generator.parameters(),
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
        trace_loc = list(map(int, row["region_list"].split(",")))
        trace_tim = list(map(encode_time, row["time_list"].split(",")))

        region_generator.train(False)
        neg_trace_loc, neg_trace_tim = searcher.region_random_sample(
            region_model=region_generator,
            trace_loc=[trace_loc[0]],
            trace_tim=[trace_tim[0]],
            des=trace_loc[-1],
            default_len=len(trace_loc)
        )

        reward, yaw_distance = rollout.get_region_reward(
            generate_trace=(neg_trace_loc, neg_trace_tim),
            des=trace_loc[-1],
            discriminator=discriminator,
            rollout_times=rollout_times
        )

        region_generator.train(True)

        seq_len = len(neg_trace_loc)
        if seq_len <= 1:
            continue

        candidate_prob_list = []
        gen_candidate = []

        for i in range(1, seq_len):
            des_tensor = torch.tensor([trace_loc[-1]]).to(device)

            input_trace_loc = neg_trace_loc[:i]
            input_trace_tim = neg_trace_tim[:i]
            now_region = input_trace_loc[-1]

            candidate_region_dict = region_adjacent_list[str(now_region)]
            candidate_set = [int(k) for k in candidate_region_dict.keys()]

            candidate_dis = []
            for candidate in candidate_set:
                candidate_dis.append(region_dist[now_region][candidate] / 100)

            trace_loc_tensor = torch.LongTensor(input_trace_loc).to(device).unsqueeze(0)
            trace_tim_tensor = torch.LongTensor(input_trace_tim).to(device).unsqueeze(0)
            candidate_set_tensor = torch.LongTensor(candidate_set).to(device).unsqueeze(0)
            candidate_dis_tensor = torch.FloatTensor(candidate_dis).to(device).unsqueeze(0)

            candidate_prob = region_generator.predict(
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

        gen_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(region_generator.parameters(), clip)
        gen_optimizer.step()

        region_generator.function_h.update_node_emb()

        total_edit_distance += edit_distance(neg_trace_loc, trace_loc)

        generated_gps_list = []
        for region_id in neg_trace_loc:
            gps = region_gps[str(region_id)]
            generated_gps_list.append([gps[1], gps[0]])

        true_gps_list = []
        for region_id in trace_loc:
            gps = region_gps[str(region_id)]
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

    return total_hausdorff / total_cnt


if __name__ == "__main__":
    logger.info("Loading true trajectories.")
    trace = pd.read_csv(trajectory_file)
    total_trace = trace.shape[0]

    searcher = DoubleLayerSearcher(
        device=device,
        adjacent_list=adjacent_list,
        road_center_gps=rid_gps,
        road_length=road_length,
        region_adjacent_list=region_adjacent_list,
        region_dist=region_dist,
        region_transfer_freq=region_transfer_freq,
        rid2region=rid2region,
        road_time_distribution=road_time_distribution,
        region_time_distribution=region_time_distribution,
        region2rid=region2rid
    )

    logger.info(f"Loading pretrained region generator from {pretrained_region_generator_file}")

    region_generator = GeneratorV4(
        config=region_gen_config,
        data_feature=region_data_feature
    ).to(device)

    region_generator_state = torch.load(pretrained_region_generator_file, map_location=device)
    region_generator.load_state_dict(region_generator_state)

    rollout = Rollout(
        searcher=searcher,
        generator=region_generator,
        device=device,
        od_distinct_route=od_distinct_route,
        road_gps=rid_gps
    )

    discriminator = DiscriminatorV1(
        config=region_dis_config,
        data_feature=region_data_feature
    ).to(device)

    if pretrain_discriminator:
        logger.info("Starting discriminator pretraining.")
        train_discriminator(max_epoch=pretrain_dis_epoch)
    else:
        logger.info("Loading saved discriminator checkpoint.")
        discriminator_state = torch.load(
            os.path.join(save_folder, "adversarial_region_discriminator.pt"),
            map_location=device
        )
        discriminator.load_state_dict(discriminator_state)

    prev_hausdorff = None
    patience = 2

    for epoch in range(total_epoch):
        logger.info(f"Starting generator training at epoch {epoch}")
        current_hausdorff = train_generator(stage=1)

        if prev_hausdorff is None:
            prev_hausdorff = current_hausdorff
        elif prev_hausdorff < current_hausdorff:
            patience -= 1
        else:
            patience = 2

        if patience == 0:
            logger.info("Early stop triggered.")
            break

        logger.info(f"Starting discriminator training at epoch {epoch}")
        train_discriminator(max_epoch=1)
        rollout.update_params(region_generator)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    torch.save(region_generator.state_dict(), os.path.join(save_folder, "adversarial_region_generator.pt"))
    torch.save(discriminator.state_dict(), os.path.join(save_folder, "adversarial_region_discriminator.pt"))