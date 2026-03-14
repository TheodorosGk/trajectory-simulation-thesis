import os
import json

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from tqdm import tqdm
from torch.utils.data import DataLoader

from generator.distance_gat_fc import DistanceGatFC
from utils.ListDataset import ListDataset
from utils.utils import get_logger

# Adapted for thesis documentation purposes.
# Simplified for the Porto_Taxi thesis experiment only.
# This script pretrains the region-level GAT-FC model.
# Replace the paths below with your own local paths before running.

# =========================================================
# USER CONFIGURATION
# =========================================================
device = "cuda:0" if torch.cuda.is_available() else "cpu"
debug = False
train_model = True

batch_size = 32
max_epoch = 50
learning_rate = 0.0005
weight_decay = 0.0001
lr_patience = 2
lr_decay_ratio = 0.01
early_stop_lr = 1e-6

# Path to the Porto pretrain_data folder
data_root = r"PASTE_YOUR_PRETRAIN_DATA_PATH_HERE"

# Path to the Porto save folder
save_root = r"PASTE_YOUR_MODEL_SAVE_PATH_HERE"

save_folder = save_root
save_file_name = "region_gat_fc.pt"
temp_folder = os.path.join(save_root, "temp_region_gat")

# =========================================================
# MODEL CONFIGURATION
# =========================================================
config = {
    "embed_dim": 128,
    "gps_emb_dim": 5,
    "num_of_heads": 5,
    "concat": False,
    "device": device,
    "distance_mode": "l2",
    "no_gps_emb": True
}

logger = get_logger(name="RegionGatFC")
logger.info("Reading data")

# =========================================================
# LOAD GRAPH AND NODE FEATURES
# =========================================================
with open(os.path.join(data_root, "region2rid.json"), "r", encoding="utf-8") as f:
    region2rid = json.load(f)

road_num = len(region2rid)
road_num_with_pad = road_num + 1

adjacent_np_file = os.path.join(data_root, "region_adj_mx.npz")
adj_mx = sp.load_npz(adjacent_np_file)

node_feature_file = os.path.join(data_root, "region_feature.pt")
node_features = torch.load(node_feature_file, map_location="cpu").to(device)

data_feature = {
    "adj_mx": adj_mx,
    "node_features": node_features
}

# =========================================================
# INIT MODEL
# =========================================================
gat = DistanceGatFC(config=config, data_feature=data_feature).to(device)

logger.info("Initialized GAT model")
logger.info(gat)

optimizer = torch.optim.Adam(
    gat.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode="max",
    patience=lr_patience,
    factor=lr_decay_ratio
)

# =========================================================
# LOAD TRAIN / EVAL / TEST DATA
# =========================================================
train_csv = os.path.join(data_root, "porto_taxi_region_pretrain_input_train.csv")
eval_csv = os.path.join(data_root, "porto_taxi_region_pretrain_input_eval.csv")
test_csv = os.path.join(data_root, "porto_taxi_region_pretrain_input_test.csv")

train_data = pd.read_csv(train_csv).values.tolist()
eval_data = pd.read_csv(eval_csv).values.tolist()
test_data = pd.read_csv(test_csv).values.tolist()

train_num = len(train_data)
eval_num = len(eval_data)
test_num = len(test_data)
total_data = train_num + eval_num + test_num

logger.info(
    "Total input records: {}. Train: {}, Val: {}, Test: {}".format(
        total_data, train_num, eval_num, test_num
    )
)

train_dataset = ListDataset(train_data)
eval_dataset = ListDataset(eval_data)
test_dataset = ListDataset(test_data)

region_dist = np.load(os.path.join(data_root, "region_count_dist.npy"))


def collate_fn(indices):
    """
    Custom DataLoader collation function for region-level candidate prediction.
    """
    batch_des = []
    batch_candidate_set = []
    batch_candidate_dis = []
    batch_target = []
    candidate_set_len = []

    for item in indices:
        batch_des.append(item[2])

        candidate_set = [int(i) for i in item[3].split(",")]

        candidate_dis = []
        for candidate in candidate_set:
            dis = region_dist[candidate][item[2]]
            if dis == -1:
                dis = 100000
            candidate_dis.append(dis / 100)

        batch_candidate_set.append(candidate_set)
        batch_candidate_dis.append(candidate_dis)
        batch_target.append(item[5])
        candidate_set_len.append(len(candidate_set))

    max_candidate_size = max(candidate_set_len)

    for i in range(len(batch_des)):
        while len(batch_candidate_set[i]) < max_candidate_size:
            assert len(batch_candidate_set[i]) != 1, "Candidate set size is 1"
            pad_index = np.random.randint(len(batch_candidate_set[i]))
            if pad_index != batch_target[i]:
                batch_candidate_set[i].append(batch_candidate_set[i][pad_index])
                batch_candidate_dis[i].append(batch_candidate_dis[i][pad_index])

    return [
        torch.LongTensor(batch_des).to(device),
        torch.LongTensor(batch_candidate_set).to(device),
        torch.FloatTensor(batch_candidate_dis).to(device),
        torch.LongTensor(batch_target).to(device),
    ]


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# =========================================================
# TRAIN / LOAD MODEL
# =========================================================
if train_model:
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    metrics = []

    for epoch in range(max_epoch):
        logger.info(f"Starting train epoch {epoch}")

        gat.train(True)
        train_loss = 0

        for des, candidate_set, candidate_distance, target in tqdm(train_loader, desc="Training model"):
            optimizer.zero_grad()

            loss = gat.calculate_loss(
                candidate_set=candidate_set,
                candidate_distance=candidate_distance,
                des=des,
                target=target
            )

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        gat.train(False)
        val_hit = 0

        for des, candidate_set, candidate_distance, target in tqdm(val_loader, desc="Validating model"):
            with torch.no_grad():
                candidate_score = gat.predict(
                    candidate_set=candidate_set,
                    des=des,
                    candidate_distance=candidate_distance
                )

            target = target.tolist()
            _, index = torch.topk(candidate_score, 1, dim=1)

            for i, pred in enumerate(index):
                if target[i] in pred:
                    val_hit += 1

        val_ac = val_hit / eval_num
        metrics.append(val_ac)
        lr_scheduler.step(val_ac)

        torch.save(gat.state_dict(), os.path.join(temp_folder, f"region_gat_{epoch}.pt"))

        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "==> Train Epoch {}: Train Loss {:.6f}, val acc {}, lr {}".format(
                epoch, train_loss, val_ac, lr
            )
        )

        if lr < early_stop_lr:
            logger.info("Early stop triggered")
            break

    best_epoch = np.argmax(metrics)
    load_temp_file = f"region_gat_{best_epoch}.pt"
    logger.info(f"Loading best model from epoch {best_epoch}")
    gat.load_state_dict(torch.load(os.path.join(temp_folder, load_temp_file), map_location=device))

else:
    gat.load_state_dict(torch.load(os.path.join(save_folder, save_file_name), map_location=device))

# =========================================================
# TEST MODEL
# =========================================================
gat.train(False)
test_hit = 0

for des, candidate_set, candidate_distance, target in tqdm(test_loader, desc="Testing model"):
    with torch.no_grad():
        candidate_score = gat.predict(
            candidate_set=candidate_set,
            des=des,
            candidate_distance=candidate_distance
        )

    target = target.tolist()
    _, index = torch.topk(candidate_score, 1, dim=1)

    for i, pred in enumerate(index):
        if target[i] in pred:
            test_hit += 1

test_ac = test_hit / test_num
logger.info("==> Test Result: test accuracy {}".format(test_ac))

# =========================================================
# SAVE FINAL MODEL
# =========================================================
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

torch.save(gat.state_dict(), os.path.join(save_folder, save_file_name))

# =========================================================
# CLEAN TEMP FILES
# =========================================================
if os.path.exists(temp_folder):
    for root, dirs, files in os.walk(temp_folder):
        for name in files:
            remove_path = os.path.join(root, name)
            os.remove(remove_path)