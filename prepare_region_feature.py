import os
import json

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

from generator.distance_gat_fc import DistanceGatFC
from utils.map_manager import MapManager

# Adapted for thesis documentation purposes.
# Simplified for the Porto_Taxi thesis experiment only.
# This script builds region_feature.pt from pretrained road/node embeddings.
# Replace the paths below with your own local paths before running.

# =========================================================
# USER CONFIGURATION
# =========================================================
device = "cpu"

# Path to the Porto pretrain_data folder
data_root = r"PASTE_YOUR_PRETRAIN_DATA_PATH_HERE"

# Path to the Porto save folder
save_root = r"PASTE_YOUR_MODEL_SAVE_PATH_HERE"

rid_gps_path = os.path.join(data_root, "porto_rid_gps.json")
out_path = os.path.join(data_root, "region_feature.pt")

# =========================================================
# MODEL CONFIGURATION
# Must match the road-level GAT-FC used during pretraining
# =========================================================
config = {
    "embed_dim": 256,
    "gps_emb_dim": 10,
    "num_of_heads": 5,
    "concat": False,
    "device": device,
    "distance_mode": "l2",
}

road_num = 11095

adjacent_np_file = os.path.join(data_root, "porto_adjacent_mx.npz")
node_feature_file = os.path.join(data_root, "porto_node_feature.pt")
gat_checkpoint_file = os.path.join(save_root, "gat_fc.pt")

rid2region_path = os.path.join(data_root, "rid2region.json")
region2rid_path = os.path.join(data_root, "region2rid.json")

# =========================================================
# LOAD ROAD-LEVEL GRAPH AND FEATURES
# =========================================================
adj_mx = sp.load_npz(adjacent_np_file)
node_features = torch.load(node_feature_file, map_location=device).to(device)

map_manager = MapManager(rid_gps_path=rid_gps_path, img_unit=0.005)

data_feature = {
    "adj_mx": adj_mx,
    "node_features": node_features,
    "img_width": map_manager.img_width,
    "img_height": map_manager.img_height,
}

# =========================================================
# LOAD PRETRAINED ROAD GAT MODEL
# =========================================================
road_gat = DistanceGatFC(config=config, data_feature=data_feature).to(device)
road_gat.load_state_dict(torch.load(gat_checkpoint_file, map_location=device))

# Build / refresh node embeddings
road_gat._setup_node_emb()

# Shape: (road_num_with_pad, feature_dim)
node_emb_feature = road_gat.node_emb_feature

# =========================================================
# LOAD ROAD-TO-REGION MAPPINGS
# =========================================================
with open(rid2region_path, "r", encoding="utf-8") as f:
    rid2region = json.load(f)

with open(region2rid_path, "r", encoding="utf-8") as f:
    region2rid = json.load(f)

region_num = len(region2rid)

# =========================================================
# BUILD REGION-TO-ROAD AGGREGATION MATRIX
# Shape: (region_num, road_num)
# =========================================================
region2rid_mat = np.zeros((region_num, road_num), dtype=float)

for rid in tqdm(rid2region, desc="Building region-to-road aggregation matrix"):
    region_id = rid2region[rid]
    region2rid_mat[region_id][int(rid)] = 1.0

region2rid_mat = torch.FloatTensor(region2rid_mat).to(device)

# Remove padding row if present
node_emb_feature = node_emb_feature[:road_num]

# =========================================================
# BUILD REGION FEATURES
# Each region feature is the sum of its road embeddings
# =========================================================
region_feature = torch.matmul(region2rid_mat, node_emb_feature)

torch.save(region_feature, out_path)

print("Saved:", out_path)
print("Region feature shape:", tuple(region_feature.shape))