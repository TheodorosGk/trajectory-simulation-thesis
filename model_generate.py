import os
import json

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from tqdm import tqdm

from utils.data_util import encode_time
from search import DoubleLayerSearcher
from generator.generator_v4 import GeneratorV4
from utils.map_manager import MapManager

# Adapted for thesis documentation purposes.
# Simplified for the Porto_Taxi thesis experiment only.
# This version loads pretrained function_g / function_h checkpoints
# (not GAN-trained generator checkpoints).
# Replace the paths below with your own local paths before running.

# =========================================================
# USER CONFIGURATION
# =========================================================
device = "cpu"
max_step = 5000

# Path to the Porto pretrain_data folder
data_root = r"PASTE_YOUR_PRETRAIN_DATA_PATH_HERE"

# Path to the Porto save folder
save_root = r"PASTE_YOUR_MODEL_SAVE_PATH_HERE"

# =========================================================
# INPUT / OUTPUT FILES
# =========================================================
true_traj_file = os.path.join(data_root, "porto_mm_test.csv")
generated_trace_file = os.path.join(data_root, "TS_TrajGen_generate.csv")

pretrain_gen_file = os.path.join(save_root, "function_g_fc.pt")
pretrain_gat_file = os.path.join(save_root, "gat_fc.pt")
pretrain_region_gen_file = os.path.join(save_root, "region_function_g_fc.pt")
pretrain_region_gat_file = os.path.join(save_root, "region_gat_fc.pt")

# =========================================================
# LOAD TEST TRAJECTORIES
# =========================================================
true_traj = pd.read_csv(true_traj_file)

# =========================================================
# MODEL CONFIGURATION
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
    },
    "dis_weight": 0.45
}

# =========================================================
# MAP MANAGER
# =========================================================
map_manager = MapManager(
    rid_gps_path=os.path.join(data_root, "porto_rid_gps.json"),
    img_unit=0.005
)

# =========================================================
# LOAD FEATURES AND GRAPH DATA
# =========================================================
node_feature_file = os.path.join(data_root, "porto_node_feature.pt")
node_features = torch.load(node_feature_file, map_location=device).to(device)

adjacent_np_file = os.path.join(data_root, "porto_adjacent_mx.npz")
adj_mx = sp.load_npz(adjacent_np_file)

region_adjacent_np_file = os.path.join(data_root, "region_adj_mx.npz")
region_adj_mx = sp.load_npz(region_adjacent_np_file)

region_feature_file = os.path.join(data_root, "region_feature.pt")
region_features = torch.load(region_feature_file, map_location=device)

road_num = 11095
time_size = 2880
loc_pad = road_num
time_pad = time_size

data_feature = {
    "road_num": road_num + 1,
    "time_size": time_size + 1,
    "road_pad": loc_pad,
    "time_pad": time_pad,
    "adj_mx": adj_mx,
    "node_features": node_features,
    "img_height": map_manager.img_height,
    "img_width": map_manager.img_width
}

with open(os.path.join(data_root, "region2rid.json"), "r", encoding="utf-8") as f:
    region2rid = json.load(f)

region_num = len(region2rid)

region_data_feature = {
    "road_num": region_num + 1,
    "time_size": time_size + 1,
    "road_pad": region_num,
    "time_pad": time_pad,
    "adj_mx": region_adj_mx,
    "node_features": region_features,
    "img_height": map_manager.img_height,
    "img_width": map_manager.img_width
}

with open(os.path.join(data_root, "adjacent_list.json"), "r", encoding="utf-8") as f:
    adjacent_list = json.load(f)

with open(os.path.join(data_root, "porto_rid_gps.json"), "r", encoding="utf-8") as f:
    rid_gps = json.load(f)

with open(os.path.join(data_root, "porto_road_length.json"), "r", encoding="utf-8") as f:
    road_length = json.load(f)

with open(os.path.join(data_root, "region_adjacent_list.json"), "r", encoding="utf-8") as f:
    region_adjacent_list = json.load(f)

region_dist = np.load(os.path.join(data_root, "region_count_dist.npy"))

with open(os.path.join(data_root, "region_transfer_prob.json"), "r", encoding="utf-8") as f:
    region_transfer_freq = json.load(f)

with open(os.path.join(data_root, "rid2region.json"), "r", encoding="utf-8") as f:
    rid2region = json.load(f)

road_time_distribution = np.load(os.path.join(data_root, "road_time_distribution.npy"))
region_time_distribution = np.load(os.path.join(data_root, "region_time_distribution.npy"))

# =========================================================
# INIT GENERATORS
# =========================================================
road_generator = GeneratorV4(config=gen_config, data_feature=data_feature).to(device)
region_generator = GeneratorV4(config=region_gen_config, data_feature=region_data_feature).to(device)

print(
    "Region generator G dims:",
    region_gen_config["function_g"]["road_emb_size"],
    region_gen_config["function_g"]["time_emb_size"],
    region_gen_config["function_g"]["hidden_size"]
)

road_generator.function_g.load_state_dict(torch.load(pretrain_gen_file, map_location=device))
road_generator.function_h.load_state_dict(torch.load(pretrain_gat_file, map_location=device))
road_generator.train(False)

region_generator.function_g.load_state_dict(torch.load(pretrain_region_gen_file, map_location=device))
region_generator.function_h.load_state_dict(torch.load(pretrain_region_gat_file, map_location=device))
region_generator.train(False)

# =========================================================
# INIT SEARCHER
# =========================================================
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

# =========================================================
# GENERATE TRAJECTORIES
# =========================================================
f = open(generated_trace_file, "w", encoding="utf-8")
f.write("traj_id,rid_list,time_list\n")

fail_cnt = 0
region_astar_fail_cnt = 0

for _, row in tqdm(true_traj.iterrows(), total=true_traj.shape[0], desc="Generating trajectories"):
    rid_list = [int(i) for i in row["rid_list"].split(",")]
    traj_id = row["traj_id"]
    time_list = list(map(encode_time, row["time_list"].split(",")))

    with torch.no_grad():
        gen_trace_loc, gen_trace_tim, is_astar = searcher.astar_search(
            region_model=region_generator,
            road_model=road_generator,
            start_rid=rid_list[0],
            start_tim=time_list[0],
            des=rid_list[-1],
            default_len=len(rid_list),
            max_step=max_step
        )

    f.write(
        '{},\"{}\",\"{}\"\n'.format(
            str(traj_id),
            ",".join([str(rid) for rid in gen_trace_loc]),
            ",".join([str(t) for t in gen_trace_tim])
        )
    )

    if gen_trace_loc[-1] != rid_list[-1]:
        fail_cnt += 1

    if is_astar == 0:
        region_astar_fail_cnt += 1

f.close()

print("fail cnt", fail_cnt)
print("region astar fail cnt", region_astar_fail_cnt)

searcher.save_fail_log()