import os

import torch
import scipy.sparse as sp

from generator.generator_v4 import GeneratorV4

# Adapted for thesis documentation purposes.
# Replace the paths below with your own local paths before running.

# =========================================================
# PATH CONFIGURATION
# =========================================================
base_pretrain_path = r"PASTE_YOUR_PRETRAIN_DATA_PATH_HERE"
base_save_path = r"PASTE_YOUR_MODEL_SAVE_PATH_HERE"

region_adj_mx_path = os.path.join(base_pretrain_path, "region_adj_mx.npz")
region_features_path = os.path.join(base_pretrain_path, "region_feature.pt")

function_g_checkpoint_path = os.path.join(base_save_path, "region_function_g_fc.pt")
function_h_checkpoint_path = os.path.join(base_save_path, "region_gat_fc.pt")
output_generator_path = os.path.join(base_save_path, "my_region_generator.pt")

# =========================================================
# REGION DATA FEATURES
# =========================================================
region_adj_mx = sp.load_npz(region_adj_mx_path)
region_features = torch.load(region_features_path, map_location="cpu")

region_num = 100
time_size = 2880
loc_pad = region_num
time_pad = time_size

region_data_feature = {
    "road_num": region_num + 1,
    "time_size": time_size + 1,
    "road_pad": loc_pad,
    "time_pad": time_pad,
    "adj_mx": region_adj_mx,
    "node_features": region_features,
}

# =========================================================
# GENERATOR CONFIGURATION
# Must match the configuration used to produce the checkpoints
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
        "device": "cpu",
    },
    "function_h": {
        "embed_dim": 128,
        "gps_emb_dim": 5,
        "num_of_heads": 5,
        "concat": False,
        "device": "cpu",
        "distance_mode": "l2",
        "no_gps_emb": True,
    }
}

# =========================================================
# BUILD GENERATOR
# =========================================================
generator = GeneratorV4(config=region_gen_config, data_feature=region_data_feature).to("cpu")

# =========================================================
# LOAD PRETRAINED FUNCTION G AND FUNCTION H
# =========================================================
function_g_state = torch.load(function_g_checkpoint_path, map_location="cpu")
generator.function_g.load_state_dict(function_g_state)

function_h_state = torch.load(function_h_checkpoint_path, map_location="cpu")
generator.function_h.load_state_dict(function_h_state)

# =========================================================
# SAVE COMBINED GENERATOR
# =========================================================
torch.save(generator.state_dict(), output_generator_path)
print(f"Saved combined region generator to: {output_generator_path}")