import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.data_util import encode_time
from utils.parser import str2bool
from search import DoubleLayerSearcher
import json
from generator.generator_v4 import GeneratorV4
import torch
import scipy.sparse as sp
import argparse
import os
import math
from utils.map_manager import MapManager


def get_dataset_root_from_data_root(data_root: str, dataset_name: str, local: bool) -> str:
    """
    For Porto_Taxi local setup, data_root usually ends with:
    .../Porto_Taxi/TS_TrajGen_data_archive/pretrain_data
    so dataset_root is two levels up.
    Otherwise, fall back to current working directory.
    """
    if local and dataset_name == "Porto_Taxi":
        # pretrain_data -> TS_TrajGen_data_archive -> Porto_Taxi
        return os.path.abspath(os.path.join(data_root, os.pardir, os.pardir))
    return os.path.abspath(".")


parser = argparse.ArgumentParser()
parser.add_argument('--local', type=str2bool, default=True)
parser.add_argument('--dataset_name', type=str, default='Porto_Taxi')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--use_gan', type=str2bool, default=True, help='Load GAN-trained generator checkpoints if available')
parser.add_argument('--max_traj', type=int, default=0, help='0 = all, otherwise sample N trajectories')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_step', type=int, default=5000)

args = parser.parse_args()
local = args.local
dataset_name = args.dataset_name
device = args.device
use_gan = args.use_gan

archive_data_folder = 'TS_TrajGen_data_archive'

if local:
    data_root = r'C:\Users\thodo\Desktop\DIPLOMATIKI ERGASIA\EFARMOGI\TS_TRAJGEN DATASET DONLOADED FROM USER\TS-TrajGen_Dataset\Porto_Taxi\TS_TrajGen_data_archive\pretrain_data'
else:
    data_root = '/mnt/data/jwj/'

dataset_root = get_dataset_root_from_data_root(data_root, dataset_name, local)
save_root = os.path.join(dataset_root, "save", dataset_name)

# -----------------------
# Dataset-specific config
# -----------------------
if dataset_name == 'BJ_Taxi':
    true_traj = pd.read_csv(os.path.join(data_root, dataset_name, 'chaoyang_traj_mm_test.csv'))

    pretrain_gen_file = os.path.join(save_root, 'function_g_fc.pt')
    pretrain_gat_file = os.path.join(save_root, 'gat_fc.pt')
    pretrain_region_gen_file = os.path.join(save_root, 'region_function_g_fc.pt')
    pretrain_region_gat_file = os.path.join(save_root, 'region_gat_fc.pt')

    ganerate_trace_file = os.path.join(data_root, dataset_name, 'TS_TrajGen_chaoyang_generate.csv')

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
            'embed_dim': 256,
            'gps_emb_dim': 10,
            'num_of_heads': 5,
            'concat': False,
            'device': device,
            'distance_mode': 'l2'
        },
        'dis_weight': 0.45
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
            'embed_dim': 128,
            'gps_emb_dim': 5,
            'num_of_heads': 5,
            'concat': False,
            'device': device,
            'distance_mode': 'l2',
            'no_gps_emb': True
        },
        'dis_weight': 0.45
    }

elif dataset_name == 'Porto_Taxi':
    # Porto_Taxi
    true_traj = pd.read_csv(os.path.join(data_root, 'porto_mm_test.csv'))

    # quick sampling for faster evaluation
    if args.max_traj and args.max_traj > 0:
        true_traj = true_traj.sample(n=min(args.max_traj, len(true_traj)),
                                     random_state=args.seed).reset_index(drop=True)
        print(f"[INFO] Using sample: {len(true_traj)} trajectories (seed={args.seed})")

    pretrain_gen_file = os.path.join(save_root, 'function_g_fc.pt')
    pretrain_gat_file = os.path.join(save_root, 'gat_fc.pt')
    pretrain_region_gen_file = os.path.join(save_root, 'region_function_g_fc.pt')
    pretrain_region_gat_file = os.path.join(save_root, 'region_gat_fc.pt')

    # FIX: for Porto, data_root is already .../pretrain_data (no Porto_Taxi subfolder)
    ganerate_trace_file = os.path.join(data_root, 'TS_TrajGen_generate.csv')

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
            'embed_dim': 256,
            'gps_emb_dim': 10,
            'num_of_heads': 5,
            'concat': False,
            'device': device,
            'distance_mode': 'l2'
        },
        'dis_weight': 0.45
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
            'embed_dim': 128,
            'gps_emb_dim': 5,
            'num_of_heads': 5,
            'concat': False,
            'device': device,
            'distance_mode': 'l2',
            'no_gps_emb': True
        },
        'dis_weight': 0.45
    }

else:
    assert dataset_name == 'Xian'
    true_traj = pd.read_csv(os.path.join(data_root, dataset_name, 'xianshi_partA_mm_test.csv'))

    pretrain_gen_file = os.path.join(save_root, 'function_g_fc.pt')
    pretrain_gat_file = os.path.join(save_root, 'gat_fc.pt')
    pretrain_region_gen_file = os.path.join(save_root, 'region_function_g_fc.pt')
    pretrain_region_gat_file = os.path.join(save_root, 'region_gat_fc.pt')

    ganerate_trace_file = os.path.join(data_root, dataset_name, 'TS_TrajGen_generate.csv')

    gen_config = {
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
            'embed_dim': 128,
            'gps_emb_dim': 5,
            'num_of_heads': 4,
            'concat': False,
            'device': device,
            'distance_mode': 'l2'
        },
        'dis_weight': 0.45
    }

    region_gen_config = {
        "function_g": {
            "road_emb_size": 64,
            "time_emb_size": 16,
            "hidden_size": 64,
            "dropout_p": 0.6,
            "lstm_layer_num": 2,
            "pretrain_road_rep": None,
            "dis_weight": 0.5,
            "device": device
        },
        "function_h": {
            'embed_dim': 68,
            'gps_emb_dim': 5,
            'num_of_heads': 4,
            'concat': False,
            'device': device,
            'distance_mode': 'l2',
            'no_gps_emb': True
        },
        'dis_weight': 0.45
    }

# -----------------------
# Map manager
# -----------------------
map_manager = MapManager(
    rid_gps_path=os.path.join(data_root, "porto_rid_gps.json"),
    img_unit=0.005
)

# -----------------------
# Load features & graph
# -----------------------
if dataset_name == 'BJ_Taxi':
    node_feature_file = os.path.join(data_root, archive_data_folder, 'node_feature.pt')
    node_features = torch.load(node_feature_file).to(device)
    adjacent_np_file = os.path.join(data_root, archive_data_folder, 'adjacent_mx.npz')
    adj_mx = sp.load_npz(adjacent_np_file)

    region_adjacent_np_file = os.path.join(data_root, archive_data_folder, 'kaffpa_tarjan_region_adj_mx.npz')
    region_adj_mx = sp.load_npz(region_adjacent_np_file)
    region_feature_file = os.path.join(data_root, archive_data_folder, 'kaffpa_tarjan_region_feature.pt')
    region_features = torch.load(region_feature_file, map_location=device)

    road_num = 40306
    time_size = 2880
    loc_pad = road_num
    time_pad = time_size
    lon_range = 0.2507
    lat_range = 0.21
    img_unit = 0.005
    lon_0 = 116.25
    lat_0 = 39.79
    img_width = math.ceil(lon_range / img_unit) + 1
    img_height = math.ceil(lat_range / img_unit) + 1

    data_feature = {
        'road_num': road_num + 1,
        'time_size': time_size + 1,
        'road_pad': loc_pad,
        'time_pad': time_pad,
        'adj_mx': adj_mx,
        'node_features': node_features,
        'img_width': img_width,
        'img_height': img_height
    }

    with open(os.path.join(data_root, archive_data_folder, 'kaffpa_tarjan_region2rid.json'), 'r') as f:
        region2rid = json.load(f)
    region_num = len(region2rid)

    region_data_feature = {
        'road_num': region_num + 1,
        'time_size': time_size + 1,
        'road_pad': region_num,
        'time_pad': time_pad,
        'adj_mx': region_adj_mx,
        'node_features': region_features,
        'img_width': 52,
        'img_height': 43
    }

    with open(os.path.join(data_root, archive_data_folder, 'adjacent_list.json'), 'r') as f:
        adjacent_list = json.load(f)
    with open(os.path.join(data_root, archive_data_folder, 'rid_gps.json'), 'r') as f:
        rid_gps = json.load(f)
    with open(os.path.join(data_root, archive_data_folder, 'road_length.json'), 'r') as f:
        road_length = json.load(f)

    with open(os.path.join(data_root, archive_data_folder, 'kaffpa_tarjan_region_adjacent_list.json'), 'r') as f:
        region_adjacent_list = json.load(f)
    region_dist = np.load(os.path.join(data_root, archive_data_folder, 'kaffpa_tarjan_region_dist.npy'))
    with open(os.path.join(data_root, archive_data_folder, 'kaffpa_tarjan_region_transfer_prob.json'), 'r') as f:
        region_transfer_freq = json.load(f)
    with open(os.path.join(data_root, archive_data_folder, 'kaffpa_tarjan_rid2region.json'), 'r') as f:
        rid2region = json.load(f)

    road_time_distribution = np.load(os.path.join(data_root, archive_data_folder, 'road_time_distribution.npy'))
    region_time_distribution = np.load(os.path.join(
        data_root, archive_data_folder, 'kaffpa_tarjan_region_time_distribution.npy'
    ))

elif dataset_name == 'Porto_Taxi':
    node_feature_file = os.path.join(data_root, 'porto_node_feature.pt')
    node_features = torch.load(node_feature_file).to(device)

    adjacent_np_file = os.path.join(data_root, 'porto_adjacent_mx.npz')
    adj_mx = sp.load_npz(adjacent_np_file)

    region_adjacent_np_file = os.path.join(data_root, 'region_adj_mx.npz')
    region_adj_mx = sp.load_npz(region_adjacent_np_file)

    region_feature_file = os.path.join(data_root, 'region_feature.pt')
    region_features = torch.load(region_feature_file, map_location=device)

    road_num = 11095
    time_size = 2880
    loc_pad = road_num
    time_pad = time_size

    data_feature = {
        'road_num': road_num + 1,
        'time_size': time_size + 1,
        'road_pad': loc_pad,
        'time_pad': time_pad,
        'adj_mx': adj_mx,
        'node_features': node_features,
        'img_height': map_manager.img_height,
        'img_width': map_manager.img_width
    }

    with open(os.path.join(data_root, 'region2rid.json'), 'r') as f:
        region2rid = json.load(f)
    region_num = len(region2rid)

    region_data_feature = {
        'road_num': region_num + 1,
        'time_size': time_size + 1,
        'road_pad': region_num,
        'time_pad': time_pad,
        'adj_mx': region_adj_mx,
        'node_features': region_features,
        'img_height': map_manager.img_height,
        'img_width': map_manager.img_width
    }

    with open(os.path.join(data_root, 'adjacent_list.json'), 'r') as f:
        adjacent_list = json.load(f)
    with open(os.path.join(data_root, 'porto_rid_gps.json'), 'r') as f:
        rid_gps = json.load(f)
    with open(os.path.join(data_root, 'porto_road_length.json'), 'r') as f:
        road_length = json.load(f)

    with open(os.path.join(data_root, 'region_adjacent_list.json'), 'r') as f:
        region_adjacent_list = json.load(f)
    region_dist = np.load(os.path.join(data_root, 'region_count_dist.npy'))
    with open(os.path.join(data_root, 'region_transfer_prob.json'), 'r') as f:
        region_transfer_freq = json.load(f)
    with open(os.path.join(data_root, 'rid2region.json'), 'r') as f:
        rid2region = json.load(f)

    road_time_distribution = np.load(os.path.join(data_root, 'road_time_distribution.npy'))
    region_time_distribution = np.load(os.path.join(data_root, 'region_time_distribution.npy'))

else:
    node_feature_file = os.path.join(data_root, archive_data_folder, dataset_name, 'node_feature.pt')
    node_features = torch.load(node_feature_file).to(device)

    adjacent_np_file = os.path.join(data_root, archive_data_folder, dataset_name, 'adjacent_mx.npz')
    adj_mx = sp.load_npz(adjacent_np_file)

    region_adjacent_np_file = os.path.join(data_root, archive_data_folder, dataset_name, 'region_adj_mx.npz')
    region_adj_mx = sp.load_npz(region_adjacent_np_file)

    region_feature_file = os.path.join(data_root, archive_data_folder, dataset_name, 'region_feature.pt')
    region_features = torch.load(region_feature_file, map_location=device)

    road_num = 17378
    time_size = 2880
    loc_pad = road_num
    time_pad = time_size

    data_feature = {
        'road_num': road_num + 1,
        'time_size': time_size + 1,
        'road_pad': loc_pad,
        'time_pad': time_pad,
        'adj_mx': adj_mx,
        'node_features': node_features,
        'img_height': map_manager.img_height,
        'img_width': map_manager.img_width
    }

    with open(os.path.join(data_root, archive_data_folder, dataset_name, 'region2rid.json'), 'r') as f:
        region2rid = json.load(f)
    region_num = len(region2rid)

    region_data_feature = {
        'road_num': region_num + 1,
        'time_size': time_size + 1,
        'road_pad': region_num,
        'time_pad': time_pad,
        'adj_mx': region_adj_mx,
        'node_features': region_features,
        'img_height': map_manager.img_height,
        'img_width': map_manager.img_width
    }

    with open(os.path.join(data_root, archive_data_folder, dataset_name, 'adjacent_list.json'), 'r') as f:
        adjacent_list = json.load(f)
    with open(os.path.join(data_root, archive_data_folder, dataset_name, 'rid_gps.json'), 'r') as f:
        rid_gps = json.load(f)
    with open(os.path.join(data_root, archive_data_folder, dataset_name, 'road_length.json'), 'r') as f:
        road_length = json.load(f)

    with open(os.path.join(data_root, archive_data_folder, dataset_name, 'region_adjacent_list.json'), 'r') as f:
        region_adjacent_list = json.load(f)
    region_dist = np.load(os.path.join(data_root, archive_data_folder, dataset_name, 'region_count_dist.npy'))
    with open(os.path.join(data_root, archive_data_folder, dataset_name, 'region_transfer_prob.json'), 'r') as f:
        region_transfer_freq = json.load(f)
    with open(os.path.join(data_root, archive_data_folder, dataset_name, 'rid2region.json'), 'r') as f:
        rid2region = json.load(f)

    road_time_distribution = np.load(os.path.join(
        data_root, archive_data_folder, dataset_name, 'road_time_distribution.npy'
    ))
    region_time_distribution = np.load(os.path.join(
        data_root, archive_data_folder, dataset_name, 'region_time_distribution.npy'
    ))

# -----------------------
# Init generators + load weights
# -----------------------
road_generator = GeneratorV4(config=gen_config, data_feature=data_feature).to(device)
region_generator = GeneratorV4(config=region_gen_config, data_feature=region_data_feature).to(device)

road_gan_ckpt = os.path.join(save_root, "our_gan", "adversarial_3_generator_1.pt")
region_gan_ckpt = os.path.join(save_root, "our_region_gan", "adversarial_region_generator.pt")

loaded_gan = False
if use_gan and os.path.exists(road_gan_ckpt) and os.path.exists(region_gan_ckpt):
    print(f"[INFO] Loading GAN-trained generators:\n  road:   {road_gan_ckpt}\n  region: {region_gan_ckpt}")
    road_generator.load_state_dict(torch.load(road_gan_ckpt, map_location=device))
    region_generator.load_state_dict(torch.load(region_gan_ckpt, map_location=device))
    loaded_gan = True
else:
    print("[WARN] GAN ckpts not found (or --use_gan=False). Falling back to pretrained function_g/function_h.")
    print(f"  expected road ckpt:   {road_gan_ckpt}")
    print(f"  expected region ckpt: {region_gan_ckpt}")

    road_generator.function_g.load_state_dict(torch.load(pretrain_gen_file, map_location=device))
    road_generator.function_h.load_state_dict(torch.load(pretrain_gat_file, map_location=device))
    region_generator.function_g.load_state_dict(torch.load(pretrain_region_gen_file, map_location=device))
    region_generator.function_h.load_state_dict(torch.load(pretrain_region_gat_file, map_location=device))

road_generator.eval()
region_generator.eval()

try:
    road_generator.update_node_emb()
except Exception:
    pass
try:
    region_generator.update_node_emb()
except Exception:
    pass

print("region G dims:",
      region_gen_config["function_g"]["road_emb_size"],
      region_gen_config["function_g"]["time_emb_size"],
      region_gen_config["function_g"]["hidden_size"])
print(f"[INFO] Using GAN weights: {loaded_gan}")

# Ensure output folder exists
out_dir = os.path.dirname(ganerate_trace_file) or "."
os.makedirs(out_dir, exist_ok=True)

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

# -----------------------
# Generate & save
# -----------------------
max_step = args.max_step  # <-- FIX: define max_step variable

f = open(ganerate_trace_file, 'w', encoding='utf-8')
f.write("traj_id,rid_list,time_list\n")

fail_cnt = 0
region_astar_fail_cnt = 0

for index, row in tqdm(true_traj.iterrows(), total=true_traj.shape[0]):
    rid_list = [int(i) for i in row['rid_list'].split(',')]
    mm_id = row['traj_id']
    time_list = list(map(encode_time, row['time_list'].split(',')))

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

    f.write('{},\"{}\",\"{}\"\n'.format(
        str(mm_id),
        ','.join([str(rid) for rid in gen_trace_loc]),
        ','.join([str(time) for time in gen_trace_tim])
    ))

    if gen_trace_loc[-1] != rid_list[-1]:
        fail_cnt += 1
    if is_astar == 0:
        region_astar_fail_cnt += 1

print('fail cnt ', fail_cnt)
print('region astar fail cnt ', region_astar_fail_cnt)

f.close()
searcher.save_fail_log()
