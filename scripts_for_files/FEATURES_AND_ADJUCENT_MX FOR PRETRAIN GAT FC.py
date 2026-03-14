import os
import json
import math

import torch
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

# Adapted for thesis documentation purposes.
# Simplified for the Porto_Taxi thesis experiment only.
# Replace the path below with your own local path before running.

# =========================================================
# PATH CONFIGURATION
# =========================================================
base_path = r"PASTE_YOUR_PRETRAIN_DATA_PATH_HERE"

geo_path = os.path.join(base_path, "porto.geo")
rel_path = os.path.join(base_path, "porto.rel")
rid_gps_path = os.path.join(base_path, "porto_rid_gps.json")

adjacent_mx_out = os.path.join(base_path, "adjacent_mx.npz")
node_feature_out = os.path.join(base_path, "node_feature.pt")

# Number of road segments in the processed Porto road network
road_num = 11095
road_num_with_pad = road_num + 1


class MapManager:
    """
    Utility class for converting GPS coordinates to coarse grid coordinates.
    """

    def __init__(self, rid_gps_path, img_unit=0.005):
        with open(rid_gps_path, "r", encoding="utf-8") as f:
            rid_gps = json.load(f)

        lons = [v[0] for v in rid_gps.values()]
        lats = [v[1] for v in rid_gps.values()]

        self.lon_0 = min(lons)
        self.lon_1 = max(lons)
        self.lat_0 = min(lats)
        self.lat_1 = max(lats)

        self.img_unit = img_unit

        self.img_width = math.ceil((self.lon_1 - self.lon_0) / self.img_unit) + 1
        self.img_height = math.ceil((self.lat_1 - self.lat_0) / self.img_unit) + 1

    def gps2grid(self, lon, lat):
        x = math.floor((lon - self.lon_0) / self.img_unit)
        y = math.floor((lat - self.lat_0) / self.img_unit)

        x = min(max(x, 0), self.img_width)
        y = min(max(y, 0), self.img_height)

        return x, y


# =========================================================
# STEP 1: BUILD ROAD ADJACENCY MATRIX
# =========================================================
print("Loading road relation file...")
road_rel = pd.read_csv(rel_path)

adj_row = []
adj_col = []
adj_data = []
adj_set = set()

for _, row in tqdm(road_rel.iterrows(), total=road_rel.shape[0], desc="Building adjacency matrix"):
    from_id = row["origin_id"]
    to_id = row["destination_id"]

    if (from_id, to_id) not in adj_set:
        adj_set.add((from_id, to_id))
        adj_row.append(from_id)
        adj_col.append(to_id)
        adj_data.append(1.0)

adj_mx = sp.coo_matrix(
    (adj_data, (adj_row, adj_col)),
    shape=(road_num_with_pad, road_num_with_pad)
)

sp.save_npz(adjacent_mx_out, adj_mx)

print(f"Saved adjacency matrix to: {adjacent_mx_out}")
print(f"Adjacency matrix shape: {adj_mx.shape}")
print(f"Number of non-zero entries: {adj_mx.nnz}")


# =========================================================
# STEP 2: BUILD NODE FEATURES
# =========================================================
print("Loading road geometry and GPS information...")
road_info = pd.read_csv(geo_path)

with open(rid_gps_path, "r", encoding="utf-8") as f:
    rid_gps = json.load(f)

map_manager = MapManager(rid_gps_path=rid_gps_path, img_unit=0.005)

# Keep only the features used in the thesis preprocessing
feature_cols = [
    "highway",
    "lanes",
    "tunnel",
    "bridge",
    "oneway",
    "roundabout",
    "length",
    "maxspeed",
]

node_features = road_info[feature_cols].copy()

# Normalize continuous features
for col in ["length", "maxspeed"]:
    min_val = node_features[col].min()
    max_val = node_features[col].max()
    node_features[col] = (node_features[col] - min_val) / (max_val - min_val + 1e-6)

# Add spatial grid features
lon_grid = []
lat_grid = []

for i in range(len(node_features)):
    lon, lat = rid_gps[str(i)]
    x, y = map_manager.gps2grid(lon=lon, lat=lat)
    lon_grid.append(x)
    lat_grid.append(y)

node_features["lon_grid"] = lon_grid
node_features["lat_grid"] = lat_grid

# Convert to tensor
node_features = torch.tensor(node_features.values, dtype=torch.float)

# Add one padding row so that node_feature shape matches road_num + 1
pad = torch.zeros(1, node_features.shape[1], dtype=node_features.dtype)
node_features = torch.cat([node_features, pad], dim=0)

torch.save(node_features, node_feature_out)

print(f"Saved node features to: {node_feature_out}")
print(f"Node feature shape: {node_features.shape}")