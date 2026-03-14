import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

# Adapted for thesis documentation purposes.
# Replace the paths below with your own local paths before running.

base_path = r"PASTE_YOUR_PRETRAIN_DATA_PATH_HERE"

road_len_file = os.path.join(base_path, "road_length.json")
road_info_file = os.path.join(base_path, "porto.geo")
rid2region_file = os.path.join(base_path, "rid2region.json")
traj_file = os.path.join(base_path, "PASTE_YOUR_TRAJECTORY_FILE_HERE.csv")
region_gps_file = os.path.join(base_path, "porto_region_gps.json")
output_file = os.path.join(base_path, "region_count_dist.npy")

# distance_dict[from_region][to_region] = (total_distance, count)
# Used to compute the average travel distance between region pairs
distance_dict = {}

# =========================================================
# LOAD OR BUILD ROAD LENGTH DICTIONARY
# =========================================================
if not os.path.exists(road_len_file):
    road_info = pd.read_csv(road_info_file)
    road_length = {}

    for _, row in tqdm(road_info.iterrows(), desc="Building road length dictionary", total=road_info.shape[0]):
        rid = row["geo_id"]
        length = row["length"]
        road_length[str(rid)] = length

    with open(road_len_file, "w", encoding="utf-8") as f:
        json.dump(road_length, f)
else:
    with open(road_len_file, "r", encoding="utf-8") as f:
        road_length = json.load(f)

# =========================================================
# LOAD ROAD-TO-REGION MAPPING
# =========================================================
with open(rid2region_file, "r", encoding="utf-8") as f:
    rid2region = json.load(f)

# =========================================================
# PROCESS TRAJECTORIES
# =========================================================
traj = pd.read_csv(traj_file)

for _, row in tqdm(traj.iterrows(), desc="Counting trajectory distances", total=traj.shape[0]):
    rid_list = [int(i) for i in row["rid_list"].split(",")]

    if len(rid_list) < 2:
        continue

    cumulative_length = 0
    step_length = []

    for rid in rid_list:
        # road_length is assumed to be in meters
        cumulative_length += road_length[str(rid)]
        step_length.append(cumulative_length)

    for i in range(len(rid_list)):
        from_rid = rid_list[i]
        from_region = int(rid2region[str(from_rid)])

        for j in range(i + 1, len(rid_list)):
            to_rid = rid_list[j]
            to_region = int(rid2region[str(to_rid)])
            travel_length = step_length[j] - step_length[i]

            if to_region != from_region:
                if from_region not in distance_dict:
                    distance_dict[from_region] = {to_region: (travel_length, 1)}
                elif to_region not in distance_dict[from_region]:
                    distance_dict[from_region][to_region] = (travel_length, 1)
                else:
                    total_dist, count = distance_dict[from_region][to_region]
                    distance_dict[from_region][to_region] = (total_dist + travel_length, count + 1)

# =========================================================
# BUILD REGION DISTANCE MATRIX
# =========================================================
with open(region_gps_file, "r", encoding="utf-8") as f:
    region_gps = json.load(f)

region_num = len(region_gps)
region_dist = np.zeros((region_num, region_num), dtype=float)

for from_region in tqdm(range(region_num), desc="Generating region distance matrix"):
    for to_region in range(region_num):
        if from_region != to_region:
            if from_region in distance_dict and to_region in distance_dict[from_region]:
                total_dist, count = distance_dict[from_region][to_region]
                avg_length = total_dist / count
                region_dist[from_region][to_region] = avg_length
            else:
                region_dist[from_region][to_region] = -1

np.save(output_file, region_dist)
print(f"Saved region distance matrix to: {output_file}")