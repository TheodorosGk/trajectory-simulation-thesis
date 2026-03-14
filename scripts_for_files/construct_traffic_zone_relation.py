import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp

# Adapted for thesis documentation purposes.
# Simplified for the Porto_Taxi thesis experiment only.
# Replace the path below with your own local path before running.

# =========================================================
# PATH CONFIGURATION
# =========================================================
base_path = r"PASTE_YOUR_PRETRAIN_DATA_PATH_HERE"

rel_path = os.path.join(base_path, "porto.rel")
road_adjacent_out = os.path.join(base_path, "adjacent_list.json")

rid2region_path = os.path.join(base_path, "rid2region.json")
region2rid_path = os.path.join(base_path, "region2rid.json")

region_adj_mx_out = os.path.join(base_path, "region_adj_mx.npz")
region_adjacent_out = os.path.join(base_path, "region_adjacent_list.json")

# =========================================================
# BUILD ROAD ADJACENCY LIST
# =========================================================
rid_rel = pd.read_csv(rel_path)

rid_adjacent_list = {}
for _, row in tqdm(rid_rel.iterrows(), total=rid_rel.shape[0], desc="Building road adjacency list"):
    from_rid = str(row["origin_id"])
    to_rid = row["destination_id"]

    if from_rid not in rid_adjacent_list:
        rid_adjacent_list[from_rid] = [to_rid]
    else:
        rid_adjacent_list[from_rid].append(to_rid)

with open(road_adjacent_out, "w", encoding="utf-8") as f:
    json.dump(rid_adjacent_list, f)

# =========================================================
# LOAD ROAD-TO-REGION AND REGION-TO-ROAD MAPPINGS
# =========================================================
with open(rid2region_path, "r", encoding="utf-8") as f:
    rid2region = json.load(f)

with open(region2rid_path, "r", encoding="utf-8") as f:
    region2rid = json.load(f)

# =========================================================
# BUILD REGION ADJACENCY STRUCTURE
# region_adjacent_list format:
# {
#   current_region: {
#       downstream_region: [boundary road segments in downstream region]
#   }
# }
# =========================================================
region_adjacent_list = {}

# Sparse adjacency matrix components
region_adj_row = []
region_adj_col = []
region_adj_data = []

for region in tqdm(region2rid, desc="Building region adjacency"):
    next_region_dict = {}
    rid_set = region2rid[region]

    for rid in rid_set:
        if str(rid) in rid_adjacent_list:
            for next_rid in rid_adjacent_list[str(rid)]:
                next_region = rid2region[str(next_rid)]

                if int(region) != next_region:
                    if next_region not in next_region_dict:
                        next_region_dict[next_region] = set()
                        next_region_dict[next_region].add(next_rid)

                        # Add one edge to the sparse adjacency matrix
                        region_adj_row.append(int(region))
                        region_adj_col.append(next_region)
                        region_adj_data.append(1.0)
                    else:
                        next_region_dict[next_region].add(next_rid)

    # Convert sets to lists for JSON serialization
    for next_region in next_region_dict:
        next_region_dict[next_region] = list(next_region_dict[next_region])

    region_adjacent_list[region] = next_region_dict

# =========================================================
# BUILD AND SAVE SPARSE REGION ADJACENCY MATRIX
# =========================================================
total_region = len(region2rid)

region_adj_mx = sp.coo_matrix(
    (region_adj_data, (region_adj_row, region_adj_col)),
    shape=(total_region, total_region)
)

sp.save_npz(region_adj_mx_out, region_adj_mx)

with open(region_adjacent_out, "w", encoding="utf-8") as f:
    json.dump(region_adjacent_list, f)

# =========================================================
# BASIC STATISTICS
# =========================================================
adjacent_cnt = []
border_rid_cnt = []

for region in region_adjacent_list:
    adjacent_cnt.append(len(region_adjacent_list[region]))
    for next_region in region_adjacent_list[region]:
        border_rid_cnt.append(len(region_adjacent_list[region][next_region]))

adjacent_cnt = np.array(adjacent_cnt)
border_rid_cnt = np.array(border_rid_cnt)

print(
    "Region adjacency stats -> "
    f"avg: {np.average(adjacent_cnt)}, "
    f"max: {np.max(adjacent_cnt)}, "
    f"min: {np.min(adjacent_cnt)}"
)

print(
    "Boundary road stats -> "
    f"avg: {np.average(border_rid_cnt)}, "
    f"max: {np.max(border_rid_cnt)}, "
    f"min: {np.min(border_rid_cnt)}"
)