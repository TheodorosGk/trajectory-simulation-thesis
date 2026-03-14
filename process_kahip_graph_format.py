import os

import numpy as np
import pandas as pd
from tqdm import tqdm

# Adapted for thesis documentation purposes.
# Simplified for the Porto_Taxi thesis experiment only.
# This script exports the road network in METIS graph format.
# Replace the path below with your own local path before running.

# =========================================================
# PATH CONFIGURATION
# =========================================================
base_path = r"PASTE_YOUR_PRETRAIN_DATA_PATH_HERE"

geo_path = os.path.join(base_path, "porto.geo")
rel_path = os.path.join(base_path, "porto.rel")
graph_output_path = os.path.join(base_path, "porto.graph")

# =========================================================
# LOAD ROAD NETWORK
# =========================================================
road_info = pd.read_csv(geo_path)
road_rel = pd.read_csv(rel_path)

total_road_num = road_info.shape[0]

# =========================================================
# FIND ISOLATED ROAD SEGMENTS
# =========================================================
outlier_set = set(road_info["geo_id"])

for _, row in tqdm(road_rel.iterrows(), total=road_rel.shape[0], desc="Finding isolated road segments"):
    from_id = row["origin_id"]
    to_id = row["destination_id"]

    if from_id in outlier_set:
        outlier_set.remove(from_id)
    if to_id in outlier_set:
        outlier_set.remove(to_id)

print("Isolated road segments:", len(outlier_set))

# =========================================================
# REINDEX ROAD IDS AFTER REMOVING OUTLIERS
# =========================================================
rid2new = {}
new2rid = {}
new_id = 1

for rid in range(total_road_num):
    if rid not in outlier_set:
        rid2new[rid] = new_id
        new2rid[new_id] = rid
        new_id += 1

# =========================================================
# BUILD UNDIRECTED GRAPH FOR METIS / KaHIP
# =========================================================
total_road_num = len(new2rid)
assert total_road_num + 1 == new_id

road_undirected_adj_mx = np.zeros((total_road_num, total_road_num), dtype=int)
road_undirected_rel = {}
total_edge_num = 0

for _, row in tqdm(road_rel.iterrows(), total=road_rel.shape[0], desc="Building undirected graph"):
    from_road = rid2new[row["origin_id"]]
    to_road = rid2new[row["destination_id"]]

    if from_road == to_road:
        continue

    min_road = min(from_road, to_road)
    max_road = max(from_road, to_road)

    if min_road not in road_undirected_rel:
        road_undirected_rel[min_road] = {max_road}
        road_undirected_adj_mx[min_road - 1][max_road - 1] = 1
        road_undirected_adj_mx[max_road - 1][min_road - 1] = 1
        total_edge_num += 1
    elif max_road not in road_undirected_rel[min_road]:
        road_undirected_rel[min_road].add(max_road)
        road_undirected_adj_mx[min_road - 1][max_road - 1] = 1
        road_undirected_adj_mx[max_road - 1][min_road - 1] = 1
        total_edge_num += 1

# =========================================================
# WRITE GRAPH IN METIS FORMAT
# =========================================================
with open(graph_output_path, "w", encoding="utf-8") as f:
    f.write(f"{total_road_num} {total_edge_num}\n")
    print("Nodes:", total_road_num, "| Edges:", total_edge_num)

    output_cnt = 0
    for road_id in range(1, total_road_num + 1):
        road_adjacent = (np.where(road_undirected_adj_mx[road_id - 1] == 1)[0] + 1).tolist()
        output_cnt += len(road_adjacent)
        adjacent_str = " ".join([str(x) for x in road_adjacent])
        f.write(adjacent_str + "\n")

print("Total adjacency entries written:", output_cnt)
print(f"Saved graph to: {graph_output_path}")