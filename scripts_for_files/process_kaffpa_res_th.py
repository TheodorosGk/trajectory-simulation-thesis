import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

# Adapted for thesis documentation purposes.
# Simplified for the Porto_Taxi thesis experiment only.
# Replace the path below with your own local path before running.

# =========================================================
# PATH CONFIGURATION
# =========================================================
base_path = r"PASTE_YOUR_TS_TRAJGEN_ARCHIVE_PATH_HERE"

geo_path = os.path.join(base_path, "pretrain_data", "porto.geo")
rel_path = os.path.join(base_path, "pretrain_data", "porto.rel")

# Output file produced by KaHIP / kaffpa
partition_path = os.path.join(base_path, "Porto_Taxi", "porto.part100")

# Output JSON files
out_dir = os.path.join(base_path, "pretrain_data")
os.makedirs(out_dir, exist_ok=True)

rid2region_out = os.path.join(out_dir, "rid2region.json")
region2rid_out = os.path.join(out_dir, "region2rid.json")

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
    from_id = int(row["origin_id"])
    to_id = int(row["destination_id"])
    outlier_set.discard(from_id)
    outlier_set.discard(to_id)

print("Isolated road segments:", len(outlier_set))

# =========================================================
# REINDEX ROAD IDS
# KaHIP works on consecutive node ids after removing outliers
# =========================================================
rid2new = {}
new2rid = {}
new_id = 1

for rid in range(total_road_num):
    if rid not in outlier_set:
        rid2new[rid] = new_id
        new2rid[new_id] = rid
        new_id += 1

num_partitioned_nodes = len(new2rid)
print("Nodes used in partition:", num_partitioned_nodes)

# =========================================================
# READ PARTITION FILE
# kaffpa output format: one partition id per line
# =========================================================
rid2region = {}
region2rid = {}

with open(partition_path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip() != ""]

if len(lines) != num_partitioned_nodes:
    raise ValueError(
        f"Partition lines ({len(lines)}) != nodes used in partition ({num_partitioned_nodes}). "
        f"Check partition file and path."
    )

for new_road_id in range(1, num_partitioned_nodes + 1):
    region_id = int(lines[new_road_id - 1])
    original_rid = new2rid[new_road_id]

    rid2region[str(original_rid)] = region_id
    region2rid.setdefault(str(region_id), []).append(original_rid)

# =========================================================
# BASIC CHECKS
# =========================================================
num_regions = len(region2rid)
print("Regions:", num_regions)
print("rid2region size:", len(rid2region))

region_sizes = np.array([len(v) for v in region2rid.values()], dtype=int)
print(
    f"Average roads per region: {region_sizes.mean():.3f}, "
    f"min: {region_sizes.min()}, max: {region_sizes.max()}"
)

# =========================================================
# SAVE OUTPUT
# JSON keys are stored as strings
# =========================================================
with open(rid2region_out, "w", encoding="utf-8") as f:
    json.dump(rid2region, f)

with open(region2rid_out, "w", encoding="utf-8") as f:
    json.dump(region2rid, f)

print("Saved:", rid2region_out)
print("Saved:", region2rid_out)