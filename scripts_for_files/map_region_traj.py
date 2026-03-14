import os
import json

import pandas as pd
from tqdm import tqdm

# Adapted from the original TS-TrajGen preprocessing workflow.
# Simplified for the Porto_Taxi thesis experiment only.
# Replace the path below with your own local path before running.

# =========================================================
# PATH CONFIGURATION
# =========================================================
base_path = r"PASTE_YOUR_PRETRAIN_DATA_PATH_HERE"

rid2region_path = os.path.join(base_path, "rid2region.json")
region_adjacent_path = os.path.join(base_path, "region_adjacent_list.json")  # kept for consistency
train_mm_traj_path = os.path.join(base_path, "porto_mm_train.csv")
test_mm_traj_path = os.path.join(base_path, "porto_mm_test.csv")

train_output_path = os.path.join(base_path, "porto_taxi_mm_region_train.csv")
eval_output_path = os.path.join(base_path, "porto_taxi_mm_region_eval.csv")
test_output_path = os.path.join(base_path, "porto_taxi_mm_region_test.csv")

# =========================================================
# LOAD INPUT DATA
# =========================================================
with open(rid2region_path, "r", encoding="utf-8") as f:
    rid2region = json.load(f)

# Loaded for consistency with the original workflow
with open(region_adjacent_path, "r", encoding="utf-8") as f:
    region_adjacent_list = json.load(f)

train_mm_traj = pd.read_csv(train_mm_traj_path)
test_mm_traj = pd.read_csv(test_mm_traj_path)

train_file = open(train_output_path, "w", encoding="utf-8")
eval_file = open(eval_output_path, "w", encoding="utf-8")
test_file = open(test_output_path, "w", encoding="utf-8")

header = "traj_id,region_list,time_list\n"
train_file.write(header)
eval_file.write(header)
test_file.write(header)


def write_row(output_file, row, region_list, time_list):
    """
    Write a mapped region-level trajectory row.

    Args:
        output_file: Open file handle for writing.
        row: Original dataframe row.
        region_list: Region sequence.
        time_list: Corresponding timestamp sequence.
    """
    traj_id = row["traj_id"]
    mapped_region_str = ",".join([str(x) for x in region_list])
    mapped_time_str = ",".join(time_list)
    output_file.write(f'{traj_id},"{mapped_region_str}","{mapped_time_str}"\n')


train_rate = 0.9
total_data_num = train_mm_traj.shape[0]
train_num = int(total_data_num * train_rate)

for index, row in tqdm(train_mm_traj.iterrows(), total=train_mm_traj.shape[0], desc="Mapping train trajectories"):
    rid_list = row["rid_list"].split(",")
    time_list = row["time_list"].split(",")

    start_region = rid2region[rid_list[0]]
    start_time = time_list[0]

    mapped_region_list = [start_region]
    mapped_time_list = [start_time]

    for j, rid in enumerate(rid_list[1:]):
        mapped_region = rid2region[rid]
        if mapped_region != mapped_region_list[-1]:
            mapped_region_list.append(mapped_region)
            mapped_time_list.append(time_list[j + 1])

    if index <= train_num:
        write_row(train_file, row, mapped_region_list, mapped_time_list)
    else:
        write_row(eval_file, row, mapped_region_list, mapped_time_list)

for _, row in tqdm(test_mm_traj.iterrows(), total=test_mm_traj.shape[0], desc="Mapping test trajectories"):
    rid_list = row["rid_list"].split(",")
    time_list = row["time_list"].split(",")

    start_region = rid2region[rid_list[0]]
    start_time = time_list[0]

    mapped_region_list = [start_region]
    mapped_time_list = [start_time]

    for j, rid in enumerate(rid_list[1:]):
        mapped_region = rid2region[rid]
        if mapped_region != mapped_region_list[-1]:
            mapped_region_list.append(mapped_region)
            mapped_time_list.append(time_list[j + 1])

    write_row(test_file, row, mapped_region_list, mapped_time_list)

train_file.close()
eval_file.close()
test_file.close()