import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from geopy import distance

# Adapted from the original TS-TrajGen preprocessing workflow.
# Simplified for the Porto_Taxi thesis experiment only.
# Replace the path below with your own local path before running.

max_step = 4

# =========================================================
# PATH CONFIGURATION
# =========================================================
base_path = r"PASTE_YOUR_PRETRAIN_DATA_PATH_HERE"

rid2region_path = os.path.join(base_path, "rid2region.json")
region_adjacent_path = os.path.join(base_path, "region_adjacent_list.json")
region_gps_path = os.path.join(base_path, "porto_region_gps.json")

train_file = os.path.join(base_path, "porto_taxi_mm_region_train.csv")
eval_file = os.path.join(base_path, "porto_taxi_mm_region_eval.csv")
test_file = os.path.join(base_path, "porto_taxi_mm_region_test.csv")

train_output_path = os.path.join(base_path, "porto_taxi_region_pretrain_input_train.csv")
eval_output_path = os.path.join(base_path, "porto_taxi_region_pretrain_input_eval.csv")
test_output_path = os.path.join(base_path, "porto_taxi_region_pretrain_input_test.csv")

# =========================================================
# LOAD DATA
# =========================================================
with open(rid2region_path, "r", encoding="utf-8") as f:
    rid2region = json.load(f)

with open(region_adjacent_path, "r", encoding="utf-8") as f:
    region_adjacent_list = json.load(f)

with open(region_gps_path, "r", encoding="utf-8") as f:
    region_gps = json.load(f)

train_data = pd.read_csv(train_file)
eval_data = pd.read_csv(eval_file)
test_data = pd.read_csv(test_file)


def encode_time(timestamp):
    """
    Encode timestamp into minute-of-day representation.
    Weekend timestamps are shifted by +1440 to distinguish them from weekdays.
    """
    time_obj = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    if time_obj.weekday() in [5, 6]:
        return time_obj.hour * 60 + time_obj.minute + 1440
    return time_obj.hour * 60 + time_obj.minute


def encode_trace(trace, output_file):
    """
    Encode one region-level trajectory into pretraining samples.

    Args:
        trace: One trajectory record.
        output_file: Open file handle for writing encoded results.
    """
    region_list = [int(i) for i in trace["region_list"].split(",")]
    time_list = [encode_time(i) for i in trace["time_list"].split(",")]

    destination = region_list[-1]
    destination_gps = region_gps[str(destination)]

    i = 1
    while i < len(region_list):
        cur_loc = region_list[:i]
        cur_time = time_list[:i]
        cur_region = cur_loc[-1]

        if str(cur_region) not in region_adjacent_list or str(region_list[i]) not in region_adjacent_list[str(cur_region)]:
            # Path discontinuity detected; discard the rest of the trajectory.
            return

        candidate_set = list(region_adjacent_list[str(cur_region)].keys())

        if len(candidate_set) > 1:
            target = str(region_list[i])
            target_index = 0
            candidate_dis = []

            for index, candidate in enumerate(candidate_set):
                if candidate == target:
                    target_index = index

                candidate_gps = region_gps[candidate]
                dist_scaled = distance.distance(
                    (destination_gps[1], destination_gps[0]),
                    (candidate_gps[1], candidate_gps[0]),
                ).kilometers * 10  # unit: 100 meters
                candidate_dis.append(dist_scaled)

            cur_loc_str = ",".join([str(x) for x in cur_loc])
            cur_time_str = ",".join([str(x) for x in cur_time])
            candidate_set_str = ",".join([str(x) for x in candidate_set])
            candidate_dis_str = ",".join([str(x) for x in candidate_dis])

            output_file.write(
                "\"{}\",\"{}\",{},\"{}\",\"{}\",{}\n".format(
                    cur_loc_str,
                    cur_time_str,
                    destination,
                    candidate_set_str,
                    candidate_dis_str,
                    target_index,
                )
            )

        # Advance by a random number of steps
        step = np.random.randint(1, max_step)
        i += step


if __name__ == "__main__":
    train_output = open(train_output_path, "w", encoding="utf-8")
    eval_output = open(eval_output_path, "w", encoding="utf-8")
    test_output = open(test_output_path, "w", encoding="utf-8")

    header = "trace_loc,trace_time,des,candidate_set,candidate_dis,target\n"
    train_output.write(header)
    eval_output.write(header)
    test_output.write(header)

    for _, row in tqdm(train_data.iterrows(), total=train_data.shape[0], desc="Encoding train trajectories"):
        encode_trace(row, train_output)

    for _, row in tqdm(eval_data.iterrows(), total=eval_data.shape[0], desc="Encoding eval trajectories"):
        encode_trace(row, eval_output)

    for _, row in tqdm(test_data.iterrows(), total=test_data.shape[0], desc="Encoding test trajectories"):
        encode_trace(row, test_output)

    train_output.close()
    eval_output.close()
    test_output.close()