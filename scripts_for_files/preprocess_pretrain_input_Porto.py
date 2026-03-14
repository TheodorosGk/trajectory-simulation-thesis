import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from geopy import distance
from shapely.geometry import LineString

# Adapted from the original TS-TrajGen preprocessing workflow.
# Simplified for the Porto_Taxi thesis experiment only.
# Replace the path below with your own local path before running.

max_step = 4
random_encode = True  # Random-step encoding reduces data volume and may help avoid overfitting.

# =========================================================
# PATH CONFIGURATION
# =========================================================
base_path = r"PASTE_YOUR_PRETRAIN_DATA_PATH_HERE"

train_data_path = os.path.join(base_path, "porto_mm_train.csv")
test_data_path = os.path.join(base_path, "porto_mm_test.csv")

adjacent_file = os.path.join(base_path, "adjacent_list.json")
rid_gps_file = os.path.join(base_path, "porto_rid_gps.json")

train_output_path = os.path.join(base_path, "porto_taxi_pretrain_input_train.csv")
eval_output_path = os.path.join(base_path, "porto_taxi_pretrain_input_eval.csv")
test_output_path = os.path.join(base_path, "porto_taxi_pretrain_input_test.csv")

# =========================================================
# LOAD TRAIN / TEST DATA
# =========================================================
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# =========================================================
# LOAD OR BUILD ROAD ADJACENCY LIST
# =========================================================
if os.path.exists(adjacent_file):
    with open(adjacent_file, "r", encoding="utf-8") as f:
        adjacent_list = json.load(f)
else:
    rid_rel = pd.read_csv(os.path.join(base_path, "porto.rel"))
    road_adjacent_list = {}

    for _, row in tqdm(rid_rel.iterrows(), total=rid_rel.shape[0], desc="Building road adjacency list"):
        from_rid = str(row["origin_id"])
        to_rid = row["destination_id"]

        if from_rid not in road_adjacent_list:
            road_adjacent_list[from_rid] = [to_rid]
        else:
            road_adjacent_list[from_rid].append(to_rid)

    with open(adjacent_file, "w", encoding="utf-8") as f:
        json.dump(road_adjacent_list, f)

    adjacent_list = road_adjacent_list

# =========================================================
# LOAD OR BUILD ROAD GPS DICTIONARY
# =========================================================
if os.path.exists(rid_gps_file):
    with open(rid_gps_file, "r", encoding="utf-8") as f:
        rid_gps = json.load(f)
else:
    rid_gps = {}
    rid_info = pd.read_csv(os.path.join(base_path, "porto.geo"))

    for _, row in tqdm(rid_info.iterrows(), total=rid_info.shape[0], desc="Building road GPS dictionary"):
        rid = row["geo_id"]
        coordinate = eval(row["coordinates"])
        road_line = LineString(coordinates=coordinate)
        center = road_line.centroid
        rid_gps[str(rid)] = (center.x, center.y)

    with open(rid_gps_file, "w", encoding="utf-8") as f:
        json.dump(rid_gps, f)


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
    Encode one road-level trajectory into pretraining samples.

    Args:
        trace: One trajectory record.
        output_file: Open file handle for writing encoded results.
    """
    rid_list = [int(i) for i in trace["rid_list"].split(",")]
    time_list = [encode_time(i) for i in trace["time_list"].split(",")]

    destination = rid_list[-1]
    destination_gps = rid_gps[str(destination)]

    i = 1
    while i < len(rid_list):
        cur_loc = rid_list[:i]
        cur_time = time_list[:i]
        cur_rid = cur_loc[-1]

        if str(cur_rid) not in adjacent_list or rid_list[i] not in adjacent_list[str(cur_rid)]:
            # Path discontinuity detected; discard the rest of the trajectory.
            return

        candidate_set = adjacent_list[str(cur_rid)]

        if len(candidate_set) > 1:
            target = rid_list[i]
            target_index = 0
            candidate_dis = []

            for index, candidate in enumerate(candidate_set):
                if candidate == target:
                    target_index = index

                candidate_gps = rid_gps[str(candidate)]
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
    train_rate = 0.9
    total_data_num = train_data.shape[0]
    train_num = int(total_data_num * train_rate)

    train_output = open(train_output_path, "w", encoding="utf-8")
    eval_output = open(eval_output_path, "w", encoding="utf-8")
    test_output = open(test_output_path, "w", encoding="utf-8")

    header = "trace_loc,trace_time,des,candidate_set,candidate_dis,target\n"
    train_output.write(header)
    eval_output.write(header)
    test_output.write(header)

    for index, row in tqdm(train_data.iterrows(), total=train_data.shape[0], desc="Encoding train trajectories"):
        if index <= train_num:
            encode_trace(row, train_output)
        else:
            encode_trace(row, eval_output)

    for _, row in tqdm(test_data.iterrows(), total=test_data.shape[0], desc="Encoding test trajectories"):
        encode_trace(row, test_output)

    train_output.close()
    eval_output.close()
    test_output.close()
