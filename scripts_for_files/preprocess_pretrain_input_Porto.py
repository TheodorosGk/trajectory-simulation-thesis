import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from geopy import distance
from shapely.geometry import LineString


def str2bool(value):
    """Convert common string values to boolean."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true"):
        return True
    if value.lower() in ("no", "false"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


# =========================================================
# USER CONFIGURATION
# =========================================================
local = True
dataset_name = "Porto_Taxi"
max_step = 4
random_encode = True  # Random step encoding reduces data volume and may help avoid overfitting.

# Paste your dataset root path here
# Example: data_root = r"C:\path\to\TS-TrajGen_Dataset"
data_root = r"PASTE_YOUR_DATASET_ROOT_PATH_HERE"

# Paste your output path here
# Example: out_dir = r"C:\path\to\Porto_Taxi\pretrain_data"
out_dir = r"PASTE_YOUR_OUTPUT_PATH_HERE"


# =========================================================
# LOAD TRAIN / TEST DATA
# =========================================================
if dataset_name == "BJ_Taxi":
    train_data = pd.read_csv(os.path.join(data_root, "BJ-Taxi", "BJ-Taxi_mm_train_str.csv"))
    test_data = pd.read_csv(os.path.join(data_root, "BJ-Taxi", "BJ-Taxi_mm_test_str.csv"))

elif dataset_name == "Porto_Taxi":
    train_data = pd.read_csv(os.path.join(data_root, "Porto_Taxi", "porto_mm_train.csv"))
    test_data = pd.read_csv(os.path.join(data_root, "Porto_Taxi", "porto_mm_test.csv"))

else:
    # Xian
    train_data = pd.read_csv(os.path.join(data_root, dataset_name, "xianshi_partA_mm_train.csv"))
    test_data = pd.read_csv(os.path.join(data_root, dataset_name, "xianshi_partA_mm_test.csv"))


# =========================================================
# LOAD OR BUILD ROAD ADJACENCY LIST
# =========================================================
if dataset_name == "BJ_Taxi":
    adjacent_file = os.path.join(data_root, "BJ-Taxi JSON File", "adjacent_list.json")
    with open(adjacent_file, "r") as f:
        adjacent_list = json.load(f)

elif dataset_name == "Porto_Taxi":
    adjacent_file = os.path.join(data_root, "Porto_Taxi", "adjacent_list.json")

    if os.path.exists(adjacent_file):
        with open(adjacent_file, "r") as f:
            adjacent_list = json.load(f)
    else:
        rid_rel = pd.read_csv(os.path.join(data_root, "Porto_Taxi", "porto.rel"))
        road_adjacent_list = {}

        for _, row in tqdm(rid_rel.iterrows(), total=rid_rel.shape[0], desc="Building road adjacency list"):
            from_rid = str(row["origin_id"])
            to_rid = row["destination_id"]

            if from_rid not in road_adjacent_list:
                road_adjacent_list[from_rid] = [to_rid]
            else:
                road_adjacent_list[from_rid].append(to_rid)

        with open(adjacent_file, "w") as f:
            json.dump(road_adjacent_list, f)

        adjacent_list = road_adjacent_list

else:
    adjacent_file = os.path.join(data_root, dataset_name, "adjacent_list.json")

    if os.path.exists(adjacent_file):
        with open(adjacent_file, "r") as f:
            adjacent_list = json.load(f)
    else:
        rid_rel = pd.read_csv(os.path.join(data_root, dataset_name, "porto.rel"))
        road_adjacent_list = {}

        for _, row in tqdm(rid_rel.iterrows(), total=rid_rel.shape[0], desc="Building road adjacency list"):
            from_rid = str(row["origin_id"])
            to_rid = row["destination_id"]

            if from_rid not in road_adjacent_list:
                road_adjacent_list[from_rid] = [to_rid]
            else:
                road_adjacent_list[from_rid].append(to_rid)

        with open(adjacent_file, "w") as f:
            json.dump(road_adjacent_list, f)

        adjacent_list = road_adjacent_list


# =========================================================
# LOAD OR BUILD ROAD GPS DICTIONARY
# =========================================================
if dataset_name == "BJ_Taxi":
    rid_gps_file = os.path.join(data_root, "BJ-Taxi JSON File", "rid_gps.json")
    with open(rid_gps_file, "r") as f:
        rid_gps = json.load(f)

elif dataset_name == "Porto_Taxi":
    rid_gps_file = os.path.join(data_root, "Porto_Taxi", "porto_rid_gps.json")

    if os.path.exists(rid_gps_file):
        with open(rid_gps_file, "r") as f:
            rid_gps = json.load(f)
    else:
        rid_gps = {}
        rid_info = pd.read_csv(os.path.join(data_root, "Porto_Taxi", "porto.geo"))

        for _, row in tqdm(rid_info.iterrows(), total=rid_info.shape[0], desc="Building road GPS dictionary"):
            rid = row["geo_id"]
            coordinate = eval(row["coordinates"])
            road_line = LineString(coordinates=coordinate)
            center = road_line.centroid
            rid_gps[str(rid)] = (center.x, center.y)

        with open(rid_gps_file, "w") as f:
            json.dump(rid_gps, f)

else:
    rid_gps_file = os.path.join(data_root, dataset_name, "rid_gps.json")

    if os.path.exists(rid_gps_file):
        with open(rid_gps_file, "r") as f:
            rid_gps = json.load(f)
    else:
        rid_gps = {}
        rid_info = pd.read_csv(os.path.join(data_root, dataset_name, "porto.geo"))

        for _, row in tqdm(rid_info.iterrows(), total=rid_info.shape[0], desc="Building road GPS dictionary"):
            rid = row["geo_id"]
            coordinate = eval(row["coordinates"])
            road_line = LineString(coordinates=coordinate)
            center = road_line.centroid
            rid_gps[str(rid)] = (center.x, center.y)

        with open(rid_gps_file, "w") as f:
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
    Encode a single trajectory record into pretraining samples.

    Args:
        trace: One trajectory row from the dataframe.
        output_file: Open file handle for writing encoded output.
    """
    rid_list = [int(i) for i in trace["rid_list"].split(",")]
    time_list = [encode_time(i) for i in trace["time_list"].split(",")]

    destination = rid_list[-1]
    destination_gps = rid_gps[str(destination)]

    if not random_encode:
        for i in range(1, len(rid_list)):
            cur_loc = rid_list[:i]
            cur_time = time_list[:i]
            cur_rid = cur_loc[-1]

            if str(cur_rid) not in adjacent_list or rid_list[i] not in adjacent_list[str(cur_rid)]:
                # Path discontinuity detected; discard the rest of this trajectory.
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
                    dist = (
                        distance.distance(
                            (destination_gps[1], destination_gps[0]),
                            (candidate_gps[1], candidate_gps[0]),
                        ).kilometers
                        * 10
                    )  # unit: 100 meters
                    candidate_dis.append(dist)

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

    else:
        i = 1
        while i < len(rid_list):
            cur_loc = rid_list[:i]
            cur_time = time_list[:i]
            cur_rid = cur_loc[-1]

            if str(cur_rid) not in adjacent_list or rid_list[i] not in adjacent_list[str(cur_rid)]:
                # Path discontinuity detected; discard the rest of this trajectory.
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
                    dist = (
                        distance.distance(
                            (destination_gps[1], destination_gps[0]),
                            (candidate_gps[1], candidate_gps[0]),
                        ).kilometers
                        * 10
                    )  # unit: 100 meters
                    candidate_dis.append(dist)

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

            # Instead of i += 1, advance by a random number of steps.
            step = np.random.randint(1, max_step)
            i += step


os.makedirs("pretrain_data", exist_ok=True)

if __name__ == "__main__":
    train_rate = 0.9
    total_data_num = train_data.shape[0]
    train_num = int(total_data_num * train_rate)

    if dataset_name == "BJ_Taxi":
        train_output = open("pretrain_data/bj_taxi_pretrain_input_train.csv", "w")
        eval_output = open("pretrain_data/bj_taxi_pretrain_input_eval.csv", "w")
        test_output = open("pretrain_data/bj_taxi_pretrain_input_test.csv", "w")

    elif dataset_name == "Porto_Taxi":
        os.makedirs(out_dir, exist_ok=True)
        train_output = open(os.path.join(out_dir, "porto_taxi_pretrain_input_train.csv"), "w")
        eval_output = open(os.path.join(out_dir, "porto_taxi_pretrain_input_eval.csv"), "w")
        test_output = open(os.path.join(out_dir, "porto_taxi_pretrain_input_test.csv"), "w")

    else:
        train_output = open(os.path.join(data_root, dataset_name, "xianshi_partA_pretrain_input_train.csv"), "w")
        eval_output = open(os.path.join(data_root, dataset_name, "xianshi_partA_pretrain_input_eval.csv"), "w")
        test_output = open(os.path.join(data_root, dataset_name, "xianshi_partA_pretrain_input_test.csv"), "w")

    train_output.write("trace_loc,trace_time,des,candidate_set,candidate_dis,target\n")
    eval_output.write("trace_loc,trace_time,des,candidate_set,candidate_dis,target\n")
    test_output.write("trace_loc,trace_time,des,candidate_set,candidate_dis,target\n")

    for index, row in tqdm(train_data.iterrows(), total=train_data.shape[0], desc="Encoding training trajectories"):
        if index <= train_num:
            encode_trace(row, train_output)
        else:
            encode_trace(row, eval_output)

    for _, row in tqdm(test_data.iterrows(), total=test_data.shape[0], desc="Encoding test trajectories"):
        encode_trace(row, test_output)

    train_output.close()
    eval_output.close()
    test_output.close()