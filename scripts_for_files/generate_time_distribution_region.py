import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# Adapted for thesis documentation purposes.
# Replace the paths below with your own local paths before running.

base_path = r"PASTE_YOUR_PRETRAIN_DATA_PATH_HERE"

region2rid_path = os.path.join(base_path, "region2rid.json")
train_file = os.path.join(base_path, "porto_taxi_mm_region_train.csv")
eval_file = os.path.join(base_path, "porto_taxi_mm_region_eval.csv")
test_file = os.path.join(base_path, "porto_taxi_mm_region_test.csv")
output_path = os.path.join(base_path, "region_time_distribution.npy")


with open(region2rid_path, "r", encoding="utf-8") as f:
    region2rid = json.load(f)

region_num = len(region2rid)

# Shape: [24 hours, number of regions]
# Default value = 1
time_distribution = np.ones((24, region_num))
time_distribution_cnt = {}


def parse_time(time_in):
    """
    Convert timestamp string in JSON time format to datetime.
    Example format: YYYY-MM-DDTHH:MM:SSZ
    """
    return datetime.strptime(time_in, "%Y-%m-%dT%H:%M:%SZ")


data_files = [train_file, eval_file, test_file]

for file_path in data_files:
    data = pd.read_csv(file_path)

    for _, row in tqdm(data.iterrows(), total=data.shape[0], desc=f"Processing {os.path.basename(file_path)}"):
        rid_list = [int(x) for x in row["region_list"].split(",")]
        time_list = row["time_list"].split(",")

        current_time = parse_time(time_list[0])

        for i in range(len(rid_list) - 1):
            next_time = parse_time(time_list[i + 1])
            cost_time = (next_time - current_time).seconds

            assert cost_time >= 0

            if cost_time > 0:
                current_region = rid_list[i]
                current_hour = current_time.hour

                if current_hour not in time_distribution_cnt:
                    time_distribution_cnt[current_hour] = {current_region: [1, cost_time]}
                elif current_region not in time_distribution_cnt[current_hour]:
                    time_distribution_cnt[current_hour][current_region] = [1, cost_time]
                else:
                    time_distribution_cnt[current_hour][current_region][0] += 1
                    time_distribution_cnt[current_hour][current_region][1] += cost_time

            current_time = next_time

filled_count = 0
for hour in time_distribution_cnt:
    for region_id in time_distribution_cnt[hour]:
        count, total_time = time_distribution_cnt[hour][region_id]
        avg_cost_time = total_time // count

        if avg_cost_time > 0:
            time_distribution[hour][region_id] = avg_cost_time
            filled_count += 1

print(f"Filled entries: {filled_count} / {24 * region_num}")

np.save(output_path, time_distribution)
print(f"Saved time distribution to: {output_path}")