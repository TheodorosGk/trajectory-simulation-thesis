import os
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# Adapted for thesis documentation purposes.
# Simplified for the Porto_Taxi thesis experiment only.
# Replace the path below with your own local path before running.

# =========================================================
# PATH CONFIGURATION
# =========================================================
base_path = r"PASTE_YOUR_PRETRAIN_DATA_PATH_HERE"

traj_path = os.path.join(base_path, "Porto_2026.csv")
output_path = os.path.join(base_path, "road_time_distribution.npy")

# Number of road segments in the processed Porto road network
road_num = 11095

# Shape: [24 hours, number of roads]
# Default value = 1
time_distribution = np.ones((24, road_num))
time_distribution_cnt = {}


def parse_time(time_in):
    """
    Convert timestamp string in JSON time format to datetime.
    Example format: YYYY-MM-DDTHH:MM:SSZ
    """
    return datetime.strptime(time_in, "%Y-%m-%dT%H:%M:%SZ")


data = pd.read_csv(traj_path)

for _, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing road travel times"):
    rid_list = [int(x) for x in row["rid_list"].split(",")]
    time_list = row["time_list"].split(",")

    current_time = parse_time(time_list[0])

    for i in range(len(rid_list) - 1):
        next_time = parse_time(time_list[i + 1])
        cost_time = (next_time - current_time).seconds

        assert cost_time >= 0

        if cost_time > 0:
            current_road = rid_list[i]
            current_hour = current_time.hour

            if current_hour not in time_distribution_cnt:
                time_distribution_cnt[current_hour] = {current_road: [1, cost_time]}
            elif current_road not in time_distribution_cnt[current_hour]:
                time_distribution_cnt[current_hour][current_road] = [1, cost_time]
            else:
                time_distribution_cnt[current_hour][current_road][0] += 1
                time_distribution_cnt[current_hour][current_road][1] += cost_time

        current_time = next_time

filled_count = 0
for hour in time_distribution_cnt:
    for road_id in time_distribution_cnt[hour]:
        count, total_time = time_distribution_cnt[hour][road_id]
        avg_cost_time = total_time // count

        if avg_cost_time > 0:
            time_distribution[hour][road_id] = avg_cost_time
            filled_count += 1

print(f"Filled entries: {filled_count} / {24 * road_num}")

np.save(output_path, time_distribution)
print(f"Saved road time distribution to: {output_path}")