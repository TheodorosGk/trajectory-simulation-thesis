import json
import os

import pandas as pd
from tqdm import tqdm

# Adapted for thesis documentation purposes.
# Replace the paths below with your own local paths before running.

base_path = r"PASTE_YOUR_PRETRAIN_DATA_PATH_HERE"

rid2region_path = os.path.join(base_path, "rid2region.json")
traj_path = os.path.join(base_path, "PASTE_YOUR_TRAJECTORY_FILE_HERE.csv")
region_adjacent_path = os.path.join(base_path, "region_adjacent_list.json")
output_path = os.path.join(base_path, "region_transfer_prob.json")

region_transfer_cnt = {}

with open(rid2region_path, "r", encoding="utf-8") as f:
    rid2region = json.load(f)

mm_traj = pd.read_csv(traj_path)

for _, row in tqdm(mm_traj.iterrows(), total=mm_traj.shape[0], desc="Counting region transfer frequencies"):
    rid_list = row["rid_list"].split(",")
    prev_region = rid2region[rid_list[0]]

    for rid in rid_list[1:]:
        now_region = rid2region[rid]

        if prev_region != now_region:
            # A transition between two different regions occurred.
            # The current road segment is treated as the transfer road.
            if prev_region not in region_transfer_cnt:
                region_transfer_cnt[prev_region] = {now_region: {rid: 1}}
            elif now_region not in region_transfer_cnt[prev_region]:
                region_transfer_cnt[prev_region][now_region] = {rid: 1}
            elif rid not in region_transfer_cnt[prev_region][now_region]:
                region_transfer_cnt[prev_region][now_region][rid] = 1
            else:
                region_transfer_cnt[prev_region][now_region][rid] += 1

            # Update the current region after the transition
            prev_region = now_region

# Preprocess the counting results into the final output format
final_result = {}
for region_f in region_transfer_cnt:
    final_result[region_f] = {}

    for region_t in region_transfer_cnt[region_f]:
        border_rid_set = []
        border_rid_cnt = []
        rid_cnt = region_transfer_cnt[region_f][region_t]

        for rid in rid_cnt:
            border_rid_set.append(int(rid))
            border_rid_cnt.append(rid_cnt[rid])

        final_result[region_f][region_t] = {
            "transfer_rid": border_rid_set,
            "transfer_freq": border_rid_cnt,
        }

with open(region_adjacent_path, "r", encoding="utf-8") as f:
    region_adjacent_list = json.load(f)

missing_source_region_count = 0
missing_transfer_count = 0

# Check whether there are adjacent region pairs with no observed transitions
for region_f in region_adjacent_list:
    region_f_eval = eval(region_f)

    if region_f_eval not in final_result:
        final_result[region_f_eval] = {}
        missing_source_region_count += 1

    for region_t in region_adjacent_list[str(region_f_eval)]:
        region_t_eval = eval(region_t)

        if region_t_eval not in final_result[region_f_eval]:
            # If no transition was observed, assign uniform frequency = 1
            border_rid_set = region_adjacent_list[str(region_f_eval)][str(region_t_eval)]
            border_rid_cnt = [1 for _ in border_rid_set]

            final_result[region_f_eval][region_t_eval] = {
                "transfer_rid": border_rid_set,
                "transfer_freq": border_rid_cnt,
            }
            missing_transfer_count += 1

print("Missing source region count:", missing_source_region_count)
print("Missing transfer count:", missing_transfer_count)

# Save the final statistics
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_result, f)