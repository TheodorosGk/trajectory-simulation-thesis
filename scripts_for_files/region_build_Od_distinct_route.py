import json
import os
from collections import defaultdict

import pandas as pd

# Adapted for thesis documentation purposes.
# Replace the paths below with your own local paths before running.

base = r"PASTE_YOUR_PRETRAIN_DATA_PATH_HERE"

# Example files inside the selected folder:
traj_path = os.path.join(base, "porto_taxi_mm_region_train.csv")   # You may also use eval/test files
region_gps_path = os.path.join(base, "porto_region_gps.json")
out_path = os.path.join(base, "region_od_distinct_route.json")

MAX_ROUTES_PER_OD = 30  # Limit the number of stored distinct routes per OD pair

df = pd.read_csv(traj_path)

with open(region_gps_path, "r", encoding="utf-8") as f:
    region_gps = json.load(f)  # Example format: {"0": [lon, lat], ...}

# od_routes["o-d"] = set(route_str) for uniqueness
od_routes = defaultdict(set)

for _, row in df.iterrows():
    regs = [int(x) for x in str(row["region_list"]).split(",") if x != ""]
    if len(regs) < 2:
        continue

    origin, destination = regs[0], regs[-1]
    od_key = f"{origin}-{destination}"

    if len(od_routes[od_key]) >= MAX_ROUTES_PER_OD:
        continue

    # Build route as GPS sequence: [[lat, lon], ...]
    # This format matches the expected input of yaw_loss
    gps_seq = []
    valid_route = True

    for region_id in regs:
        if str(region_id) not in region_gps:
            valid_route = False
            break

        lon, lat = region_gps[str(region_id)]
        gps_seq.append([lat, lon])

    if not valid_route:
        continue

    # Store as string first to ensure uniqueness, then convert back to list later
    od_routes[od_key].add(json.dumps(gps_seq))

# Convert sets -> lists of GPS sequences
output_data = {key: [json.loads(route) for route in routes] for key, routes in od_routes.items()}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f)

print("Saved:", out_path, "| OD pairs:", len(output_data))