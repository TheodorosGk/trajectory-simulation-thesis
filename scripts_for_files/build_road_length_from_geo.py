import os
import json

import pandas as pd

# Adapted for thesis documentation purposes.
# Simplified for the Porto_Taxi thesis experiment only.
# Replace the path below with your own local path before running.

# =========================================================
# PATH CONFIGURATION
# =========================================================
base_path = r"PASTE_YOUR_PRETRAIN_DATA_PATH_HERE"

geo_file = os.path.join(base_path, "porto.geo")
out_json = os.path.join(base_path, "road_length.json")

# =========================================================
# LOAD ROAD NETWORK GEO FILE
# Expected columns:
# - geo_id: road segment id
# - length: road segment length in meters
# =========================================================
geo = pd.read_csv(geo_file)

road_length = {
    str(int(row["geo_id"])): float(row["length"])
    for _, row in geo.iterrows()
}

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(road_length, f)

print(f"Saved road length dictionary to: {out_json}")
print(f"Total road segments: {len(road_length)}")