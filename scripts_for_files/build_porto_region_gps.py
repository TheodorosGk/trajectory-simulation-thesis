import json
import numpy as np

base = r"PATH"

region2rid_path = base + r"\region2rid.json"
rid_gps_path    = base + r"\porto_rid_gps.json"   
out_path        = base + r"\porto_region_gps.json"

with open(region2rid_path, "r") as f:
    region2rid = json.load(f)

with open(rid_gps_path, "r") as f:
    rid_gps = json.load(f)

region_gps = {}

for region, rid_list in region2rid.items():
    lons, lats = [], []
    for rid in rid_list:
        lon, lat = rid_gps[str(rid)]
        lons.append(lon)
        lats.append(lat)
    region_gps[str(region)] = [float(np.mean(lons)), float(np.mean(lats))]  # [lon, lat]

with open(out_path, "w") as f:
    json.dump(region_gps, f)

print("saved:", out_path, "regions:", len(region_gps))
