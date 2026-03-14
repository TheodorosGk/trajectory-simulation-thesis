import os
import json

import pandas as pd
from geopy import distance

# Adapted for thesis documentation purposes.
# Simplified for the Porto_Taxi thesis experiment only.
# Replace the paths below with your own local paths before running.

# =========================================================
# PATH CONFIGURATION
# =========================================================
base_path = r"PASTE_YOUR_PRETRAIN_DATA_PATH_HERE"
output_path = r"PASTE_YOUR_EVALUATION_OUTPUT_PATH_HERE"

true_test_csv = os.path.join(base_path, "porto_mm_test.csv")
generated_csv = os.path.join(base_path, "TS_TrajGen_generate.csv")
rid_gps_json = os.path.join(base_path, "porto_rid_gps.json")

out_failure_samples = os.path.join(output_path, "failure_samples.csv")
out_success_samples = os.path.join(output_path, "success_samples.csv")

sample_n = 50

# =========================================================
# LOAD DATA
# =========================================================
true_df = pd.read_csv(true_test_csv)
gen_df = pd.read_csv(generated_csv)

# Align on common traj_id values
true_df = true_df.set_index("traj_id")
gen_df = gen_df.set_index("traj_id")

common_ids = true_df.index.intersection(gen_df.index)
true_df = true_df.loc[common_ids]
gen_df = gen_df.loc[common_ids]

with open(rid_gps_json, "r", encoding="utf-8") as f:
    rid_gps = json.load(f)

os.makedirs(output_path, exist_ok=True)


def last_rid(rid_list_str):
    """
    Return the last road id from a comma-separated rid_list string.
    """
    parts = str(rid_list_str).split(",")
    return int(parts[-1])


def first_rid(rid_list_str):
    """
    Return the first road id from a comma-separated rid_list string.
    """
    parts = str(rid_list_str).split(",")
    return int(parts[0])


def destination_error_km(rid_a, rid_b):
    """
    Compute geographic distance in kilometers between two road destinations
    using the road-center GPS dictionary.
    """
    gps_a = rid_gps.get(str(rid_a))
    gps_b = rid_gps.get(str(rid_b))

    if gps_a is None or gps_b is None:
        return float("nan")

    # rid_gps format is [lon, lat], while geopy expects (lat, lon)
    return float(distance.distance((gps_a[1], gps_a[0]), (gps_b[1], gps_b[0])).kilometers)


# =========================================================
# COMPARE TRUE VS GENERATED DESTINATIONS
# =========================================================
records = []

for traj_id in common_ids:
    true_rids = true_df.at[traj_id, "rid_list"]
    gen_rids = gen_df.at[traj_id, "rid_list"]

    true_last = last_rid(true_rids)
    gen_last = last_rid(gen_rids)

    success = (true_last == gen_last)
    error_km = destination_error_km(true_last, gen_last)

    records.append({
        "traj_id": traj_id,
        "true_start": first_rid(true_rids),
        "true_dest": true_last,
        "gen_dest": gen_last,
        "success": int(success),
        "dest_error_km": error_km,
        "true_len": len(str(true_rids).split(",")),
        "gen_len": len(str(gen_rids).split(",")),
    })

results = pd.DataFrame(records)

# =========================================================
# SUMMARY
# =========================================================
total = len(results)
success_count = int(results["success"].sum())
fail_count = total - success_count

print("Total:", total)
print("Success:", success_count, f"({success_count / total:.3f})")
print("Fail:", fail_count, f"({fail_count / total:.3f})")

print("\nDestination error (km) statistics on failures:")
fail_df = results[results["success"] == 0].copy()
print(fail_df["dest_error_km"].describe())

# =========================================================
# SAVE SAMPLE FAILURES / SUCCESSES
# =========================================================
fail_sample = fail_df.sort_values("dest_error_km", ascending=False).head(sample_n)

success_df = results[results["success"] == 1].copy()
success_sample = success_df.sample(
    n=min(sample_n, success_count),
    random_state=42
)

fail_sample.to_csv(out_failure_samples, index=False)
success_sample.to_csv(out_success_samples, index=False)

print("\nSaved:", out_failure_samples)
print("Saved:", out_success_samples)