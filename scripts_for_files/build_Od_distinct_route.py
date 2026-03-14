# build_od_distinct_route.py

import json

import pandas as pd

from collections import defaultdict

 

# === inputs (change as needed) ===

traj_csv = r'PATH\Porto_2026.csv'   # your trajectory_file
rid_gps_file = r'PATH\porto_rid_gps.json'              # {rid: [lon, lat]}

out_json = r'PATH\od_distinct_route.json'
 

# === load ===

df = pd.read_csv(traj_csv)

with open(rid_gps_file, 'r') as f:

    rid_gps = json.load(f)

 

# === build OD -> list of trajectories (each as [[lat, lon], ...]) ===

od_routes = defaultdict(list)

 

for _, row in df.iterrows():

    # rid_list may contain padding like '000' -> drop non-numeric/zero tokens

    rid_list = [r for r in row['rid_list'].split(',') if r.isdigit() and int(r) != 0]

    if len(rid_list) < 2:

        continue

 

    origin, dest = rid_list[0], rid_list[-1]

 

    # IMPORTANT: Rollout.yaw_loss uses the format ' {o}- {d}' (with a space before each)

    # See: rollout.py (od_key = ' {}- {}'.format(origin, des))

    od_key = f' {origin}- {dest}'  # leading spaces to match the code path  [1](https://github.com/WenMellors/TS-TrajGen/blob/master/rollout.py)

 

    # Build GPS trajectory as [lat, lon] for each rid

    gps_traj = []

    ok = True

    for rid in rid_list:

        g = rid_gps.get(str(rid))

        if g is None or len(g) != 2:

            ok = False

            break

        lon, lat = g[0], g[1]      # repo stores [lon, lat]

        gps_traj.append([lat, lon]) # store as [lat, lon]  [1](https://github.com/WenMellors/TS-TrajGen/blob/master/rollout.py)

    if not ok:

        continue

 

    # Optional: de-duplicate identical trajectories per OD using a string key on rid_list

    # (keeps JSON smaller but not required)

    od_routes[od_key].append(gps_traj)

 

# === write ===

with open(out_json, 'w') as f:

    json.dump(od_routes, f)

print(f'Wrote {out_json} with {len(od_routes)} OD keys.')