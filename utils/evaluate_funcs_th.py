import os
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy
from scipy.spatial.distance import euclidean, cosine, cityblock, chebyshev
from fastdtw import fastdtw
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from pyproj import Geod
from geopy import distance
import hausdorff

# Adapted for thesis documentation purposes.
# Simplified for the Porto_Taxi thesis experiment only.
# Replace the path below with your own local path before running.

debug = False

# =========================================================
# PATH CONFIGURATION
# =========================================================
base_path = r"PASTE_YOUR_PRETRAIN_DATA_PATH_HERE"

road_len_file = os.path.join(base_path, "porto_road_length.json")
road_gps_file = os.path.join(base_path, "porto_rid_gps.json")
geo_file = os.path.join(base_path, "porto.geo")
road2grid_file = os.path.join(base_path, "porto_road2grid.json")

# =========================================================
# LOAD OR BUILD ROAD LENGTH / ROAD GPS DICTIONARIES
# =========================================================
if not os.path.exists(road_len_file) or not os.path.exists(road_gps_file):
    road_info = pd.read_csv(geo_file)
    road_length = {}
    road_gps = {}

    for _, row in tqdm(road_info.iterrows(), desc="Building road length and GPS dictionaries", total=road_info.shape[0]):
        rid = row["geo_id"]
        length = row["length"]

        coordinate = row["coordinates"].replace("[", "").replace("]", "").split(",")
        lon1 = float(coordinate[0])
        lat1 = float(coordinate[1])
        lon2 = float(coordinate[2])
        lat2 = float(coordinate[3])

        center_gps = ((lon1 + lon2) / 2, (lat1 + lat2) / 2)
        road_gps[str(rid)] = center_gps
        road_length[str(rid)] = length

    with open(road_len_file, "w", encoding="utf-8") as f:
        json.dump(road_length, f)

    with open(road_gps_file, "w", encoding="utf-8") as f:
        json.dump(road_gps, f)
else:
    with open(road_len_file, "r", encoding="utf-8") as f:
        road_length = json.load(f)

    with open(road_gps_file, "r", encoding="utf-8") as f:
        road_gps = json.load(f)

# =========================================================
# PORTO GRID CONFIGURATION
# =========================================================
lon_range = 0.133
lat_range = 0.046
lon_0 = -8.6887
lat_0 = 41.1405

img_unit = 0.005
img_width = math.ceil(lon_range / img_unit) + 1
img_height = math.ceil(lat_range / img_unit) + 1

road_pad = 11095

max_distance = 100
real_max_distance = 31
max_radius = 31.6764 * 31.6764
real_max_radius = 7.2

travel_distance_bins = np.arange(0, real_max_distance, float(real_max_distance) / 1000).tolist()
travel_distance_bins.append(real_max_distance + 1)
travel_distance_bins.append(max_distance)
travel_distance_bins = np.array(travel_distance_bins)

travel_radius_bins = np.arange(0, real_max_radius, float(real_max_radius) / 100).tolist()
travel_radius_bins.append(real_max_radius + 1)
travel_radius_bins.append(max_radius)
travel_radius_bins = np.array(travel_radius_bins)

# =========================================================
# LOAD OR BUILD ROAD-TO-GRID MAPPING
# =========================================================
if not os.path.exists(road2grid_file):
    road2grid = {}

    for road in road_gps:
        gps = road_gps[road]
        x = math.ceil((gps[0] - lon_0) / img_unit)
        y = math.ceil((gps[1] - lat_0) / img_unit)
        road2grid[road] = (x, y)

    with open(road2grid_file, "w", encoding="utf-8") as f:
        json.dump(road2grid, f)
else:
    with open(road2grid_file, "r", encoding="utf-8") as f:
        road2grid = json.load(f)

road_num = len(road2grid)


def cal_polygon_area(polygon, mode=1):
    """
    Compute polygon area from longitude/latitude coordinates.

    Args:
        polygon: List of polygon vertices.
        mode:
            1 -> square degrees
            2 -> square kilometers

    Returns:
        Polygon area.
    """
    if len(polygon) < 3:
        return 0

    if mode == 1:
        area = Polygon(polygon)
        return area.area

    geod = Geod(ellps="WGS84")
    area, _ = geod.geometry_area_perimeter(orient(Polygon(polygon)))
    return area / 1_000_000


def arr_to_distribution(arr, min_val, max_val, bins=10000):
    """
    Convert an array into a histogram distribution.
    """
    distribution, _ = np.histogram(
        arr,
        np.arange(min_val, max_val, float(max_val - min_val) / bins)
    )
    return distribution


def get_geogradius(rid_lat, rid_lon):
    """
    Compute the mean distance of trajectory points from their centroid
    as a radius-of-gyration style measure.
    """
    if len(rid_lat) == 0:
        return 0

    center_lon = np.mean(rid_lon)
    center_lat = np.mean(rid_lat)

    distances = []
    for i in range(len(rid_lat)):
        point_lon = rid_lon[i]
        point_lat = rid_lat[i]
        dist_km = distance.distance((center_lat, center_lon), (point_lat, point_lon)).kilometers
        distances.append(dist_km)

    return np.mean(distances)


def count_statistics(trace_set, use_real_timestamps):
    """
    Compute trajectory statistics for a trajectory dataset.

    Statistics include:
    - travel distance distribution
    - travel radius distribution
    - grid OD frequencies
    - time-sliced grid OD frequencies
    - road visit frequencies
    - time-sliced road visit frequencies

    Args:
        trace_set: pandas DataFrame with trajectory data.
        use_real_timestamps:
            True  -> timestamps are raw datetime strings
            False -> timestamps are already encoded time slots

    Returns:
        Dictionary with aggregated statistics.
    """
    travel_distance_hour = {}
    travel_radius_hour = {}
    travel_distance_total = []
    travel_radius_total = []

    total_grid = img_width * img_height
    grid_od_cnt = np.zeros((total_grid, total_grid), dtype=int)
    grid_time_od_cnt = np.zeros((24, total_grid, total_grid), dtype=int)

    rid_cnt = np.zeros(road_num, dtype=int)
    rid_time_cnt = np.zeros((24, road_num), dtype=int)

    for index, row in tqdm(trace_set.iterrows(), total=trace_set.shape[0], desc="Counting trajectory statistics"):
        rid_list = [int(x) for x in row["rid_list"].split(",")]
        rid_list = np.array(rid_list)
        rid_list = rid_list[rid_list != road_pad].tolist()

        if len(rid_list) == 0:
            continue

        time_list = row["time_list"].split(",")

        if use_real_timestamps:
            start_timestamp = time_list[0]
            start_time = datetime.strptime(start_timestamp, "%Y-%m-%dT%H:%M:%SZ")
            start_hour = start_time.hour
        else:
            start_hour = (int(time_list[0]) % 1440) // 60

        travel_distance = 0
        previous_gps = None
        rid_lat = []
        rid_lon = []

        if str(rid_list[0]) not in road2grid:
            continue

        start_rid_grid = road2grid[str(rid_list[0])]
        destination_rid_grid = None
        start_rid_grid_index = start_rid_grid[0] * img_height + start_rid_grid[1]

        for rid_index, rid in enumerate(rid_list):
            gps = road_gps[str(rid)]

            if previous_gps is None:
                previous_gps = gps
            else:
                travel_distance += distance.distance(
                    (gps[1], gps[0]),
                    (previous_gps[1], previous_gps[0])
                ).kilometers
                previous_gps = gps

            rid_lat.append(gps[1])
            rid_lon.append(gps[0])

            if use_real_timestamps:
                rid_time = datetime.strptime(time_list[rid_index], "%Y-%m-%dT%H:%M:%SZ")
                rid_hour = rid_time.hour
            else:
                rid_hour = (int(time_list[rid_index]) % 1440) // 60

            rid_cnt[rid] += 1
            rid_time_cnt[rid_hour, rid] += 1

        travel_distance_total.append(travel_distance)

        travel_radius = get_geogradius(rid_lat=rid_lat, rid_lon=rid_lon)
        travel_radius_total.append(travel_radius)

        if destination_rid_grid is None:
            destination_rid_grid = road2grid[str(rid_list[-1])]

        destination_rid_grid_index = destination_rid_grid[0] * img_height + destination_rid_grid[1]

        grid_od_cnt[start_rid_grid_index][destination_rid_grid_index] += 1
        grid_time_od_cnt[start_hour][start_rid_grid_index][destination_rid_grid_index] += 1

        if start_hour not in travel_distance_hour:
            travel_distance_hour[start_hour] = [travel_distance]
            travel_radius_hour[start_hour] = [travel_radius]
        else:
            travel_distance_hour[start_hour].append(travel_distance)
            travel_radius_hour[start_hour].append(travel_radius)

        if index == 1000 and debug:
            break

    travel_distance_total_distribution, _ = np.histogram(travel_distance_total, travel_distance_bins)
    travel_radius_total_distribution, _ = np.histogram(travel_radius_total, travel_radius_bins)

    grid_od_cnt = grid_od_cnt.flatten()
    grid_time_od_cnt = grid_time_od_cnt.reshape(24, -1)

    result = {
        "travel_distance_total_distribution": travel_distance_total_distribution,
        "travel_radius_total_distribution": travel_radius_total_distribution,
        "travel_distance_hour_distribution": np.zeros((24, travel_distance_total_distribution.shape[0])),
        "travel_radius_hour_distribution": np.zeros((24, travel_radius_total_distribution.shape[0])),
        "grid_od_freq": grid_od_cnt,
        "grid_time_od_freq": grid_time_od_cnt,
        "rid_freq": rid_cnt,
        "rid_time_freq": np.zeros((24, road_num)),
    }

    for hour in range(24):
        if hour in travel_distance_hour:
            result["rid_time_freq"][hour] = rid_time_cnt[hour]
            result["travel_distance_hour_distribution"][hour], _ = np.histogram(
                travel_distance_hour[hour], travel_distance_bins
            )
            result["travel_radius_hour_distribution"][hour], _ = np.histogram(
                travel_radius_hour[hour], travel_radius_bins
            )

    return result


def js_divergence(p, q):
    """
    Jensen-Shannon divergence.
    """
    m = (p + q) / 2
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)


def edit_distance(trace1, trace2):
    """
    Compute edit distance between two trajectories.
    """
    matrix = [[i + j for j in range(len(trace2) + 1)] for i in range(len(trace1) + 1)]

    for i in range(1, len(trace1) + 1):
        for j in range(1, len(trace2) + 1):
            d = 0 if trace1[i - 1] == trace2[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + d
            )

    return matrix[len(trace1)][len(trace2)]


def hausdorff_metric(truth, pred, distance_metric="haversine"):
    """
    Compute Hausdorff distance between two trajectories.
    """
    return hausdorff.hausdorff_distance(truth, pred, distance=distance_metric)


def haversine(array_x, array_y):
    """
    Haversine distance in kilometers between two [lat, lon] points.
    """
    earth_radius_km = 6378.0
    radians = np.pi / 180.0

    lat_x = radians * array_x[0]
    lon_x = radians * array_x[1]
    lat_y = radians * array_y[0]
    lon_y = radians * array_y[1]

    dlon = lon_y - lon_x
    dlat = lat_y - lat_x

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat_x) * math.cos(lat_y) * math.sin(dlon / 2.0) ** 2
    )
    return earth_radius_km * 2 * math.asin(math.sqrt(a))


def dtw_metric(truth, pred, distance_metric="haversine"):
    """
    Compute Dynamic Time Warping distance between two trajectories.
    """
    if distance_metric == "haversine":
        dist_value, _ = fastdtw(truth, pred, dist=haversine)
    elif distance_metric == "manhattan":
        dist_value, _ = fastdtw(truth, pred, dist=cityblock)
    elif distance_metric == "euclidean":
        dist_value, _ = fastdtw(truth, pred, dist=euclidean)
    elif distance_metric == "chebyshev":
        dist_value, _ = fastdtw(truth, pred, dist=chebyshev)
    elif distance_metric == "cosine":
        dist_value, _ = fastdtw(truth, pred, dist=cosine)
    else:
        dist_value, _ = fastdtw(truth, pred, dist=euclidean)

    return dist_value


rad = math.pi / 180.0
earth_radius_m = 6378137.0


def great_circle_distance(lon1, lat1, lon2, lat2):
    """
    Compute great-circle distance in meters.
    """
    dlat = rad * (lat2 - lat1)
    dlon = rad * (lon2 - lon1)

    a = (
        math.sin(dlat / 2.0) * math.sin(dlat / 2.0)
        + math.cos(rad * lat1) * math.cos(rad * lat2)
        * math.sin(dlon / 2.0) * math.sin(dlon / 2.0)
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_radius_m * c


def s_edr(t0, t1, eps):
    """
    Compute Edit Distance on Real sequence (EDR) between two trajectories.
    """
    n0 = len(t0)
    n1 = len(t1)

    c_matrix = np.full((n0 + 1, n1 + 1), np.inf)
    c_matrix[:, 0] = np.arange(n0 + 1)
    c_matrix[0, :] = np.arange(n1 + 1)

    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            if great_circle_distance(t0[i - 1][0], t0[i - 1][1], t1[j - 1][0], t1[j - 1][1]) < eps:
                subcost = 0
            else:
                subcost = 1

            c_matrix[i][j] = min(
                c_matrix[i][j - 1] + 1,
                c_matrix[i - 1][j] + 1,
                c_matrix[i - 1][j - 1] + subcost
            )

    return float(c_matrix[n0][n1]) / max([n0, n1])


def cosine_similarity(x, y):
    """
    Compute cosine similarity between two vectors.
    """
    numerator = x.dot(y.T)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return numerator / denominator


def rid_cnt2heat_level(rid_cnt):
    """
    Convert road visit counts to coarse heat levels.
    """
    min_val = 0
    max_val = np.max(rid_cnt)
    level_num = 100
    bin_size = max_val // level_num
    rid_heat_level = rid_cnt // bin_size
    return rid_heat_level