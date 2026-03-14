import json
import math

class MapManager(object):

    def __init__(self, rid_gps_path, img_unit=0.005):

        with open(rid_gps_path, 'r') as f:
            rid_gps = json.load(f)

        lons = [v[0] for v in rid_gps.values()]
        lats = [v[1] for v in rid_gps.values()]

        self.lon_0 = min(lons)
        self.lon_1 = max(lons)
        self.lat_0 = min(lats)
        self.lat_1 = max(lats)

        self.img_unit = img_unit

        self.img_width = math.ceil((self.lon_1 - self.lon_0) / self.img_unit) + 1
        self.img_height = math.ceil((self.lat_1 - self.lat_0) / self.img_unit) + 1

    def gps2grid(self, lon, lat):
        x = math.floor((lon - self.lon_0) / self.img_unit)
        y = math.floor((lat - self.lat_0) / self.img_unit)

        x = min(max(x, 0), self.img_width)
        y = min(max(y, 0), self.img_height)

        return x, y
