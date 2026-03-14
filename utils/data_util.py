from datetime import datetime


def parse_coordinate(coordinate_str, geom_type):
    """
    Parse a coordinate string from the dataset.

    Args:
        coordinate_str: Coordinate string in the original dataset format.
        geom_type: Geometry type, either "Point" or "LineString".

    Returns:
        For "Point":
            (lon, lat)

        For "LineString":
            (lon1, lat1, lon2, lat2)
    """
    values = coordinate_str.replace("[", "").replace("]", "").split(",")

    if geom_type == "LineString":
        lon1 = float(values[0])
        lat1 = float(values[1])
        lon2 = float(values[2])
        lat2 = float(values[3])
        return lon1, lat1, lon2, lat2

    if geom_type == "Point":
        lon = float(values[0])
        lat = float(values[1])
        return lon, lat

    raise ValueError(f"Unsupported geometry type: {geom_type}")


def encode_time(timestamp):
    """
    Encode a timestamp into minute-of-day representation.

    Weekend timestamps are shifted by +1440 to distinguish them from weekdays.

    Args:
        timestamp: String timestamp in the format YYYY-MM-DDTHH:MM:SSZ

    Returns:
        Encoded integer time value.
    """
    time_obj = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")

    if time_obj.weekday() in [5, 6]:
        return time_obj.hour * 60 + time_obj.minute + 1440

    return time_obj.hour * 60 + time_obj.minute