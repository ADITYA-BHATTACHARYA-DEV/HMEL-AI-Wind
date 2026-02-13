def analyze_local_wind(data, lat, lng, radius=5):
    subset = data[
        (data["lat"].between(lat-radius, lat+radius)) &
        (data["lng"].between(lng-radius, lng+radius))
    ]
    if subset.empty:
        return None

    avg_speed = subset["speed"].mean()
    max_speed = subset["speed"].max()
    avg_angle = subset["angle"].mean()

    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    cardinal = dirs[int(((avg_angle + 22.5) % 360) / 45)]

    return {
        "avg_speed": avg_speed,
        "max_speed": max_speed,
        "direction": cardinal
    }
