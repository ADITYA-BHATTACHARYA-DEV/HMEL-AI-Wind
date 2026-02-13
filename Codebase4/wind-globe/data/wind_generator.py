import numpy as np
import pandas as pd

def generate_3d_wind_data():
    lats = np.arange(-60, 75, 2)
    lons = np.arange(-180, 180, 2)
    lon2d, lat2d = np.meshgrid(lons, lats)

    u = (np.cos(np.radians(lat2d)) * 20) + (np.sin(np.radians(lon2d)) * 5)
    v = (np.sin(np.radians(lat2d)) * 8)

    mag = np.sqrt(u**2 + v**2)
    angle = (np.arctan2(u, v) * 180 / np.pi + 360) % 360

    return pd.DataFrame({
        "lng": lon2d.flatten(),
        "lat": lat2d.flatten(),
        "speed": mag.flatten(),
        "angle": angle.flatten(),
        "elevation": np.log1p(mag.flatten()) * 8000
    })
