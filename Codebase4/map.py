import pydeck as pdk
import pandas as pd
import numpy as np
import webbrowser
import os


# --- 1. Scientific Data Generation ---
def generate_3d_wind_data():
    """Simulates a global wind field for 3D visualization"""
    # Create a grid of latitudes and longitudes
    lats = np.arange(-60, 75, 2)
    lons = np.arange(-180, 180, 2)
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Simulate atmospheric flow (U-component)
    # Higher velocity near the equator and simulated jet streams
    u = (np.cos(np.radians(lat2d)) * 20) + (np.sin(np.radians(lon2d)) * 5)

    df = pd.DataFrame({
        "lng": lon2d.flatten(),
        "lat": lat2d.flatten(),
        "speed": np.abs(u.flatten()),
        # Height is scaled for dramatic 3D effect (meters)
        "elevation": np.abs(u.flatten()) * 15000
    })
    return df


# --- 2. Build the 3D Map ---
data = generate_3d_wind_data()

# Set the camera to a 3D tilted perspective
view_state = pdk.ViewState(
    latitude=25,
    longitude=10,
    zoom=2.5,
    pitch=50,  # Tilt for 3D depth
    bearing=0
)

# Define the 3D Column Layer
wind_layer = pdk.Layer(
    "ColumnLayer",
    data,
    get_position=["lng", "lat"],
    get_elevation="elevation",
    elevation_scale=1,
    radius=40000,
    # Color scale: Blue (slow) to Red (fast)
    get_fill_color="[speed * 12, 100, 255 - (speed * 8), 160]",
    pickable=True,
    auto_highlight=True,
    extruded=True,  # Critical for 3D
)

# --- 3. Render and Export ---
r = pdk.Deck(
    layers=[wind_layer],
    initial_view_state=view_state,
    map_style="mapbox://styles/mapbox/dark-v10",  # Dark mode for better contrast
    tooltip={"text": "Wind Speed: {speed} m/s\nAltitude Effect: {elevation}m"}
)

# Save to an HTML file in your current folder
filename = "wind_3d_analysis.html"
r.to_html(filename, open_browser=False)

# --- 4. Open in Browser ---
# Get the full path and open it automatically
full_path = "file://" + os.path.realpath(filename)
print(f"Opening 3D Map: {full_path}")
webbrowser.open(full_path)