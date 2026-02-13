import pydeck as pdk

def build_globe(data, search_result=None, analysis=None):
    view_state = pdk.ViewState(
        latitude=search_result["lat"] if search_result else 0,
        longitude=search_result["lng"] if search_result else 0,
        zoom=0.8,
        pitch=30,
        bearing=0
    )

    wind_layer = pdk.Layer(
        "ColumnLayer",
        data,
        get_position=["lng", "lat"],
        get_elevation="elevation",
        elevation_scale=1,
        radius=40000,
        get_fill_color="[speed * 12, 100, 255 - (speed * 8), 160]",
        pickable=True,
        auto_highlight=True,
        extruded=True,
    )

    layers = [wind_layer]

    if search_result:
        highlight_layer = pdk.Layer(
            "ScatterplotLayer",
            [search_result],
            get_position=["lng", "lat"],
            get_color=[255, 0, 0, 255],
            get_radius=100000,
        )
        layers.append(highlight_layer)

    tooltip_text = "Wind Speed: {speed} m/s\nDir: {angle}°"
    if analysis:
        tooltip_text += f"\nLocal Avg: {analysis['avg_speed']:.2f} m/s\nPeak: {analysis['max_speed']:.2f} m/s\nDir: {analysis['direction']}"

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        views=[pdk.View(type="OrbitView", controller=True)],
        map_style=None,
        tooltip={"text": tooltip_text}
    )
