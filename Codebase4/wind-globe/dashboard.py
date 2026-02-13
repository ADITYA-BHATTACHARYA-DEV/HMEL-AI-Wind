import streamlit as st
from data.wind_generator import generate_3d_wind_data
from utils.geocode import search_place
from analysis.wind_analysis import analyze_local_wind
from visualization.globe import build_globe

st.set_page_config(page_title="Wind Globe Dashboard", layout="wide")

st.title("🌍 Wind Globe Dashboard")
st.markdown("Explore global wind flows, search places, and view local wind analysis.")

# Generate wind data
data = generate_3d_wind_data()

# Search bar
place = st.text_input("Search for a place:", "New York")

search_result = None
analysis = None
if place:
    search_result = search_place(place)
    if search_result:
        analysis = analyze_local_wind(data, search_result["lat"], search_result["lng"])
        st.success(f"Found {search_result['name']} at ({search_result['lat']:.2f}, {search_result['lng']:.2f})")
        if analysis:
            st.metric("Average Speed", f"{analysis['avg_speed']:.2f} m/s")
            st.metric("Peak Speed", f"{analysis['max_speed']:.2f} m/s")
            st.metric("Direction", analysis['direction'])
    else:
        st.error("Place not found.")

# Build globe visualization
deck = build_globe(data, search_result, analysis)
st.pydeck_chart(deck)
