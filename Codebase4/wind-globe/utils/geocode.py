from geopy.geocoders import Nominatim
import time

def search_place(place_name, retries=3, timeout=10):
    """
    Search for a place using Nominatim with retries and timeout.
    Falls back to None if geocoding fails.
    """
    geolocator = Nominatim(user_agent="wind_globe_dashboard", timeout=timeout)

    for attempt in range(retries):
        try:
            location = geolocator.geocode(place_name)
            if location:
                return {
                    "lat": location.latitude,
                    "lng": location.longitude,
                    "name": location.address
                }
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)  # wait before retry

    # If all retries fail, return None
    return None
