# data_loader.py
import pandas as pd
import numpy as np


def generate_refinery_data(n_samples=2000):
    """
    Generates synthetic wind data simulating a complex refinery environment.
    """
    np.random.seed(42)

    # FIX 1: Changed '15T' to '15min' to solve the FutureWarning
    timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='15min')

    # 1. Base Wind Speed (Weibull-ish distribution)
    # Background flow + random gusts
    base_speed = 5 + 3 * np.sin(np.linspace(0, 10, n_samples)) + np.random.normal(0, 1.5, n_samples)
    speed_10m = np.clip(base_speed, 0.5, 25)

    # 2. Vertical Shear (Hellmann Law)
    # Alpha = 0.25 (Refinery/Urban terrain)
    alpha = 0.25
    speed_30m = speed_10m * (30 / 10) ** alpha
    speed_50m = speed_10m * (50 / 10) ** alpha

    # 3. Direction (0-360 degrees)
    # Prevailing wind from NW (315) with variance
    direction = (315 + np.random.normal(0, 30, n_samples)) % 360

    # 4. Air Density (Rho)
    # Refineries are hot -> Lower density than standard 1.225
    temp_c = 25 + 5 * np.sin(np.linspace(0, 20, n_samples))  # Fluctuating temp
    rho = 1.225 * (288.15 / (temp_c + 273.15))  # Ideal gas law approx

    # 5. Power Density (W/m^2)
    # FIX 2: Added WPD calculation for 30m height
    wpd = 0.5 * rho * (speed_10m ** 3)  # Ground level WPD
    wpd_30m = 0.5 * rho * (speed_30m ** 3)  # Tower level WPD (This was missing)

    # Create DataFrame
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'Speed': speed_10m,  # Primary speed column
        'Speed_10m': speed_10m,
        'Speed_30m': speed_30m,
        'Speed_50m': speed_50m,
        'Direction': direction,
        'Rho': rho,
        'WPD': wpd,
        'WPD_30m': wpd_30m  # Added to DataFrame
    })

    # Add Vector Components for AI (important for direction clustering)
    df['Wx'] = df['Speed'] * np.cos(np.deg2rad(df['Direction']))
    df['Wy'] = df['Speed'] * np.sin(np.deg2rad(df['Direction']))

    return df


# Helper for loading external CSVs if you have real data
def load_and_process_data(filepath=None):
    if filepath:
        return pd.read_csv(filepath)
    else:
        return generate_refinery_data()