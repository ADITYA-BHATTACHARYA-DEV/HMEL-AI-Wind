# data_loader.py
import pandas as pd
import numpy as np

#
# def generate_refinery_data(n_samples=2000):
#     """
#     Generates synthetic wind data simulating a complex refinery environment.
#     """
#     np.random.seed(42)
#
#     # FIX 1: Changed '15T' to '15min' to solve the FutureWarning
#     timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='15min')
#
#     # 1. Base Wind Speed (Weibull-ish distribution)
#     # Background flow + random gusts
#     base_speed = 5 + 3 * np.sin(np.linspace(0, 10, n_samples)) + np.random.normal(0, 1.5, n_samples)
#     speed_10m = np.clip(base_speed, 0.5, 25)
#
#     # 2. Vertical Shear (Hellmann Law)
#     # Alpha = 0.25 (Refinery/Urban terrain)
#     alpha = 0.25
#     speed_30m = speed_10m * (30 / 10) ** alpha
#     speed_50m = speed_10m * (50 / 10) ** alpha
#
#     # 3. Direction (0-360 degrees)
#     # Prevailing wind from NW (315) with variance
#     direction = (315 + np.random.normal(0, 30, n_samples)) % 360
#
#     # 4. Air Density (Rho)
#     # Refineries are hot -> Lower density than standard 1.225
#     temp_c = 25 + 5 * np.sin(np.linspace(0, 20, n_samples))  # Fluctuating temp
#     rho = 1.225 * (288.15 / (temp_c + 273.15))  # Ideal gas law approx
#
#     # 5. Power Density (W/m^2)
#     # FIX 2: Added WPD calculation for 30m height
#     wpd = 0.5 * rho * (speed_10m ** 3)  # Ground level WPD
#     wpd_30m = 0.5 * rho * (speed_30m ** 3)  # Tower level WPD (This was missing)
#
#     # Create DataFrame
#     df = pd.DataFrame({
#         'Timestamp': timestamps,
#         'Speed': speed_10m,  # Primary speed column
#         'Speed_10m': speed_10m,
#         'Speed_30m': speed_30m,
#         'Speed_50m': speed_50m,
#         'Direction': direction,
#         'Rho': rho,
#         'WPD': wpd,
#         'WPD_30m': wpd_30m  # Added to DataFrame
#     })
#
#     # Add Vector Components for AI (important for direction clustering)
#     df['Wx'] = df['Speed'] * np.cos(np.deg2rad(df['Direction']))
#     df['Wy'] = df['Speed'] * np.sin(np.deg2rad(df['Direction']))
#
#     return df
#
#
# # Helper for loading external CSVs if you have real data
# def load_and_process_data(filepath=None):
#     if filepath:
#         return pd.read_csv(filepath)
#     else:
#         return generate_refinery_data()


# data_loader.py
import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
REAL_DATA_PATH = r"F:\HMEL_Project\data_folder\merged_refinery_data.csv"


def clean_data(df):
    """
    CRITICAL: Removes or fills missing values (NaNs) to prevent model crashes.
    """
    # 1. Force columns to numeric (coerces errors to NaN)
    cols_to_fix = ['Speed', 'Direction', 'Temperature', 'Humidity']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. Fill small gaps (Interpolate)
    # If sensor missed 1 reading, guess it based on neighbors
    df = df.interpolate(method='linear', limit_direction='both')

    # 3. Drop any remaining rows that are still empty
    initial_len = len(df)
    df = df.dropna()
    dropped = initial_len - len(df)

    if dropped > 0:
        print(f"   -> [CLEANUP] Dropped {dropped} rows containing empty/bad data.")

    return df


def process_features(df):
    """Applies Physics & Vector Math."""

    # --- CLEANING STEP ADDED HERE ---
    df = clean_data(df)

    # 1. Ensure Time is standard
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    else:
        df['Timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='15min')

    # 2. Vertical Shear Extrapolation
    if 'Speed_10m' not in df.columns:
        df['Speed_10m'] = df['Speed']

    alpha = 0.25
    df['Speed_30m'] = df['Speed_10m'] * (30 / 10) ** alpha
    df['Speed_50m'] = df['Speed_10m'] * (50 / 10) ** alpha

    # 3. Air Density (Rho)
    if 'Rho' not in df.columns:
        if 'Temp' in df.columns and 'RH' in df.columns:
            # Complex Moist Air Density
            Tk = df['Temp'] + 273.15
            es = 6.1078 * 10 ** ((7.5 * df['Temp']) / (df['Temp'] + 237.3))
            pv = es * (df['RH'] / 100.0)
            pdry = 1013.25 - pv
            df['Rho'] = (pdry * 100) / (287.05 * Tk) + (pv * 100) / (461.495 * Tk)
        elif 'Temp' in df.columns:
            df['Rho'] = 1.225 * (288.15 / (df['Temp'] + 273.15))
        else:
            df['Rho'] = 1.225

            # 4. Power Density
    df['WPD'] = 0.5 * df['Rho'] * (df['Speed_10m'] ** 3)
    df['WPD_30m'] = 0.5 * df['Rho'] * (df['Speed_30m'] ** 3)

    # 5. Vector Components
    if 'Direction' in df.columns:
        df['Wx'] = df['Speed_10m'] * np.cos(np.deg2rad(df['Direction']))
        df['Wy'] = df['Speed_10m'] * np.sin(np.deg2rad(df['Direction']))
    else:
        df['Wx'] = 0
        df['Wy'] = df['Speed_10m']

    return df


def load_data():
    if os.path.exists(REAL_DATA_PATH):
        print(f"   -> [SUCCESS] Loading real data from: {REAL_DATA_PATH}")
        try:
            df = pd.read_csv(REAL_DATA_PATH)

            # Rename columns to match system
            rename_map = {'Temperature': 'Temp', 'Humidity': 'RH'}
            df = df.rename(columns=rename_map)

            # Run calculations
            df = process_features(df)
            return df

        except Exception as e:
            print(f"   -> [ERROR] Could not read CSV: {e}")
            raise e
    else:
        print(f"   -> [CRITICAL ERROR] File not found: {REAL_DATA_PATH}")
        raise FileNotFoundError("Check file path.")