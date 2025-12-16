import pandas as pd
import numpy as np
import os
import torch
from sklearn.preprocessing import MinMaxScaler
import config

# --- CONFIGURATION ---
# UPDATE THIS PATH TO MATCH YOUR ACTUAL CSV FILE LOCATION
REAL_DATA_PATH = r"F:\HMEL_Project\data_folder\merged_refinery_data.csv"


def process_features(df):
    """
    Cleans raw data and adds Physics + CatBoost features.
    """
    # 1. Standard cleaning
    cols = ['Speed', 'Direction', 'Temperature', 'Humidity']
    for c in cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

    # Interpolate small gaps, drop massive empty chunks
    df = df.interpolate(method='linear').dropna()

    # 2. Physics Features
    df['Rho'] = 1.225  # Simplified air density
    df['Speed_10m'] = df['Speed']
    # Hellmann Law extrapolation
    df['Speed_30m'] = df['Speed'] * (30 / 10) ** 0.25
    df['Speed_50m'] = df['Speed'] * (50 / 10) ** 0.25
    # Power Density
    df['WPD'] = 0.5 * df['Rho'] * df['Speed'] ** 3

    # 3. Features for CatBoost (Context)
    # Cyclical Time
    if 'Timestamp' in df.columns:
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Timestamp'].dt.hour / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Timestamp'].dt.hour / 24)

    # Lag Features (The "Memory" for CatBoost)
    # We shift the speed column down to create "Past Speed" columns
    df['Lag_1'] = df['Speed'].shift(1)  # Speed 15 mins ago
    df['Lag_4'] = df['Speed'].shift(4)  # Speed 1 hour ago

    # Drop the first few rows that now have NaNs due to shifting
    return df.dropna()


def prepare_hybrid_tensors(df, seq_len=24):
    """
    Prepares data for the Hybrid Model:
    1. X_seq: Tensor [Samples, Seq_Len, 1] for BiLSTM
    2. X_tab: Array [Samples, Features] for CatBoost (aligned)
    3. y: Tensor [Samples, 1] Targets
    4. scaler: The scaler used for Speed
    """
    # Safety check for empty dataframe
    if len(df) < seq_len + 1:
        raise ValueError("Dataframe is too small for the requested sequence length.")

    data = df['Speed'].values.reshape(-1, 1)

    # Tabular features for CatBoost
    # Ensure these columns exist
    req_cols = ['Hour_Sin', 'Hour_Cos', 'Lag_1', 'Lag_4', 'Direction', 'Temperature']
    for col in req_cols:
        if col not in df.columns:
            df[col] = 0  # Fill missing cols with 0 if necessary

    tab_data = df[req_cols].values

    # Scale Speed (BiLSTM learns better on 0-1 data)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X_s, X_t, y = [], [], []

    for i in range(len(data) - seq_len):
        # BiLSTM Input: Sequence of last 'seq_len' speeds
        X_s.append(data_scaled[i: i + seq_len])

        # CatBoost Input: Features at the END of the sequence (to predict next step)
        X_t.append(tab_data[i + seq_len - 1])

        # Target: The very next speed
        y.append(data_scaled[i + seq_len])

    return (
        torch.FloatTensor(np.array(X_s)),
        np.array(X_t),
        torch.FloatTensor(np.array(y)),
        scaler
    )


def load_data():
    """
    Main function called by other scripts to get the data.
    """
    if os.path.exists(REAL_DATA_PATH):
        print(f"   -> Loading data from {REAL_DATA_PATH}")
        df = pd.read_csv(REAL_DATA_PATH)

        # Ensure Timestamp is datetime object
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        else:
            # Create dummy timestamp if missing
            print("   -> [WARN] No Timestamp column found. Generating index.")
            df['Timestamp'] = pd.date_range(start='1/1/2024', periods=len(df), freq='15T')

        # Standardize column names
        rename_map = {'Temperature': 'Temp', 'Humidity': 'RH'}
        df = df.rename(columns=rename_map)

        return process_features(df)
    else:
        print(f"   -> [ERROR] File not found: {REAL_DATA_PATH}")
        raise FileNotFoundError(f"Check the path in data_loader.py")