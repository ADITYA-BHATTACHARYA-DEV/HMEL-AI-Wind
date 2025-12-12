# data_loader.py
import pandas as pd
import os
import config


def load_and_process_data():
    """
    Loads the clean merged CSV directly.
    """
    file_path = config.DATA_FILE_PATH

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CRITICAL: The file '{file_path}' was not found. Did you run the merge script?")

    print(f"Loading data from: {file_path}")

    try:
        # 1. Read CSV
        df = pd.read_csv(file_path)

        # 2. Rename Columns to Internal Standard
        # This converts 'Humidity' -> 'RH', 'Temperature' -> 'Temp', etc.
        df = df.rename(columns=config.COLUMN_MAPPING)

        # 3. Parse Timestamp
        df[config.COL_TIMESTAMP] = pd.to_datetime(df[config.COL_TIMESTAMP])

        # 4. Filter for Valid Rows
        # Ensure we have all 4 required physics parameters
        required_cols = ['Speed', 'Temp', 'RH', 'Direction']

        # Check if columns exist
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}. Check config.COLUMN_MAPPING.")

        # Drop rows with empty values
        initial_len = len(df)
        clean_df = df.dropna(subset=required_cols).reset_index(drop=True)
        dropped_count = initial_len - len(clean_df)

        print(f"Success! Loaded {len(clean_df)} timestamps. (Dropped {dropped_count} invalid rows)")

        return clean_df

    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise e