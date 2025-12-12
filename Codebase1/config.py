# config.py

# Path to your merged CSV file (Use absolute path or relative if in same folder)
# NOTE: Use 'r' before the string to handle backslashes on Windows
DATA_FILE_PATH = r"F:\HMEL_Project\data_folder\merged_refinery_data.csv"

# CSV COLUMN MAPPING
# Key (Left) = The exact header in your merged CSV file
# Value (Right) = The internal variable name the AI expects (Speed, Temp, RH, Direction)
COLUMN_MAPPING = {
    'Speed': 'Speed',
    'Temperature': 'Temp',
    'Humidity': 'RH',
    'Direction': 'Direction'
}

# Common Columns
COL_TIMESTAMP = "Timestamp"

# Physics Constants (Refinery specific)
RD_GAS_CONSTANT = 287.058   # Dry air
RV_GAS_CONSTANT = 461.495   # Water vapor
STD_PRESSURE = 1013.25      # Standard atmospheric pressure (hPa)

# AI Settings
CLUSTERS = 3                # Regimes: Low, Medium, High
RANDOM_SEED = 42