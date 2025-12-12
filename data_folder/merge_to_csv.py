import pandas as pd
import glob
import os

# ==========================================
# CONFIGURATION
# ==========================================
SOURCE_FOLDER = "F:\HMEL_Project\data_folder"
OUTPUT_FILENAME = "merged_refinery_data.csv"


def get_param_type_from_filename(filename):
    """
    Decides what the data is based on the file name.
    """
    fname = filename.upper()
    if "SPEED" in fname: return "Speed"
    if "DIRECTION" in fname: return "Direction"
    if "TEMPERATURE" in fname: return "Temperature"
    if "HUMIDITY" in fname: return "Humidity"
    return None


def find_data_column(df, specific_header):
    """
    Finds the column with the numbers.
    Priority 1: The specific name (e.g., 'Wind_speed')
    Priority 2: 'Value' (The generic name found in your logs)
    """
    # Clean headers first (remove spaces)
    df.columns = df.columns.str.strip()

    if specific_header in df.columns:
        return specific_header
    elif "Value" in df.columns:
        return "Value"
    return None


def force_merge():
    search_path = os.path.join(SOURCE_FOLDER, "*.xlsx")
    files = glob.glob(search_path)

    if not files:
        print(f"[ERROR] No Excel files found in '{SOURCE_FOLDER}'.")
        return

    print(f"Found {len(files)} files. Processing using Filename Strategy...")

    df_list = []

    for f in files:
        try:
            filename = os.path.basename(f)
            # 1. Determine Type from Filename
            param_type = get_param_type_from_filename(filename)

            if not param_type:
                print(f"   [!] Skipping '{filename}': Filename must contain Speed, Temp, Direction, or Humidity.")
                continue

            # 2. Set the expected specific header (just in case)
            expected_headers = {
                'Speed': 'Wind_speed',
                'Direction': 'Wind_Direction',
                'Temperature': 'Temperature',
                'Humidity': 'Relative_Humidity'
            }
            specific_target = expected_headers.get(param_type)

            # 3. Read File
            df = pd.read_excel(f)

            # 4. Find the Data Column (Either 'Wind_speed' OR 'Value')
            data_col = find_data_column(df, specific_target)

            if data_col:
                print(f"   [+] '{filename}' identified as {param_type}. (Using column: '{data_col}')")

                # Standardize
                df = df.rename(columns={data_col: 'Value'})
                df['Parameter'] = param_type

                # Timestamp handling
                # Your logs show 'Timestamp' exists, so this should work safely
                if 'Timestamp' in df.columns:
                    subset = df[['Timestamp', 'Value', 'Parameter']].copy()
                    df_list.append(subset)
                else:
                    print(f"      [!] Timestamp column missing in {filename}")
            else:
                print(f"      [!] Could not find data column in {filename}. Headers: {list(df.columns)}")

        except Exception as e:
            print(f"   [!] Error reading '{filename}': {e}")

    # 5. Merge
    if df_list:
        print("\nAligning and Pivoting...")
        full_df = pd.concat(df_list, ignore_index=True)
        full_df['Timestamp'] = pd.to_datetime(full_df['Timestamp'])

        # PIVOT
        final_df = full_df.pivot_table(
            index='Timestamp',
            columns='Parameter',
            values='Value',
            aggfunc='mean'
        )

        final_df.to_csv(OUTPUT_FILENAME)
        print(f"\n[SUCCESS] merged_refinery_data.csv created with {len(final_df)} rows.")
        print(final_df.head())
    else:
        print("\n[FAILURE] No data collected.")


if __name__ == "__main__":
    force_merge()