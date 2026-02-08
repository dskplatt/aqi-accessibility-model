import pandas as pd
import os

# Verification Settings
OUTPUT_DIR = 'JOINED-aqi-race'
OUTPUT_FILE = 'aqi_race_joined.csv'
FILE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
NOTEBOOK_REF = 'data_merging_analysis.ipynb'

def verify_data():
    print(f"--- Starting Verification for AQI-Race Join ---")
    
    # Check if file exists
    if not os.path.exists(FILE_PATH):
        print(f"FAIL: Output file {FILE_PATH} not found.")
        return False
    else:
        print(f"PASS: Output file found at {FILE_PATH}")

    # Load data
    try:
        df = pd.read_csv(FILE_PATH)
    except Exception as e:
        print(f"FAIL: Could not read CSV. Error: {e}")
        return False
    
    # 1. Check Row Count
    if len(df) > 0:
        print(f"PASS: Dataset contains {len(df)} records.")
    else:
        print(f"FAIL: Dataset is empty.")
        return False

    # 2. Check Essential Columns
    expected_cols = ['State', 'County', 'median_aqi', '% White alone', '% Black or African American alone']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    
    if not missing_cols:
        print(f"PASS: All essential columns present: {expected_cols}")
    else:
        print(f"FAIL: Missing columns: {missing_cols}")
        return False

    # 3. Check for specific records (Sanity Check)
    # The AQI dataset has Baldwin, Alabama
    sample = df[(df['State'] == 'Alabama') & (df['County'] == 'Baldwin')]
    if not sample.empty:
        print(f"PASS: Sample record found (Baldwin, Alabama).")
    else:
        print(f"FAIL: Sample record not found (Baldwin, Alabama). Check normalization logic.")
        return False

    # 4. Check for Nulls in Critical Columns
    null_counts = df[['State', 'County', 'median_aqi']].isnull().sum()
    if null_counts.sum() == 0:
        print("PASS: No null values in State, County, or median_aqi columns.")
    else:
        print(f"FAIL: Null values detected in critical columns: {null_counts}")
        return False

    print(f"--- Verification Success! refer to {NOTEBOOK_REF} for full analysis steps ---")
    return True

if __name__ == "__main__":
    verify_data()
