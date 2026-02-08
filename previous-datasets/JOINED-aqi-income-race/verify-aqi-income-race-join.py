import pandas as pd
import os

# Verification Settings
OUTPUT_FILE = 'JOINED-aqi-income-race/aqi_income_race_joined.csv'

def verify_data():
    print(f"--- Starting Final Verification (AQI + Race + Income) ---")
    
    if not os.path.exists(OUTPUT_FILE):
        print(f"FAIL: Output file {OUTPUT_FILE} not found.")
        return False

    df = pd.read_csv(OUTPUT_FILE)
    
    # 1. Row Count Check
    if len(df) > 900: # We expect around 940 based on previous joins
        print(f"PASS: Dataset contains {len(df)} records.")
    else:
        print(f"FAIL: Dataset has suspiciously low row count: {len(df)}")
        return False

    # 2. Key Columns Check
    required_cols = [
        'State', 'County', 'median_aqi', 
        '% White alone', '% Black or African American alone',
        'Median_Household_Income'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if not missing:
        print(f"PASS: All key indicator columns present.")
    else:
        print(f"FAIL: Missing columns: {missing}")
        return False

    # 3. Null Check
    critical_cols = ['State', 'County', 'median_aqi', 'Median_Household_Income', '% White alone']
    nulls = df[critical_cols].isnull().sum()
    if nulls.sum() == 0:
        print("PASS: No nulls in critical joined columns.")
    else:
        print(f"FAIL: Null values found:\n{nulls}")
        return False

    # 4. Sanity Check (Baldwin, Alabama)
    baldwin = df[(df['State'] == 'Alabama') & (df['County'] == 'Baldwin')]
    if not baldwin.empty:
        print("PASS: Baldwin, Alabama record found.")
        print(f" - AQI: {baldwin['median_aqi'].values[0]}")
        print(f" - Income: ${baldwin['Median_Household_Income'].values[0]}")
        print(f" - % White: {baldwin['% White alone'].values[0]}%")
    else:
        print("FAIL: Baldwin, Alabama record missing.")
        return False

    print(f"--- Final Verification SUCCESS! ---")
    return True

if __name__ == "__main__":
    verify_data()
