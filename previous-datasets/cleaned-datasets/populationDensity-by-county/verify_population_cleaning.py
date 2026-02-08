import pandas as pd
import os

# Settings
CLEANED_FILE = 'cleaned-datasets/populationDensity-by-county/cleaned-population-by-county.csv'

def verify():
    print("--- Starting Verification for Population Cleaning ---")
    
    if not os.path.exists(CLEANED_FILE):
        print(f"FAIL: Cleaned file {CLEANED_FILE} not found.")
        return False

    df = pd.read_csv(CLEANED_FILE)
    
    # 1. Row Count Check (Expect ~3,220 for ACS county datasets)
    if len(df) > 3000:
        print(f"PASS: Dataset contains {len(df)} records.")
    else:
        print(f"FAIL: Unexpectedly low record count: {len(df)}")
        return False

    # 2. Schema Check
    expected_cols = ['GEO_ID', 'County_Area', 'Total_Population']
    if list(df.columns) == expected_cols:
        print(f"PASS: Schema matches expectation: {expected_cols}")
    else:
        print(f"FAIL: Schema mismatch. Found: {list(df.columns)}")
        return False

    # 3. Numeric Check
    if df['Total_Population'].dtype in ['int64', 'float64']:
        # Ensure no nulls
        nulls = df['Total_Population'].isna().sum()
        if nulls == 0:
            print("PASS: Population column is numeric and contains no nulls.")
        else:
            print(f"FAIL: Found {nulls} null values in population column.")
            return False
    else:
        print(f"FAIL: Population column type is {df['Total_Population'].dtype}, expected numeric.")
        return False

    # 4. Content Sanity Check (Los Angeles County)
    la_county = df[df['County_Area'].str.contains("Los Angeles County, California", na=False)]
    if not la_county.empty:
        pop = la_county['Total_Population'].values[0]
        if pop > 9000000: # LA County is ~10M
            print(f"PASS: Content sanity check (LA County Population: {pop:,})")
        else:
            print(f"FAIL: LA County population suspiciously low: {pop:,}")
            return False
    else:
        print("FAIL: Could not find Los Angeles County record.")
        return False

    print("--- Verification SUCCESS! ---")
    return True

if __name__ == "__main__":
    verify()
