import pandas as pd
import os

CWD = '/Users/williampantel/My Drive/coding-projects/datathon2026/JOINED-aqi-populationDensity'
FILE_PATH = os.path.join(CWD, 'aqi_income_race_density_joined.csv')

def verify():
    print(f"Verifying Final Dataset: {FILE_PATH}...")
    
    if not os.path.exists(FILE_PATH):
        print("FAIL: File does not exist.")
        return False
        
    df = pd.read_csv(FILE_PATH)
    
    # 1. Row count check
    if len(df) < 800:
        print(f"FAIL: Row count too low ({len(df)})")
        return False
    
    # 2. Schema check - using actual column names found in previous run
    required_indicators = [
        'State', 'County', 'median_aqi', 
        '% White alone', '% Black or African American alone', '% Asian alone', '% Hispanic or Latino',
        'Median_Household_Income', 'population_density'
    ]
    if not all(col in df.columns for col in required_indicators):
        print(f"FAIL: Missing critical indicators. Found: {df.columns.tolist()}")
        return False
        
    # 3. Numeric check
    numeric_cols = ['median_aqi', 'Median_Household_Income', 'population_density']
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"FAIL: {col} is not numeric.")
            return False
            
    # 4. Null check on critical columns
    if df[numeric_cols].isnull().any().any():
        print(f"FAIL: Found null values in critical numeric columns.")
        return False

    print(f"SUCCESS: Final dataset verified with {len(df)} complete records.")
    return True

if __name__ == "__main__":
    verify()
