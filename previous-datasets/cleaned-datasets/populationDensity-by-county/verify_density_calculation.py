import pandas as pd
import os

CWD = '/Users/williampantel/My Drive/coding-projects/datathon2026/cleaned-datasets/populationDensity-by-county'
FILE_PATH = os.path.join(CWD, 'cleaned-population-density-by-county.csv')

def verify():
    print(f"Verifying {FILE_PATH}...")
    
    if not os.path.exists(FILE_PATH):
        print("FAIL: File does not exist.")
        return False
        
    df = pd.read_csv(FILE_PATH)
    
    # 1. Row count check (should be around 3200)
    if len(df) < 3000:
        print(f"FAIL: Row count too low ({len(df)})")
        return False
    
    # 2. Schema check
    expected_cols = ['GEO_ID', 'County_Area', 'Total_Population', 'Land_Area_SqMi', 'population_density']
    if not all(col in df.columns for col in expected_cols):
        print(f"FAIL: Missing columns. Found: {df.columns.tolist()}")
        return False
        
    # 3. Data type check
    if not pd.api.types.is_numeric_dtype(df['population_density']):
        print("FAIL: population_density is not numeric.")
        return False
        
    # 4. Null check
    if df['population_density'].isnull().any():
        print("FAIL: Found null values in population_density.")
        return False
        
    # 5. Sanity check: New York County (Manhattan) should have high density
    manhattan = df[df['County_Area'] == 'New York County, New York']
    if not manhattan.empty:
        density = manhattan['population_density'].values[0]
        print(f"Sanity Check: Manhattan Density = {density:.2f}")
        if density < 10000: # Manhattan is extremely dense
            print("WARNING: Manhattan density seems low.")
            
    # 6. Sanity check: Los Angeles County
    la = df[df['County_Area'] == 'Los Angeles County, California']
    if not la.empty:
        la_density = la['population_density'].values[0]
        print(f"Sanity Check: LA County Density = {la_density:.2f}")

    print("SUCCESS: Population density data verified.")
    return True

if __name__ == "__main__":
    verify()
