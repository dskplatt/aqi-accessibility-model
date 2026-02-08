import pandas as pd
import os
import re

# File Paths
CWD = '/Users/williampantel/My Drive/coding-projects/datathon2026/JOINED-aqi-populationDensity'
BASE_DIR = '/Users/williampantel/My Drive/coding-projects/datathon2026'
AQI_INCOME_RACE_PATH = os.path.join(BASE_DIR, 'JOINED-aqi-income-race', 'aqi-income-race-joined.csv')
DENSITY_PATH = os.path.join(BASE_DIR, 'cleaned-datasets', 'populationDensity-by-county', 'cleaned-population-density-by-county.csv')
OUTPUT_PATH = os.path.join(CWD, 'aqi_income_race_density_joined.csv')

def perform_integration():
    print("Loading datasets...")
    main_df = pd.read_csv(AQI_INCOME_RACE_PATH)
    density_df = pd.read_csv(DENSITY_PATH)
    
    print("Normalizing density data geographic fields...")
    # Split County_Area
    density_df[['County', 'State']] = density_df['County_Area'].str.split(', ', expand=True)
    density_df['County'] = density_df['County'].str.strip()
    density_df['State'] = density_df['State'].str.strip()
    
    # Strip common suffixes (County, Parish, etc.)
    suffix_regex = r' (County|Parish|Borough|Census Area|Municipality|City and Borough|City)$'
    density_df['County'] = density_df['County'].str.replace(suffix_regex, '', regex=True, flags=re.IGNORECASE)
    
    # Select relevant columns
    density_join_df = density_df[['State', 'County', 'Total_Population', 'Land_Area_SqMi', 'population_density']].copy()
    
    print("Performing final inner join...")
    final_df = pd.merge(main_df, density_join_df, on=['State', 'County'], how='inner')
    
    print(f"Exporting {len(final_df)} records to {OUTPUT_PATH}...")
    final_df.to_csv(OUTPUT_PATH, index=False)
    print("Integration complete.")

if __name__ == "__main__":
    perform_integration()
