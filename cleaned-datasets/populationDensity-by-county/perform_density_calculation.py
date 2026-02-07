import pandas as pd
import os

# File Paths
CWD = '/Users/williampantel/My Drive/coding-projects/datathon2026/cleaned-datasets/populationDensity-by-county'
POPULATION_PATH = os.path.join(CWD, 'cleaned-population-by-county.csv')
LAND_AREA_PATH = os.path.join(CWD, 'GEOINFO2023.GEOINFO-2026-02-07T233836.csv')
OUTPUT_PATH = os.path.join(CWD, 'cleaned-population-density-by-county.csv')

def perform_calculation():
    print("Loading datasets...")
    pop_df = pd.read_csv(POPULATION_PATH)
    area_df = pd.read_csv(LAND_AREA_PATH)
    
    print("Cleaning land area data...")
    # Select key columns
    area_df_cleaned = area_df[['Geographic Area Name (NAME)', 'Area (Land, in square miles) (AREALAND_SQMI)']].copy()
    
    # Rename for consistency
    area_df_cleaned.rename(columns={
        'Geographic Area Name (NAME)': 'County_Area',
        'Area (Land, in square miles) (AREALAND_SQMI)': 'Land_Area_SqMi'
    }, inplace=True)
    
    # Sanitize numeric values
    area_df_cleaned['Land_Area_SqMi'] = area_df_cleaned['Land_Area_SqMi'].astype(str).str.replace(',', '')
    area_df_cleaned['Land_Area_SqMi'] = pd.to_numeric(area_df_cleaned['Land_Area_SqMi'], errors='coerce')
    
    # Drop rows with missing area
    area_df_cleaned.dropna(subset=['Land_Area_SqMi'], inplace=True)
    
    print("Merging datasets and calculating density...")
    merged_df = pd.merge(pop_df, area_df_cleaned, on='County_Area', how='inner')
    
    # Calculate density
    merged_df['population_density'] = merged_df['Total_Population'] / merged_df['Land_Area_SqMi']
    
    print(f"Exporting {len(merged_df)} records to {OUTPUT_PATH}...")
    merged_df.to_csv(OUTPUT_PATH, index=False)
    print("Export complete.")

if __name__ == "__main__":
    perform_calculation()
