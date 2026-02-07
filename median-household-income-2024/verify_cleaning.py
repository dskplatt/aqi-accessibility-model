import pandas as pd
import os

# Updated test script to verify logic and export CSV
file_path = '/Users/williampantel/My Drive/coding-projects/datathon2026/ACSST5Y2024.S1903_2026-02-07T134855/ACSST5Y2024.S1903-Data.csv'
output_file = '/Users/williampantel/My Drive/coding-projects/datathon2026/cleaned_median_household_income.csv'

try:
    # 1. Load data
    df = pd.read_csv(file_path, skiprows=[1])
    
    # 2. Select columns
    cols_to_keep = ['GEO_ID', 'NAME', 'S1903_C03_001E']
    df_cleaned = df[cols_to_keep].copy()

    # 3. Clean numeric data
    df_cleaned['S1903_C03_001E'] = df_cleaned['S1903_C03_001E'].astype(str).str.replace(',', '').str.replace('+', '')
    df_cleaned['S1903_C03_001E'] = pd.to_numeric(df_cleaned['S1903_C03_001E'], errors='coerce')

    # 4. Rename
    df_cleaned.rename(columns={
        'NAME': 'County_Area',
        'S1903_C03_001E': 'Median_Household_Income'
    }, inplace=True)

    # 5. Drop NA
    df_cleaned.dropna(subset=['Median_Household_Income'], inplace=True)

    # 6. Export
    df_cleaned.to_csv(output_file, index=False)
    
    print(f"Final shape: {df_cleaned.shape}")
    print(f"Cleaned data exported successfully to: {output_file}")
    
except Exception as e:
    print(f"Verification and export failed: {e}")
    exit(1)
