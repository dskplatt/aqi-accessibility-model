import pandas as pd
import os
import sys

def verify_data():
    file_path = 'aqi_income_joined.csv'
    
    print(f"--- Starting Verification for {file_path} ---")
    
    # 1. Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found.")
        return False
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False
    
    # 2. Check for required columns
    required_columns = ['State', 'County', 'Year', 'median_aqi', 'Median_Household_Income']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Error: Missing columns: {missing_cols}")
        return False
    else:
        print("✅ All required columns present.")
    
    # 3. Check for null values
    null_counts = df[required_columns].isnull().sum()
    if null_counts.sum() > 0:
        print("⚠️ Warning: Found null values:")
        print(null_counts[null_counts > 0])
    else:
        print("✅ No null values in target columns.")
        
    # 4. Success stats
    print(f"✅ Verification Complete. Total rows: {len(df)}")
    print("--- Sample Data ---")
    print(df.head())
    
    return True

if __name__ == "__main__":
    success = verify_data()
    if not success:
        sys.exit(1)
