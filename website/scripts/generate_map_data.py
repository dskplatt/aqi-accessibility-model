import pandas as pd
import json

# Paths
DATA_PATH = '../previous-datasets/JOINED-aqi-income-race-populationDensity/aqi-income-race-populationDensity-joined.csv'
RACE_META_PATH = '../previous-datasets/cleaned-datasets/race-by-county/cleaned-race-by-county.csv'
OUTPUT_PATH = 'public/data/aqi_map_data.json'

def generate_map_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    race_meta = pd.read_csv(RACE_META_PATH)
    
    print("Processing FIPS...")
    # Build FIPS lookup
    race_meta['FIPS'] = race_meta['GEO_ID'].astype(str).str.replace('0500000US', '')
    race_meta[['County_parse', 'State_parse']] = race_meta['County_Area'].str.split(', ', n=1, expand=True)
    race_meta['County_parse'] = race_meta['County_parse'].str.replace(' County', '', regex=False).str.replace(' Parish', '', regex=False)
    fips_lookup = race_meta.set_index(['State_parse', 'County_parse'])['FIPS'].to_dict()

    def get_fips(row):
        state, county = str(row['State']).strip(), str(row['County']).strip()
        if (state, county) in fips_lookup:
            return fips_lookup[(state, county)]
        # Fallback for slight naming mismatches
        for (s, c), f in fips_lookup.items():
            if s == state and c.startswith(county):
                return f
        return None

    df['FIPS'] = df.apply(get_fips, axis=1)
    
    # Drop rows without FIPS or target data
    map_df = df.dropna(subset=['FIPS', 'median_aqi']).copy()
    map_df['FIPS'] = map_df['FIPS'].astype(str).str.zfill(5)
    
    # Select only necessary columns to keep file size small
    output_cols = ['FIPS', 'State', 'County', 'median_aqi', 'population_density', 'Median_Household_Income']
    result = map_df[output_cols].to_dict(orient='records')
    
    print(f"Saving {len(result)} records to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(result, f)
    print("Done!")

if __name__ == "__main__":
    generate_map_data()
