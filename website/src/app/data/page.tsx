import Header from '@/components/Header';
import Link from 'next/link';

export default function DataPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-gray-50">
      <Header />

      <main className="pt-32 pb-20 px-6">
        <div className="max-w-4xl mx-auto">
          {/* Intro */}
          <section className="mb-16">
            <Link href="/" className="text-accent-start hover:text-accent-end text-sm mb-6 inline-block transition-colors">
              ← Back to Home
            </Link>
            <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-4">
              Data Cleaning Process
            </h1>
            <p className="text-xl text-gray-600 leading-relaxed">
              We combined EPA Air Quality Index data with U.S. Census Bureau demographic data to build a dataset that relates race and socioeconomic status to air quality. Below is our step-by-step pipeline.
            </p>
          </section>

          {/* Pipeline Overview */}
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-slate-900 mb-4">Pipeline Overview</h2>
            <div className="bg-white rounded-xl p-6 border border-gray-200 shadow-sm">
              <ol className="space-y-3 text-gray-600 list-decimal list-inside">
                <li><strong className="text-slate-900">AQI cleaning</strong> — EPA dataset validation, consistency checks, sample weighting</li>
                <li><strong className="text-slate-900">Census cleaning</strong> — Income, race, population, land area (ACS 5-Year Estimates)</li>
                <li><strong className="text-slate-900">Population density</strong> — Total population ÷ land area by county</li>
                <li><strong className="text-slate-900">Region/Division</strong> — U.S. Census Bureau regions and divisions</li>
                <li><strong className="text-slate-900">Joins</strong> — AQI → Income → Race → Density → Region (inner joins on State + County)</li>
                <li><strong className="text-slate-900">Feature engineering</strong> — Log transforms, minority %, density interactions (black_density, minority_density)</li>
              </ol>
            </div>
          </section>

          {/* Step 1: AQI */}
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-slate-900 mb-2">1. AQI Data Cleaning</h2>
            <p className="text-gray-600 mb-4">
              The EPA Access to a Livable Planet dataset provides county-level air quality metrics. We validated uniqueness, consistency of day counts, and removed trailing whitespace. Because some counties have very few monitoring days (e.g., 19), we use <strong className="text-slate-900">sample weighting</strong> instead of excluding them—dropping low-coverage counties would bias our sample toward urban areas. Counties with fewer than 180 days of data are down-weighted proportionally.
            </p>
            <div className="space-y-4">
              <div className="bg-gray-50 rounded-xl border border-gray-200 overflow-hidden">
                <div className="px-4 py-2 bg-gray-100 border-b border-gray-200 text-sm text-gray-600 font-mono">
                  # Uniqueness check — (State, County, Year) should uniquely identify each row
                </div>
                <pre className="p-4 text-sm text-gray-700 font-mono overflow-x-auto">
{`key_cols = ['State', 'County', 'Year']
n_unique = df[key_cols].drop_duplicates().shape[0]
assert len(df) == n_unique, "Duplicate rows detected"
print("✓ No duplicates")`}
                </pre>
              </div>
              <div className="bg-gray-50 rounded-xl border border-gray-200 overflow-hidden">
                <div className="px-4 py-2 bg-gray-100 border-b border-gray-200 text-sm text-gray-600 font-mono">
                  # Sample weight: down-weight low-coverage counties (reference = 180 days)
                </div>
                <pre className="p-4 text-sm text-gray-700 font-mono overflow-x-auto">
{`REFERENCE_DAYS = 180
df['sample_weight'] = np.minimum(1.0, df['Days with AQI'] / REFERENCE_DAYS)
df['median_aqi'] = df['Median AQI']  # Our target variable`}
                </pre>
              </div>
            </div>
          </section>

          {/* Step 2: Census Income */}
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-slate-900 mb-2">2. Median Household Income (ACS S1903)</h2>
            <p className="text-gray-600 mb-4">
              Census ACS data uses special characters (<code className="text-accent-start">250,000+</code>, <code className="text-accent-start">*****</code>) for suppressed or top-coded values. We skip the label row to use standardized column codes, select the median income column, sanitize for numeric conversion, and drop rows with missing income.
            </p>
            <div className="bg-gray-50 rounded-xl border border-gray-200 overflow-hidden">
              <div className="px-4 py-2 bg-gray-100 border-b border-gray-200 text-sm text-gray-600 font-mono">
                # Load with skiprows=[1] for Census column codes; sanitize and convert to numeric
              </div>
              <pre className="p-4 text-sm text-gray-700 font-mono overflow-x-auto">
{`df = pd.read_csv('ACSST5Y2024.S1903-Data.csv', skiprows=[1])
df['Median_Household_Income'] = (
    df['S1903_C03_001E'].astype(str)
    .str.replace(',', '').str.replace('+', '')
)
df['Median_Household_Income'] = pd.to_numeric(df['Median_Household_Income'], errors='coerce')
df.dropna(subset=['Median_Household_Income'], inplace=True)`}
              </pre>
            </div>
          </section>

          {/* Step 3: Census Race */}
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-slate-900 mb-2">3. Race by County (ACS DP05)</h2>
            <p className="text-gray-600 mb-4">
              We extract six race percentage columns: Hispanic, White, Black, American Indian, Asian, Two or More Races. We filter to county-level geographies only (<code className="text-accent-start">GEO_ID</code> starts with <code className="text-accent-start">0500000US</code>), handle Census suppression symbols, and convert to numeric. These percentages are critical for our environmental justice analysis—the model later shows black_density and minority_density as top predictors.
            </p>
            <div className="bg-gray-50 rounded-xl border border-gray-200 overflow-hidden">
              <div className="px-4 py-2 bg-gray-100 border-b border-gray-200 text-sm text-gray-600 font-mono">
                # Filter to counties; clean Census symbols; convert to numeric
              </div>
              <pre className="p-4 text-sm text-gray-700 font-mono overflow-x-auto">
{`df = df[df['GEO_ID'].astype(str).str.startswith('0500000US')]
for col in race_cols:
    df[col] = df[col].astype(str).str.replace('*****', '').str.replace('(X)', '')
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(subset=race_cols, inplace=True)`}
              </pre>
            </div>
          </section>

          {/* Step 4: Population Density */}
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-slate-900 mb-2">4. Population Density</h2>
            <p className="text-gray-600 mb-4">
              Population comes from ACS B01003; land area from GEOINFO. We join them and compute <code className="text-accent-start">population_density = Total_Population / Land_Area_SqMi</code>. County names in Census data include suffixes like &quot;County&quot;, &quot;Parish&quot;, &quot;Borough&quot;—we strip these with regex so they match the AQI dataset format.
            </p>
            <div className="bg-gray-50 rounded-xl border border-gray-200 overflow-hidden">
              <div className="px-4 py-2 bg-gray-100 border-b border-gray-200 text-sm text-gray-600 font-mono">
                # Normalize county names for join; compute density
              </div>
              <pre className="p-4 text-sm text-gray-700 font-mono overflow-x-auto">
{`suffix_regex = r' (County|Parish|Borough|Census Area|Municipality|City)$'
density_df['County'] = density_df['County'].str.replace(suffix_regex, '', regex=True)
density_df['population_density'] = (
    density_df['Total_Population'] / density_df['Land_Area_SqMi']
)`}
              </pre>
            </div>
          </section>

          {/* Step 5: Region/Division */}
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-slate-900 mb-2">5. Region & Division</h2>
            <p className="text-gray-600 mb-4">
              We attach U.S. Census Bureau Region (Northeast, Midwest, South, West) and Division (e.g., Pacific, Mountain, East South Central) to each state. Geography is inluded in our model to account for geography-specific differences across regions— eg. the West has distinct AQI patterns (wildfire, climate). We normalize state names (strip, title case) before joining.
            </p>
            <div className="bg-gray-50 rounded-xl border border-gray-200 overflow-hidden">
              <div className="px-4 py-2 bg-gray-100 border-b border-gray-200 text-sm text-gray-600 font-mono">
                # Join regions on State
              </div>
              <pre className="p-4 text-sm text-gray-700 font-mono overflow-x-auto">
{`df_aqi['State'] = df_aqi['State'].str.strip().str.title()
df_joined = df_aqi.merge(
    df_regions[['State', 'Region', 'Division']],
    on='State', how='left'
)`}
              </pre>
            </div>
          </section>

          {/* Step 6: Joins */}
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-slate-900 mb-2">6. Joining All Datasets</h2>
            <p className="text-gray-600 mb-4">
              We perform inner joins on <code className="text-accent-start">State</code> and <code className="text-accent-start">County</code>. Each dataset applies the same normalization (split &quot;County Name, State&quot;, strip suffixes). Inner join keeps only counties present in all sources—we retain ~942 counties from the original ~978 AQI counties.
            </p>
            <div className="bg-gray-50 rounded-xl border border-gray-200 overflow-hidden">
              <div className="px-4 py-2 bg-gray-100 border-b border-gray-200 text-sm text-gray-600 font-mono">
                # Merge chain: AQI → Income → Race → Density → Region
              </div>
              <pre className="p-4 text-sm text-gray-700 font-mono overflow-x-auto">
{`# Example: AQI + Income join
joined = pd.merge(aqi_df, income_df, on=['State', 'County'], how='inner')
# Success rate: ~96% (940 / 978 AQI counties matched)`}
              </pre>
            </div>
          </section>

          {/* Step 7: Feature Engineering */}
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-slate-900 mb-2">7. Feature Engineering</h2>
            <p className="text-gray-600 mb-4">
              Before modeling, we add derived features. <strong className="text-slate-900">Log transforms</strong> (log_population_density, log_median_income) handle skew. <strong className="text-slate-900">total_minority_pct</strong> = 100 − % White. The most important features in our model are <strong className="text-accent-start">black_density</strong> and <strong className="text-accent-start">minority_density</strong>—interactions of race percentages with population density. These capture how racial composition and urbanization jointly relate to air quality.
            </p>
            <div className="bg-gray-50 rounded-xl border border-gray-200 overflow-hidden">
              <div className="px-4 py-2 bg-gray-100 border-b border-gray-200 text-sm text-gray-600 font-mono">
                # Top feature engineering: density × race interactions (top predictors in model)
              </div>
              <pre className="p-4 text-sm text-gray-700 font-mono overflow-x-auto">
{`df['total_minority_pct'] = 100 - df['% White alone']
df['log_population_density'] = np.log1p(df['population_density'])
df['log_median_income'] = np.log1p(df['Median_Household_Income'])

# Race × density interactions — strongest predictors of AQI
df['black_density'] = df['% Black or African American alone'] * df['population_density'] / 100
df['minority_density'] = df['total_minority_pct'] * df['population_density'] / 100
df['hispanic_density'] = df['% Hispanic or Latino'] * df['population_density'] / 100
df['asian_density'] = df['% Asian alone'] * df['population_density'] / 100`}
              </pre>
            </div>
          </section>

          {/* Final Output */}
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-slate-900 mb-2">Final Dataset</h2>
            <p className="text-gray-600 mb-4">
              The resulting dataset has <strong className="text-slate-900">942 counties</strong>, <strong className="text-slate-900">39+ features</strong>, and <strong className="text-slate-900">median_aqi</strong> as the target. We use it to train an XGBoost model that predicts air quality from race, income, population density, and geography—demonstrating that clean air is not equally accessible and that exposure is associated with demographic factors.
            </p>
            <div className="bg-emerald-50 rounded-xl p-6 border border-emerald-200">
              <p className="text-gray-700">
                <strong className="text-slate-900">Key insight:</strong> The model ranks <span className="text-accent-start font-semibold">black_density</span> and <span className="text-accent-start font-semibold">minority_density</span> as the top non-geographic features. This supports our thesis: air quality is systematically related to who lives where—a clear environmental justice signal.
              </p>
            </div>
          </section>

          <div className="flex gap-4">
            <Link href="/process" className="px-6 py-3 bg-accent-end text-white rounded-full font-medium hover:bg-accent-start transition-colors">
              View Modeling Process
            </Link>
            <Link href="/findings" className="px-6 py-3 bg-white text-slate-900 rounded-full font-medium border border-gray-300 hover:border-gray-400 hover:bg-gray-50 transition-colors">
              View Findings
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}
