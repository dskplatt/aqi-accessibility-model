import Header from '@/components/Header';
import Link from 'next/link';

export default function ProcessPage() {
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
              Modeling Process
            </h1>
            <p className="text-xl text-gray-600 leading-relaxed">
              We built an XGBoost regression model to predict county-level median AQI from demographic and socioeconomic features. Through five iterative rounds—adding features, engineering interactions, and tuning hyperparameters—we improved from an R² of −0.04 to 0.534.
            </p>
          </section>

          {/* Journey Overview */}
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-slate-900 mb-4">Model Evolution</h2>
            <div className="bg-white rounded-xl p-6 border border-gray-200 shadow-sm">
              <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
                {[
                  { label: 'Income Only', r2: '−0.04', color: 'bg-red-100 text-red-700 border-red-200' },
                  { label: '+ Race', r2: '0.105', color: 'bg-orange-100 text-orange-700 border-orange-200' },
                  { label: '+ Density', r2: '0.212', color: 'bg-yellow-100 text-yellow-700 border-yellow-200' },
                  { label: '+ Region', r2: '0.235', color: 'bg-blue-100 text-blue-700 border-blue-200' },
                  { label: 'Final Model', r2: '0.534', color: 'bg-emerald-100 text-emerald-700 border-emerald-200' },
                ].map((step, i) => (
                  <div key={i} className={`rounded-lg p-4 border text-center ${step.color}`}>
                    <p className="text-xs font-medium uppercase tracking-wide opacity-80">R²</p>
                    <p className="text-2xl font-bold">{step.r2}</p>
                    <p className="text-sm font-medium mt-1">{step.label}</p>
                  </div>
                ))}
              </div>
              <div className="hidden md:flex items-center justify-between mt-3 px-6">
                {[0, 1, 2, 3].map((i) => (
                  <span key={i} className="text-gray-300 text-lg">→</span>
                ))}
              </div>
            </div>
          </section>

          {/* Iteration 1: Income Only */}
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-slate-900 mb-2">1. Income Only — Baseline</h2>
            <p className="text-gray-600 mb-4">
              Our first model used a single feature: <strong className="text-slate-900">Median Household Income</strong>. We used a stratified 80/20 split by AQI quartiles and sample weights to down-weight low-coverage counties. After GridSearchCV with 5-fold stratified cross-validation across 36 parameter combinations, the best model achieved an R² of <strong className="text-red-600">−0.04</strong>—income alone has essentially zero predictive power for air quality.
            </p>
            <div className="space-y-4">
              <div className="bg-gray-50 rounded-xl border border-gray-200 overflow-hidden">
                <div className="px-4 py-2 bg-gray-100 border-b border-gray-200 text-sm text-gray-600 font-mono">
                  # Stratified split + GridSearchCV with sample weights
                </div>
                <pre className="p-4 text-sm text-gray-700 font-mono overflow-x-auto">
{`X = df[['Median_Household_Income']]
y = df['median_aqi']
df['aqi_stratum'] = pd.qcut(df['median_aqi'], q=4, labels=['Q1','Q2','Q3','Q4'])

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.2, random_state=42, stratify=strata
)

param_grid = {
    'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1], 'min_child_weight': [1, 3],
}
grid_search = GridSearchCV(xgb_model, param_grid, cv=cv_splits,
                           scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train, sample_weight=w_train)`}
                </pre>
              </div>
              <div className="bg-red-50 rounded-xl p-4 border border-red-200">
                <p className="text-sm text-gray-700">
                  <strong className="text-slate-900">Result:</strong> R² = −0.04 | RMSE = 10.42 | MAE = 6.93 — Income alone cannot predict AQI. We also tested a geographic holdout (train on East/Midwest, test on West) and got R² = −0.08, confirming income-to-AQI relationships don&apos;t generalize across regions.
                </p>
              </div>
            </div>
          </section>

          {/* Iteration 2: + Race */}
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-slate-900 mb-2">2. Adding Race — Environmental Justice Signal</h2>
            <p className="text-gray-600 mb-4">
              Next we added <strong className="text-slate-900">% Black or African American</strong> and <strong className="text-slate-900">% Hispanic or Latino</strong>. With a random 80/20 split and 5-fold CV, R² jumped to <strong className="text-orange-600">0.105</strong>. Feature importance revealed that <strong className="text-accent-start">% Black</strong> was the single strongest predictor (45.3%), followed by % Hispanic (31.4%), with income trailing at 23.4%. A race-only model (no income) still achieved R² = 0.085—race carries more signal than income alone.
            </p>
            <div className="space-y-4">
              <div className="bg-gray-50 rounded-xl border border-gray-200 overflow-hidden">
                <div className="px-4 py-2 bg-gray-100 border-b border-gray-200 text-sm text-gray-600 font-mono">
                  # Income + Race features with 5-fold CV
                </div>
                <pre className="p-4 text-sm text-gray-700 font-mono overflow-x-auto">
{`feature_cols = ['Median_Household_Income',
                '% Black or African American alone',
                '% Hispanic or Latino']

grid_search.fit(X_train, y_train, sample_weight=w_train)
# Best params: lr=0.05, max_depth=3, min_child_weight=3, n_estimators=100

# Feature importance:
#   % Black or African American alone    0.453
#   % Hispanic or Latino                 0.314
#   Median_Household_Income              0.234`}
                </pre>
              </div>
              <div className="bg-orange-50 rounded-xl p-4 border border-orange-200">
                <p className="text-sm text-gray-700">
                  <strong className="text-slate-900">Result:</strong> R² = 0.105 | RMSE = 9.78 | MAE = 6.78 — Racial composition is a stronger predictor of air quality than income. This is the first environmental justice signal in our model.
                </p>
              </div>
            </div>
          </section>

          {/* Iteration 3: + Density */}
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-slate-900 mb-2">3. Adding Population Density</h2>
            <p className="text-gray-600 mb-4">
              Population density (people per square mile) nearly doubled R² to <strong className="text-yellow-600">0.212</strong>. Density became the second most important feature (29.4%), just behind % Black (32.3%). A density-only model achieved R² = 0.149, confirming that urbanization independently relates to AQI. The features interact: densely populated minority communities face compounding pollution exposure.
            </p>
            <div className="space-y-4">
              <div className="bg-gray-50 rounded-xl border border-gray-200 overflow-hidden">
                <div className="px-4 py-2 bg-gray-100 border-b border-gray-200 text-sm text-gray-600 font-mono">
                  # Full model: Income + Race + Population Density
                </div>
                <pre className="p-4 text-sm text-gray-700 font-mono overflow-x-auto">
{`feature_cols = ['Median_Household_Income',
                '% Black or African American alone',
                '% Hispanic or Latino',
                'population_density']

# Feature importance after adding density:
#   % Black or African American alone    0.323
#   population_density                   0.294
#   % Hispanic or Latino                 0.217
#   Median_Household_Income              0.166`}
                </pre>
              </div>
              <div className="bg-yellow-50 rounded-xl p-4 border border-yellow-200">
                <p className="text-sm text-gray-700">
                  <strong className="text-slate-900">Result:</strong> R² = 0.212 | RMSE = 9.51 | MAE = 6.77 — Density is the strongest single predictor, and combined with race provides a clear picture: air quality is worst in dense, majority-minority counties.
                </p>
              </div>
            </div>
          </section>

          {/* Iteration 4: + Region */}
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-slate-900 mb-2">4. Adding Geographic Region</h2>
            <p className="text-gray-600 mb-4">
              One-hot encoding the U.S. Census <strong className="text-slate-900">Region</strong> (Northeast, Midwest, South, West) added geographic context. The West has distinct AQI patterns driven by wildfires and arid climate; the industrial Midwest has different pollution sources. This lifted R² to <strong className="text-blue-600">0.235</strong>. At this point, the model explained about a quarter of AQI variance—useful, but with plenty of room to improve through feature engineering.
            </p>
            <div className="space-y-4">
              <div className="bg-gray-50 rounded-xl border border-gray-200 overflow-hidden">
                <div className="px-4 py-2 bg-gray-100 border-b border-gray-200 text-sm text-gray-600 font-mono">
                  # One-hot encode Region (drop_first=True)
                </div>
                <pre className="p-4 text-sm text-gray-700 font-mono overflow-x-auto">
{`region_dummies = pd.get_dummies(df['Region'], prefix='Region', drop_first=True)
df = pd.concat([df, region_dummies], axis=1)

feature_cols = base_cols + [c for c in df.columns if c.startswith('Region_')]
# Features now: Income, % Black, % Hispanic, density, Region_Midwest,
#               Region_South, Region_West`}
                </pre>
              </div>
              <div className="bg-blue-50 rounded-xl p-4 border border-blue-200">
                <p className="text-sm text-gray-700">
                  <strong className="text-slate-900">Result:</strong> R² = 0.235 — Geography matters. Western counties show systematically different AQI patterns. But with only raw features, we&apos;re leaving performance on the table.
                </p>
              </div>
            </div>
          </section>

          {/* Divider */}
          <div className="border-t border-gray-200 my-16" />

          {/* Final Model Section */}
          <section className="mb-16">
            <h2 className="text-3xl font-bold text-slate-900 mb-2">5. Final Model — Feature Engineering & Tuning</h2>
            <p className="text-gray-600 mb-4">
              The jump from R² 0.235 to <strong className="text-emerald-600">0.534</strong> came from three changes: extensive feature engineering, RobustScaler preprocessing, and a broader hyperparameter search across 192 combinations. This more than doubled the model&apos;s explained variance.
            </p>
          </section>

          {/* 5a: Feature Engineering */}
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-slate-900 mb-2">5a. Feature Engineering</h2>
            <p className="text-gray-600 mb-4">
              We created 11 new features capturing interactions, non-linear relationships, and domain-specific patterns. The most impactful were <strong className="text-accent-start">demographic-density interactions</strong>—multiplying race percentages by population density to model how racial composition and urbanization jointly affect air quality.
            </p>
            <div className="space-y-4">
              <div className="bg-gray-50 rounded-xl border border-gray-200 overflow-hidden">
                <div className="px-4 py-2 bg-gray-100 border-b border-gray-200 text-sm text-gray-600 font-mono">
                  # Demographic-density interactions (top predictors in final model)
                </div>
                <pre className="p-4 text-sm text-gray-700 font-mono overflow-x-auto">
{`df['minority_density'] = df['total_minority_pct'] * df['population_density'] / 100
df['black_density'] = df['% Black or African American alone'] * df['population_density'] / 100
df['hispanic_density'] = df['% Hispanic or Latino'] * df['population_density'] / 100
df['asian_density'] = df['% Asian alone'] * df['population_density'] / 100`}
                </pre>
              </div>
              <div className="bg-gray-50 rounded-xl border border-gray-200 overflow-hidden">
                <div className="px-4 py-2 bg-gray-100 border-b border-gray-200 text-sm text-gray-600 font-mono">
                  # Economic and polynomial features
                </div>
                <pre className="p-4 text-sm text-gray-700 font-mono overflow-x-auto">
{`df['income_per_capita'] = df['Median_Household_Income'] / (df['Total_Population'] + 1)
df['urban_income'] = df['population_density'] * df['Median_Household_Income'] / 1e6
df['pop_density_squared'] = df['population_density'] ** 2
df['log_density_squared'] = df['log_population_density'] ** 2`}
                </pre>
              </div>
              <div className="bg-gray-50 rounded-xl border border-gray-200 overflow-hidden">
                <div className="px-4 py-2 bg-gray-100 border-b border-gray-200 text-sm text-gray-600 font-mono">
                  # Ratio and income-demographic interactions
                </div>
                <pre className="p-4 text-sm text-gray-700 font-mono overflow-x-auto">
{`df['white_to_minority_ratio'] = df['% White alone'] / (df['total_minority_pct'] + 0.1)
df['income_to_density_ratio'] = df['Median_Household_Income'] / (df['population_density'] + 1)
df['minority_income'] = df['total_minority_pct'] * df['Median_Household_Income'] / 1e5
df['white_income'] = df['% White alone'] * df['Median_Household_Income'] / 1e5`}
                </pre>
              </div>
            </div>
          </section>

          {/* 5b: Preprocessing */}
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-slate-900 mb-2">5b. RobustScaler Preprocessing</h2>
            <p className="text-gray-600 mb-4">
              Our features span wildly different scales—population density ranges from 0.12 to 71,916, while race percentages are 0–100. We switched from StandardScaler to <strong className="text-slate-900">RobustScaler</strong>, which uses median and interquartile range instead of mean and standard deviation. This makes scaling resistant to outliers, which is critical given extreme density values in places like New York County (Manhattan).
            </p>
            <div className="bg-gray-50 rounded-xl border border-gray-200 overflow-hidden">
              <div className="px-4 py-2 bg-gray-100 border-b border-gray-200 text-sm text-gray-600 font-mono">
                # RobustScaler for outlier-resistant scaling
              </div>
              <pre className="p-4 text-sm text-gray-700 font-mono overflow-x-auto">
{`from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # fit only on training data`}
              </pre>
            </div>
          </section>

          {/* 5c: Hyperparameter Tuning */}
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-slate-900 mb-2">5c. Hyperparameter Tuning</h2>
            <p className="text-gray-600 mb-4">
              We expanded the hyperparameter grid to 192 combinations, adding regularization parameters (<code className="text-accent-start">reg_alpha</code>, <code className="text-accent-start">reg_lambda</code>, <code className="text-accent-start">gamma</code>) and subsampling (<code className="text-accent-start">subsample</code>, <code className="text-accent-start">colsample_bytree</code>) to prevent overfitting on our 942-county dataset. GridSearchCV with 5-fold cross-validation selected the best configuration.
            </p>
            <div className="bg-gray-50 rounded-xl border border-gray-200 overflow-hidden">
              <div className="px-4 py-2 bg-gray-100 border-b border-gray-200 text-sm text-gray-600 font-mono">
                # Expanded param grid: 192 combinations with regularization
              </div>
              <pre className="p-4 text-sm text-gray-700 font-mono overflow-x-auto">
{`param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [200, 300],
    'min_child_weight': [3, 5],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'gamma': [0, 0.1],
    'reg_alpha': [0.5, 1],     # L1 regularization
    'reg_lambda': [1, 2]       # L2 regularization
}

grid_search = GridSearchCV(
    XGBRegressor(random_state=42, n_jobs=-1),
    param_grid, cv=5, scoring='r2', n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)`}
              </pre>
            </div>
          </section>

          {/* 5d: Results */}
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-slate-900 mb-2">5d. Final Model Results</h2>
            <p className="text-gray-600 mb-6">
              The tuned model achieved R² = <strong className="text-emerald-600">0.534</strong> on the held-out test set, with 5-fold cross-validation confirming consistent performance. Feature importance analysis revealed that our engineered interaction features dominated the top positions.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="bg-emerald-50 rounded-xl p-5 border border-emerald-200 text-center">
                <p className="text-sm text-emerald-700 font-medium">R²</p>
                <p className="text-3xl font-bold text-emerald-700">0.534</p>
                <p className="text-xs text-emerald-600 mt-1">53.4% variance explained</p>
              </div>
              <div className="bg-emerald-50 rounded-xl p-5 border border-emerald-200 text-center">
                <p className="text-sm text-emerald-700 font-medium">Improvement</p>
                <p className="text-3xl font-bold text-emerald-700">+127%</p>
                <p className="text-xs text-emerald-600 mt-1">over iteration 4 (R² 0.235)</p>
              </div>
              <div className="bg-emerald-50 rounded-xl p-5 border border-emerald-200 text-center">
                <p className="text-sm text-emerald-700 font-medium">CV Stability</p>
                <p className="text-3xl font-bold text-emerald-700">5-fold</p>
                <p className="text-xs text-emerald-600 mt-1">consistent across folds</p>
              </div>
            </div>

            <div className="bg-gray-50 rounded-xl border border-gray-200 overflow-hidden">
              <div className="px-4 py-2 bg-gray-100 border-b border-gray-200 text-sm text-gray-600 font-mono">
                # Final evaluation
              </div>
              <pre className="p-4 text-sm text-gray-700 font-mono overflow-x-auto">
{`best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

print(f"R²:   {r2_score(y_test, y_pred):.4f}")     # 0.534
print(f"RMSE: {np.sqrt(mse(y_test, y_pred)):.4f}")
print(f"MAE:  {mae(y_test, y_pred):.4f}")

# Cross-validation on final model
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"Mean CV R²: {cv_scores.mean():.4f} ± {cv_scores.std() * 2:.4f}")`}
              </pre>
            </div>
          </section>

          {/* Feature Importance */}
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-slate-900 mb-2">Top Features</h2>
            <p className="text-gray-600 mb-4">
              The model&apos;s feature importance rankings validate both our feature engineering and the environmental justice thesis. Engineered interaction features—particularly those combining race with density—dominate the top positions. Geographic divisions also rank highly, confirming region-specific air quality patterns.
            </p>
            <div className="bg-white rounded-xl p-6 border border-gray-200 shadow-sm">
              <div className="space-y-3">
                {[
                  { name: 'black_density', desc: '% Black × population density', pct: 92 },
                  { name: 'minority_density', desc: '% Minority × population density', pct: 78 },
                  { name: 'Division_Pacific', desc: 'Pacific division (CA, OR, WA, HI, AK)', pct: 65 },
                  { name: 'pop_density_squared', desc: 'Population density² (non-linear)', pct: 52 },
                  { name: 'hispanic_density', desc: '% Hispanic × population density', pct: 45 },
                  { name: 'log_population_density', desc: 'Log-transformed density', pct: 38 },
                  { name: 'Division_Mountain', desc: 'Mountain division (CO, NM, AZ, etc.)', pct: 32 },
                ].map((feat, i) => (
                  <div key={i}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="font-mono text-slate-900 font-medium">{feat.name}</span>
                      <span className="text-gray-500">{feat.desc}</span>
                    </div>
                    <div className="w-full bg-gray-100 rounded-full h-2">
                      <div
                        className="bg-gradient-to-r from-accent-start to-accent-end h-2 rounded-full"
                        style={{ width: `${feat.pct}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* Key Takeaways */}
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-slate-900 mb-4">What We Learned</h2>
            <div className="space-y-4">
              <div className="bg-emerald-50 rounded-xl p-6 border border-emerald-200">
                <p className="text-gray-700">
                  <strong className="text-slate-900">Feature engineering mattered most.</strong> The jump from R² 0.235 to 0.534 came primarily from interaction features—especially race × density. XGBoost can learn interactions on its own, but explicitly providing them as features gave the model a massive boost on our relatively small dataset of 942 counties.
                </p>
              </div>
              <div className="bg-emerald-50 rounded-xl p-6 border border-emerald-200">
                <p className="text-gray-700">
                  <strong className="text-slate-900">Income alone is not predictive.</strong> Our baseline model with income only produced R² = −0.04. Air quality is not simply a function of wealth—where and who matters more than how much.
                </p>
              </div>
              <div className="bg-emerald-50 rounded-xl p-6 border border-emerald-200">
                <p className="text-gray-700">
                  <strong className="text-slate-900">Race × density is the strongest signal.</strong> The top non-geographic features are <span className="text-accent-start font-semibold">black_density</span> and <span className="text-accent-start font-semibold">minority_density</span>. Counties with dense minority populations systematically experience worse air quality—a clear environmental justice finding.
                </p>
              </div>
              <div className="bg-emerald-50 rounded-xl p-6 border border-emerald-200">
                <p className="text-gray-700">
                  <strong className="text-slate-900">Regularization prevented overfitting.</strong> With 942 samples and 35+ features, overfitting was a real risk. L1/L2 regularization, subsampling, and conservative tree depth kept the model generalizable, confirmed by stable 5-fold CV scores.
                </p>
              </div>
            </div>
          </section>

          <div className="flex gap-4">
            <Link href="/data" className="px-6 py-3 bg-white text-slate-900 rounded-full font-medium border border-gray-300 hover:border-gray-400 hover:bg-gray-50 transition-colors">
              ← Data Cleaning
            </Link>
            <Link href="/findings" className="px-6 py-3 bg-accent-end text-white rounded-full font-medium hover:bg-accent-start transition-colors">
              View Findings →
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}
