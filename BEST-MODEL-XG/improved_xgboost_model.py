import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('BEST-DATASET/joined-data-with-features.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nTarget variable distribution:")
print(df['median_aqi'].describe())

# Advanced Feature Engineering
def create_features(df):
    df = df.copy()

    # Interaction features
    df['income_per_capita'] = df['Median_Household_Income'] / (df['Total_Population'] + 1)
    df['urban_income'] = df['population_density'] * df['Median_Household_Income'] / 1000000
    df['minority_density'] = df['total_minority_pct'] * df['population_density'] / 100

    # Polynomial features
    df['pop_density_squared'] = df['population_density'] ** 2
    df['log_density_squared'] = df['log_population_density'] ** 2

    # Ratio features
    df['white_to_minority_ratio'] = df['% White alone'] / (df['total_minority_pct'] + 0.1)
    df['income_to_density_ratio'] = df['Median_Household_Income'] / (df['population_density'] + 1)

    # Specific demographic interactions
    df['hispanic_density'] = df['% Hispanic or Latino'] * df['population_density'] / 100
    df['black_density'] = df['% Black or African American alone'] * df['population_density'] / 100
    df['asian_density'] = df['% Asian alone'] * df['population_density'] / 100

    # Income-demographic interactions
    df['minority_income'] = df['total_minority_pct'] * df['Median_Household_Income'] / 100000
    df['white_income'] = df['% White alone'] * df['Median_Household_Income'] / 100000

    return df

df = create_features(df)

# Prepare features
feature_columns = [
    # Original features
    'sample_weight',
    '% Hispanic or Latino', '% White alone', '% Black or African American alone',
    '% American Indian and Alaska Native alone', '% Asian alone', '% Two or More Races',
    'Median_Household_Income', 'Total_Population', 'Land_Area_SqMi',
    'population_density',

    # Previously engineered features
    'log_population_density', 'log_median_income', 'total_minority_pct',

    # New interaction features
    'income_per_capita', 'urban_income', 'minority_density',
    'pop_density_squared', 'log_density_squared',
    'white_to_minority_ratio', 'income_to_density_ratio',
    'hispanic_density', 'black_density', 'asian_density',
    'minority_income', 'white_income'
]

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=['Region', 'Division'], drop_first=False)

# Get all feature columns (including one-hot encoded)
all_feature_cols = feature_columns + [col for col in df_encoded.columns if col.startswith('Region_') or col.startswith('Division_')]

X = df_encoded[all_feature_cols]
y = df_encoded['median_aqi']

print(f"\nNumber of features: {X.shape[1]}")
print(f"Features: {X.columns.tolist()}")

# Handle missing values
X = X.fillna(X.median())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (important for some features)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*50)
print("BASELINE MODEL")
print("="*50)

# Baseline model
baseline_model = XGBRegressor(random_state=42, n_jobs=-1)
baseline_model.fit(X_train_scaled, y_train)
y_pred_baseline = baseline_model.predict(X_test_scaled)

r2_baseline = r2_score(y_test, y_pred_baseline)
rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
mae_baseline = mean_absolute_error(y_test, y_pred_baseline)

print(f"Baseline R²: {r2_baseline:.4f}")
print(f"Baseline RMSE: {rmse_baseline:.4f}")
print(f"Baseline MAE: {mae_baseline:.4f}")

print("\n" + "="*50)
print("OPTIMIZED MODEL WITH HYPERPARAMETER TUNING")
print("="*50)

# Optimized hyperparameters for small datasets
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.5, 1],
    'reg_lambda': [1, 2, 3]
}

# Use fewer parameters for faster tuning
reduced_param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [200, 300],
    'min_child_weight': [3, 5],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'gamma': [0, 0.1],
    'reg_alpha': [0.5, 1],
    'reg_lambda': [1, 2]
}

xgb_model = XGBRegressor(random_state=42, n_jobs=-1)

# GridSearchCV with cross-validation
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=reduced_param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

print("Starting hyperparameter tuning (this may take a few minutes)...")
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV R²: {grid_search.best_score_:.4f}")

# Evaluate on test set
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test_scaled)

r2_tuned = r2_score(y_test, y_pred_tuned)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)

print(f"\nTuned Model R²: {r2_tuned:.4f}")
print(f"Tuned Model RMSE: {rmse_tuned:.4f}")
print(f"Tuned Model MAE: {mae_tuned:.4f}")

print("\n" + "="*50)
print("IMPROVEMENT")
print("="*50)
print(f"R² improvement: {r2_tuned - r2_baseline:.4f} ({((r2_tuned - r2_baseline) / abs(r2_baseline) * 100):.1f}% increase)")

# Feature importance
print("\n" + "="*50)
print("TOP 20 MOST IMPORTANT FEATURES")
print("="*50)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20))

# Cross-validation scores
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"\nCross-validation R² scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_tuned, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.title(f'Predictions vs Actual (R² = {r2_tuned:.4f})')
plt.tight_layout()
plt.savefig('predictions_vs_actual.png', dpi=300)
print("\nSaved plot to 'predictions_vs_actual.png'")

# Save the model
import joblib
joblib.dump(best_model, 'best_xgboost_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nSaved model to 'best_xgboost_model.pkl' and scaler to 'scaler.pkl'")
