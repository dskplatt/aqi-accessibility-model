import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load the best model to get feature importances
# Note: We need to make sure we use the feature names from the training set
model = joblib.load('../best_xgboost_model.pkl')

# Feature names as defined in the training script
feature_names = [
    'sample_weight', '% Hispanic or Latino', '% White alone', '% Black or African American alone',
    '% American Indian and Alaska Native alone', '% Asian alone', '% Two or More Races',
    'Median_Household_Income', 'Total_Population', 'Land_Area_SqMi',
    'population_density', 'log_population_density', 'log_median_income', 'total_minority_pct',
    'income_per_capita', 'urban_income', 'minority_density',
    'pop_density_squared', 'log_density_squared',
    'white_to_minority_ratio', 'income_to_density_ratio',
    'hispanic_density', 'black_density', 'asian_density',
    'minority_income', 'white_income',
    'Region_Midwest', 'Region_Northeast', 'Region_South', 'Region_West',
    'Division_East North Central', 'Division_East South Central', 'Division_Middle Atlantic',
    'Division_Mountain', 'Division_New England', 'Division_Pacific',
    'Division_South Atlantic', 'Division_West North Central', 'Division_West South Central'
]

# Create importance dataframe
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
})

# Filter out 'sample_weight' and get top 5
top_5 = importance_df[importance_df['feature'] != 'sample_weight'].sort_values('importance', ascending=False).head(5)

# Human-readable mapping for features
feature_map = {
    'black_density': 'Black Population Density',
    'minority_density': 'Minority Population Density',
    'income_per_capita': 'Income Per Capita',
    'Region_West': 'Western Region',
    'Division_Pacific': 'Pacific Division',
    'Land_Area_SqMi': 'Land Area',
    'income_to_density_ratio': 'Income-to-Density Ratio',
    'white_to_minority_ratio': 'White-to-Minority Ratio',
    'population_density': 'Population Density',
    'total_minority_pct': 'Total Minority %'
}

top_5['feature_display'] = top_5['feature'].map(lambda x: feature_map.get(x, x))

# Sort for horizontal bar chart (highest at top)
top_5 = top_5.sort_values('importance', ascending=True)

# Plotting
plt.style.use('dark_background')
fig = plt.figure(figsize=(10, 6), facecolor='#030712') # Tailwind gray-950
ax = fig.add_subplot(111)
ax.set_facecolor('#030712')

# Colors - using Electric Blue
colors = ['#3b82f6'] * len(top_5)

bars = plt.barh(top_5['feature_display'], top_5['importance'], color=colors, alpha=0.8,
                edgecolor='#60a5fa', linewidth=1)

# Add value labels
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.002, bar.get_y() + bar.get_height()/2.,
             f'{width:.3f}', ha='left', va='center', color='white', fontweight='bold')

plt.xlabel('Relative Importance Score', color='#9ca3af', fontsize=12)
plt.title('Top 5 Predictive Features for AQI', color='white', fontsize=14, pad=20)

# Customize grid and spines
plt.grid(True, axis='x', alpha=0.1, linestyle=':')
for spine in ax.spines.values():
    spine.set_color('#374151') # gray-700

plt.tight_layout()
plt.savefig('feature_importance_dark.png', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
print("\nSaved plot to 'feature_importance_dark.png'")
