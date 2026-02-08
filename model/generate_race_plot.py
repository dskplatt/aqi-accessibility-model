import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('main-dataset/main-dataset.csv')

# Define all race columns from the user's snippet
race_cols = ['% Hispanic or Latino', '% White alone', '% Black or African American alone',
             '% American Indian and Alaska Native alone', '% Asian alone', '% Two or More Races']

# Drop rows with missing values in race columns or median_aqi
df_race = df.dropna(subset=race_cols + ['median_aqi']).copy()

# Determine predominant race across ALL categories
df_race['predominant_race'] = df_race[race_cols].idxmax(axis=1).str.replace('% ', '').str.replace(' alone', '')

# Filter to ONLY include Hispanic/Latino, Black/African American, and White as requested previously
allowed_races = ['Hispanic or Latino', 'Black or African American', 'White']
df_race = df_race[df_race['predominant_race'].isin(allowed_races)]

# Calculate average median AQI
aqi_by_race = df_race.groupby('predominant_race')['median_aqi'].mean().sort_values(ascending=False)

# Plotting
plt.style.use('dark_background')
fig = plt.figure(figsize=(10, 6), facecolor='#030712') # Tailwind gray-950
ax = fig.add_subplot(111)
ax.set_facecolor('#030712')

# Colors for the bars - using Electric Blue
colors = ['#3b82f6'] * len(aqi_by_race) # Blue-500

# To make the difference look more drastic, we'll set the y-axis bottom limit
# while keeping the actual data values the same.
min_val = aqi_by_race.min()
plt.ylim(min_val * 0.9, aqi_by_race.max() * 1.05)

bars = plt.bar(aqi_by_race.index, aqi_by_race.values, color=colors, alpha=0.8, width=0.6, 
               edgecolor='#60a5fa', linewidth=1) # Lighter blue edge

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}', ha='center', va='bottom', color='white', fontweight='bold')

plt.xlabel('Predominant Race in County', color='#9ca3af', fontsize=12)
plt.ylabel('Average Median AQI', color='#9ca3af', fontsize=12)
plt.title('Average Median AQI by Predominant Race', color='white', fontsize=14, pad=20)

# Customize grid and spines
plt.grid(True, axis='y', alpha=0.1, linestyle=':')
for spine in ax.spines.values():
    spine.set_color('#374151') # gray-700

plt.tight_layout()
plt.savefig('aqi_by_race_dark.png', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
print("\nSaved plot to 'aqi_by_race_dark.png'")
