# AQI Accessibility Model

An environmental justice analysis exploring how race, income, population density, and geography relate to county-level air quality across the United States. 

## Model Performance

| Metric | Value |
|--------|-------|
| Within 12 AQI points | **91.01%** |
| Within 10 AQI points | **85.71%** |
| R² (test set) | **0.534** |

The final XGBoost model explains 53.4% of the variance in median AQI, improved from an initial R² of −0.04 through iterative feature engineering, RobustScaler preprocessing, and hyperparameter tuning across 192 combinations with 5-fold cross-validation.

## Project Structure

```
├── website/                  # Next.js website (dashboard)
├── model/                    # Final model & improvement notebooks
├── previous-models/          # Iterative XGBoost experiments
├── previous-datasets/        # Data cleaning notebooks
└── main-dataset/             # Final joined dataset
```

## Website

**https://aqi-accessibility-model.vercel.app/**

Built with Next.js 16, React 19, Tailwind CSS 4, and Plotly.js.

## Key Findings

- **Income alone does not predict air quality** (R² = −0.04)
- **Race × population density** is the strongest predictor — counties with dense minority populations experience systematically worse air quality
- Top features: `black_density`, `minority_density`, Pacific/Mountain divisions
- Feature engineering (11 new interaction/polynomial features) was the largest driver of model improvement

## Data Sources

- **EPA** — Air Quality Index by county (2024–2025)
- **U.S. Census Bureau (ACS 5-Year Estimates)** — Median household income (S1903), race (DP05), population (B01003), land area (GEOINFO)
- **U.S. Census Bureau** — Region and division classifications

## Contributors

- **William Pantel**
- **Ian Stansberry**
- **Dominic Platt**
