# XGBoost Income + Race + Population Density Model: Summary

## Overview

This model predicts **median AQI** from **median household income**, **racial composition** (% Black, % Hispanic), and **population density**. Does adding density improve prediction?

- **Target:** median_aqi
- **Full model features:** Median_Household_Income, % Black or African American alone, % Hispanic or Latino, population_density
- **Density-only model:** population_density only
- **Random 80/20 split**
- **Sample weights** to downweight low AQI monitoring coverage
- **5-fold CV** for hyperparameter tuning

---

## Methodology

| Component | Details |
|-----------|---------|
| **Target** | `median_aqi` |
| **Full model features** | Median_Household_Income, % Black or African American alone, % Hispanic or Latino, population_density |
| **Density-only features** | population_density |
| **Train/test** | Random 80/20 |
| **Sample weights** | `sample_weight` |
| **CV** | 5-fold KFold |
| **Model** | XGBoost Regressor |

---

## Results

### Best Hyperparameters (both models)

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 100 |
| `max_depth` | 3 |
| `learning_rate` | 0.05 |
| `min_child_weight` | 3 |

### Test Set Performance

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| **Full (income + race + density)** | 9.51 | 6.77 | **0.212** |
| Density-only | 9.88 | 6.98 | 0.149 |

### Comparison to Income + Race (no density)

| Model | R² |
|-------|-----|
| Full (income + race + density) | **0.212** |
| Income + race (previous) | 0.105 |
| Density-only | 0.149 |

### Feature Importance (Full model)

| Feature | Importance |
|---------|------------|
| % Black or African American alone | 0.32 |
| population_density | 0.29 |
| % Hispanic or Latino | 0.22 |
| Median_Household_Income | 0.17 |

---

## Interpretation

- **Adding population density improves prediction** — R² increases from 0.105 (income + race) to 0.212 (full model). Population density carries meaningful signal for AQI.
- **Population density is second in importance** — second only to % Black; urban vs rural geography matters for air quality patterns.
- **Density-only model (R² = 0.15)** — Population density alone predicts AQI better than income + race (0.10). Urban concentration may capture pollution sources (traffic, industry) and monitoring density.
- **Full model best** — Combining income, race, and density yields the strongest predictive power.

---

## Files

- **Notebook:** `xgboost_income_race_populationDensity.ipynb`
- **Data:** `../JOINED-aqi-income-race-populationDensity/aqi-income-race-populationDensity-joined.csv`
