# XGBoost Income + Race + Population Density + Region Model: Summary

## Overview

This model predicts **median AQI** from **median household income**, **racial composition** (% Black, % Hispanic), **population density**, and **Region** (Northeast, Midwest, South, West). Division is ignored.

- **Target:** median_aqi
- **Features:** Median_Household_Income, % Black or African American alone, % Hispanic or Latino, population_density, Region (one-hot: Midwest, Northeast, South, West with drop_first)
- **Random 80/20 split**
- **Sample weights** to downweight low AQI monitoring coverage
- **5-fold CV** for hyperparameter tuning

---

## Methodology

| Component | Details |
|-----------|---------|
| **Target** | `median_aqi` |
| **Features** | Median_Household_Income, % Black or African American alone, % Hispanic or Latino, population_density, Region (one-hot: Northeast, South, West; Midwest reference) |
| **Train/test** | Random 80/20 |
| **Sample weights** | `sample_weight` |
| **CV** | 5-fold KFold |
| **Model** | XGBoost Regressor |

---

## Results

### Best Hyperparameters

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 100 |
| `max_depth` | 3 |
| `learning_rate` | 0.05 |
| `min_child_weight` | 3 |

### Test Set Performance

| Metric | Value |
|--------|-------|
| **RMSE** | 9.37 |
| **MAE** | 6.72 |
| **R²** | 0.235 |

### Feature Importance

| Feature | Importance |
|---------|------------|
| Region_West | 0.23 |
| population_density | 0.20 |
| % Black or African American alone | 0.17 |
| % Hispanic or Latino | 0.15 |
| Region_South | 0.10 |
| Median_Household_Income | 0.09 |
| Region_Northeast | 0.05 |

---

## Interpretation

- **Region adds predictive signal** — R² improves from 0.212 (income + race + density only) to 0.235 with Region included.
- **Region_West is the top feature** — The West has distinct AQI patterns (wildfire, geography, pollution sources) that the model captures separately from density.
- **Region_South and Region_Northeast** also contribute, though less than West.

---

## Files

- **Notebook:** `xgboost_income_race_populationDensity_region.ipynb`
- **Data:** `../JOINED-aqi-income-race-populationDensity-region/joined-data-with-region.csv`
