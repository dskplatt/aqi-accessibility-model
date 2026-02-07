# XGBoost Income + Race Model: Summary

## Overview

This model predicts **median AQI** from **median household income** and **racial composition** (% Black, % Latino). Environmental justice lens: do counties with higher concentrations of Black or Latino residents face worse air quality?

- **Target:** median_aqi
- **Features:** Median_Household_Income, % Black or African American alone, % Hispanic or Latino
- **Random 80/20 split**
- **Sample weights** to downweight low AQI monitoring coverage
- **5-fold CV** for hyperparameter tuning

---

## Methodology

| Component | Details |
|-----------|---------|
| **Target** | `median_aqi` |
| **Features** | `Median_Household_Income`, `% Black or African American alone`, `% Hispanic or Latino` |
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

| Metric | Income + Race | Income only (stratified) |
|--------|---------------|--------------------------|
| **RMSE** | 9.78 | 10.42 |
| **MAE** | 6.78 | 6.93 |
| **R²** | **0.105** | -0.04 |

### Feature Importance

| Feature | Importance |
|---------|------------|
| % Black or African American alone | 0.45 |
| % Hispanic or Latino | 0.31 |
| Median_Household_Income | 0.23 |

---

## Interpretation (Environmental Justice)

- **Adding race improves prediction** — R² goes from negative (income-only) to positive 0.10. Racial composition captures information that income alone does not.
- **% Black is the strongest predictor** — highest feature importance. Counties with higher % Black tend to have different AQI patterns, consistent with environmental justice literature on disproportionate exposure.
- **% Latino is second** — also contributes meaningfully to prediction.
- **Income matters less** — third in importance. Geography and demographic composition may interact with pollution sources in ways that income alone misses.

---

## Files

- **Notebook:** `xgboost_income_race_random.ipynb`
- **Data:** `../JOINED-aqi-income-race/aqi-income-race-joined.csv`
