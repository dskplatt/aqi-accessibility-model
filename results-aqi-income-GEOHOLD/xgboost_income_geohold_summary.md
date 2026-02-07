# XGBoost Income Model: Geographic Holdout Summary

## Overview

This model predicts **median AQI** from **median household income** using a **geographic holdout** for train/test split. Train on Midwest/East Coast; test on West (CA, OR, WA, NV, AZ, HI, AK) to validate whether the income→AQI relationship generalizes to new regions.

---

## Methodology

| Component | Details |
|-----------|---------|
| **Target** | `median_aqi` |
| **Feature** | `Median_Household_Income` |
| **Train** | All states except West (811 counties) |
| **Test** | West only: CA, OR, WA, NV, AZ, HI, AK (129 counties) |
| **Sample weights** | `sample_weight` — downweights low AQI monitoring coverage |
| **CV** | 5-fold KFold on training set for hyperparameter tuning |
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

### Test Set Performance (West Holdout)

| Metric | Geographic Holdout | Stratified (original) |
|--------|--------------------|------------------------|
| **RMSE** | 16.40 | 10.42 |
| **MAE** | 13.10 | 6.93 |
| **R²** | -0.076 | -0.0375 |

---

## Interpretation

- **Worse performance on West holdout** — RMSE and MAE are substantially higher than the stratified split; R² is more negative.
- **Income does not generalize to West** — The income→AQI relationship learned from Midwest/East does not transfer well to Western states (different geography, wildfire patterns, pollution sources).
- **Baseline finding** — Income alone has limited predictive power, and that limitation is exacerbated when testing on unseen regions.

---

## Files

- **Notebook:** `xgboost_income_geohold.ipynb`
- **Data:** `../JOINED-aqi-income/aqi-income-joined.csv`
