# XGBoost AQI Prediction Model — Results Summary

## Overview

This model predicts **median AQI** (Air Quality Index) at the county level using demographic, economic, and geographic features. An XGBoost regressor is trained with hyperparameter tuning via 5-fold cross-validation.

---

## Dataset

| Property | Value |
|----------|-------|
| **Source** | `BEST-DATASET/joined-data-with-features.csv` |
| **Rows** | 942 counties |
| **Target** | `median_aqi` |

### Target Distribution

| Statistic | Value |
|-----------|-------|
| Mean | 38.65 |
| Std Dev | 10.49 |
| Range | 3 – 90 |

---

## Features

**39 features** in total:

- **Demographics:** % Hispanic, % White, % Black, % American Indian, % Asian, % Two or More Races, total_minority_pct
- **Economic:** Median_Household_Income, income_per_capita, urban_income, minority_income, white_income
- **Population:** Total_Population, Land_Area_SqMi, population_density, log_population_density
- **Interactions:** minority_density, black_density, hispanic_density, asian_density, income_to_density_ratio, white_to_minority_ratio, pop_density_squared, log_density_squared
- **Geography:** Region (4), Division (9) — one-hot encoded
- **Quality:** sample_weight (monitoring coverage)

---

## Model Performance

### Baseline vs Tuned

| Metric | Baseline | Tuned |
|--------|----------|-------|
| **R²** | 0.4532 | **0.5341** |
| **RMSE** | 7.92 | **7.31** |
| **MAE** | 5.66 | **5.31** |

Hyperparameter tuning yields a **17.9% R² improvement** over the default model.

### Best Hyperparameters

| Parameter | Value |
|-----------|-------|
| max_depth | 3 |
| learning_rate | 0.05 |
| n_estimators | 200 |
| min_child_weight | 3 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| gamma | 0.1 |
| reg_alpha | 1 |
| reg_lambda | 2 |

---

## Top 20 Feature Importances

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | sample_weight | 0.135 |
| 2 | black_density | 0.104 |
| 3 | minority_density | 0.054 |
| 4 | income_per_capita | 0.054 |
| 5 | Region_West | 0.054 |
| 6 | Division_Pacific | 0.054 |
| 7 | Land_Area_SqMi | 0.039 |
| 8 | income_to_density_ratio | 0.032 |
| 9 | white_to_minority_ratio | 0.029 |
| 10 | population_density | 0.028 |
| 11 | total_minority_pct | 0.026 |
| 12 | Division_Mountain | 0.024 |
| 13 | asian_density | 0.024 |
| 14 | Division_West South Central | 0.023 |
| 15 | % Hispanic or Latino | 0.022 |
| 16 | Total_Population | 0.021 |
| 17 | urban_income | 0.020 |
| 18 | white_income | 0.020 |
| 19 | % Black or African American alone | 0.020 |
| 20 | minority_income | 0.019 |

---

## Interpretation

**Geography is a strong predictor.** Region_West and Division_Pacific are among the top features, consistent with distinct AQI patterns (e.g., wildfire, climate, pollution sources) in the West.

**Demographic–density interactions matter.** black_density, minority_density, and asian_density capture how racial composition varies with population density and relate to AQI.

**Economic structure is relevant.** income_per_capita and income_to_density_ratio suggest that income and urban/rural structure influence air quality.

**Sample weight** reflects monitoring coverage and ranks highest, probably because counties with more monitoring tend to differ in ways that correlate with AQI.

---

## Cross-Validation

- **Method:** 5-fold CV on training set
- **Mean CV R²:** 0.4648 (± 0.1295)
- **Fold scores:** [0.38, 0.42, 0.53, 0.44, 0.55]

---

## Preprocessing

- **Split:** 80/20 train/test, random
- **Scaling:** RobustScaler (fit on train, transform on test)
- **Imputation:** Median (from training set)
- **Categoricals:** One-hot encoding for Region and Division

---

## Outputs

| File | Description |
|------|-------------|
| `predictions_vs_actual.png` | Predicted vs actual AQI scatter plot |
| `best_xgboost_model.pkl` | Trained XGBoost model |
| `scaler.pkl` | Fitted RobustScaler |
