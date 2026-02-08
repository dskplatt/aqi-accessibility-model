# XGBoost Income Model: Summary of Findings

## Overview

This document summarizes the results of an XGBoost model that predicts **median AQI** (a proxy for air quality and livability) from **median household income** at the county level. The analysis is part of a larger investigation into how socioeconomic factors correlate with access to a livable climate.

---

## Research Question

**Can income alone predict air quality (and thus livability) at the county level?**

---

## Methodology

| Component | Details |
|-----------|---------|
| **Target variable** | `median_aqi` — median Air Quality Index (higher = worse air quality) |
| **Feature** | `Median_Household_Income` — county-level median household income |
| **Sample weights** | `sample_weight` — downweights counties with low AQI monitoring coverage |
| **Train/test split** | 80/20, stratified by AQI quartiles |
| **Cross-validation** | 5-fold stratified CV for hyperparameter tuning |
| **Model** | XGBoost Regressor |

### Stratification

Data was stratified by AQI quartiles so that train and test sets have similar distributions of livability levels, ensuring fair evaluation across the full range of air quality.

---

## Data

- **Source:** `aqi_income_joined.csv` — joined AQI and Census median household income data
- **Sample size:** 940 counties (after dropping missing values)
- **AQI range:** 3 – 90
- **Train size:** 752 counties
- **Test size:** 188 counties

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
| **RMSE** | 10.42 |
| **MAE** | 6.93 |
| **R²** | -0.0375 |

### Feature Importance

| Feature | Importance |
|---------|------------|
| Median_Household_Income | 1.0 |

---

## Interpretation

### Model Performance

- **Negative R²** (-0.04) indicates that the model performs slightly worse than predicting the mean AQI for all counties. Income alone has weak predictive power for air quality.
- **RMSE of 10.42** — predictions are off by about 10 AQI points on average, which is substantial given the 3–90 range.

### Why Income Alone Fails to Predict AQI

1. **AQI is driven by geography** — pollution sources, weather, terrain, and industrial activity matter more than county-level income.
2. **Wealth and pollution can coexist** — high-income urban areas (e.g., LA, NYC) can have poor air quality; low-income rural areas can have good air quality.
3. **Income is a weak proxy** for where people live and what environmental exposures they face.

### Direction of the Relationship

Raw correlation between income and AQI in the data would clarify the *direction* of the relationship. A **negative correlation** (higher income → lower AQI) would support the narrative that wealthier counties tend to have better air quality. Even with weak predictive power, the direction matters for the environmental justice story.

---

## Conclusions

1. **Income alone cannot meaningfully predict AQI** — the model has no real predictive power (negative R²).
2. **This is expected** — air quality is largely determined by location and pollution sources, not directly by income.
3. **Next steps** — Add additional features (race, geography, urbanization, land use) to build a combined model that better captures how socioeconomic factors intersect with access to livable climate.
4. **Narrative value** — The income-only model serves as a baseline. Showing that single factors (income, race) have limited power alone, but together they can better explain environmental disparities, strengthens the hackathon story about structural barriers to a livable planet.

---

## Files

- **Notebook:** `xgboost_income_model.ipynb`
- **Data:** `aqi_income_joined.csv`
