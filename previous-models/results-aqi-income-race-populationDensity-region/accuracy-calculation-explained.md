# Accuracy Calculation for the AQI Regression Model

This document explains how to interpret and compute "accuracy" for our XGBoost regression model, which predicts **median AQI** (a continuous value) from income, race, population density, and region.

---

## Why "Accuracy" Is Different for Regression

**Classification** models (e.g., spam vs not spam) use **accuracy** = % of correct predictions.

**Regression** models predict continuous numbers, so there is no single "correct" or "incorrect" prediction. We use other metrics instead.

---

## Standard Regression Metrics

| Metric | Value | Meaning |
|--------|-------|---------|
| **RMSE** | 9.37 | Root Mean Squared Error — typical prediction error is ~9.4 AQI points |
| **MAE** | 6.72 | Mean Absolute Error — on average, predictions are off by ~6.7 AQI points |
| **R²** | 0.235 | Coefficient of determination — 23.5% of variance in AQI is explained by the model |

---

## Translating to "Percent Accuracy"

### 1. R² as Variance Explained

**R² ≈ 23.5%** can be interpreted as:

> *"23.5% of the variation in county-level AQI is explained by our features (income, race, density, region)."*

This is the closest standard metric to "percent accuracy" for regression. It does *not* mean "23.5% of predictions are correct" — it means 23.5% of the target's variability is accounted for.

---

### 2. Mean-Based Approximation

If we treat accuracy as *"how much of the typical value do we get right on average?"*:

- Mean AQI across counties ≈ 42
- MAE = 6.72
- Relative error ≈ MAE / mean ≈ 6.72 / 42 ≈ **16%**
- Approximate "accuracy" ≈ 1 − 0.16 ≈ **84%**

Interpretation: *"On average, predictions are off by about 16% of the typical AQI value."* The inverse (84%) is a rough "percent correct" feel, but this is **not** a standard metric.

---

### 3. Within-Tolerance Accuracy (Optional)

We can define accuracy as: *"What percent of predictions fall within ±X AQI points of the true value?"*

Example thresholds:

| Tolerance | Interpretation |
|-----------|----------------|
| ±5 AQI points | Predictions within 5 of true value |
| ±10 AQI points | Predictions within 10 of true value |
| ±15 AQI points | Predictions within 15 of true value |

This would need to be computed from the model's predictions on the test set. With MAE ≈ 6.72, roughly half of predictions are within ~7 points; a ±10 threshold would capture a majority.

---

## Summary

| Interpretation | Value | Use Case |
|----------------|-------|----------|
| **Variance explained** | **23.5%** (R²) | Standard, most defensible "percent" for regression |
| **Mean-based approximation** | **~84%** (1 − MAE/mean) | Informal intuition only |
| **Within ±10 AQI points** | (compute from predictions) | Custom threshold-based accuracy |

For presentations or reports, **R² = 23.5%** is the most appropriate "percent accuracy"–style metric for this regression model.
