# Model evaluation notes

## Why two evaluation designs?

A single holdout period can favour a model because of the market regime represented in that window. This project therefore reports both a recent fixed holdout and a longer expanding-window walk-forward evaluation.

## Fixed test

- Train: Mar 2015–Dec 2023 (106 observations)
- Test: Jan 2024–Jun 2026 (30 observations)

This design answers: **How did each model perform on one recent, completely later period?**

## Walk-forward test

- Initial training window: 60 observations
- Forecasts: 76
- Forecast period: Mar 2020–Jun 2026

At forecast step *i*, the model is trained only on observations available before *i*. After the actual return is observed, it is added to the training set for the next forecast.

This design answers: **Was the model’s performance stable across a longer sequence of changing market conditions?**

## Metrics

### Regression

- **MAE:** average absolute forecast error; lower is better.
- **RMSE:** penalises larger errors more strongly; lower is better.
- **R²:** improvement relative to a mean-based reference; can be negative out of sample.
- **Directional accuracy:** fraction of forecasts with the correct positive/non-positive sign.

### Classification

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

## Interpretation rule

No single metric is treated as decisive. The project compares error magnitude, sign prediction and temporal robustness before drawing conclusions.
