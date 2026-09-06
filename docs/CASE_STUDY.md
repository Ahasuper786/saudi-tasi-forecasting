# Case study: forecasting Saudi TASI returns

## 1. The question

Financial return forecasting is difficult because the target is noisy and relationships can change over time. This project asks a specific question:

> Does adding commodity, macro-financial and Saudi trade information improve one-month-ahead TASI return forecasts beyond historical market information alone?

Saudi Vision 2030 provides the economic context for including diversification-related indicators, but the project does **not** estimate a causal effect of Vision 2030 on the stock market.

## 2. The modelling challenge

A standard random train/test split would leak future market regimes into training. Economic data also have publication delays: a statistic labelled with month *t* may not have been available by the end of month *t*.

The pipeline therefore uses:

- chronological train/test separation;
- approximate release-aware feature alignment;
- simple forecasting baselines;
- nested feature groups;
- expanding-window walk-forward validation.

## 3. Feature design

The final specification contains 16 predictors:

- TASI return lags: 1, 3, 6 and 12 months;
- rolling 3-month and 6-month return standard deviations;
- Brent and Gold, lagged one month;
- Inflation YoY, lagged one month;
- Policy Rate and Repo Rate at month end;
- Trade Balance, Total Exports, Non-Oil Exports, Oil Exports and Imports, lagged two months.

The resulting complete modelling sample contains 136 monthly observations from March 2015 to June 2026.

## 4. Evaluation design

### Fixed chronological holdout

- Training: 106 observations, Mar 2015–Dec 2023.
- Test: 30 observations, Jan 2024–Jun 2026.

Models:

- Zero Return;
- Historical Mean;
- AutoReg(12);
- Random Forest;
- XGBoost.

### Walk-forward validation

An initial 60-observation window is expanded one observation at a time. Each model produces a one-step-ahead forecast before the next actual observation is added to training. This produces 76 sequential forecasts from March 2020 to June 2026.

## 5. Results

In the fixed test, XGBoost achieved the lowest MAE (3.049). Random Forest achieved the lowest RMSE (4.044), a small positive R² (0.021) and the strongest regression directional accuracy (63.33%).

The Full Economic feature group produced the best fixed-test regression results for both tree models, suggesting that broader economic information added value during that particular period.

However, the walk-forward evaluation reversed the headline conclusion. Historical Mean achieved the lowest MAE (3.879), while Zero Return achieved the lowest RMSE (4.751). Random Forest and XGBoost produced larger numerical errors and negative R² values.

## 6. Interpretation

The important result is not that machine learning “failed” or “won.” It is that **model rankings depended on the evaluation period**.

A recent fixed holdout suggested modest machine-learning gains. A longer sequential test showed that those gains were not stable. This is a realistic result for financial forecasting and illustrates why benchmark choice and temporal validation matter as much as model complexity.

## 7. What I would improve next

With a larger research horizon, I would investigate:

- global equity-market variables;
- market-news and sentiment signals;
- higher-frequency observations;
- formal hyperparameter selection using only training-era validation;
- regime-aware forecasting;
- probabilistic forecasts and prediction intervals.

Any extension would retain chronological and walk-forward evaluation to avoid overstating performance.
