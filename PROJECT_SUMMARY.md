# Project summary

## Saudi TASI Return Forecasting with Release-Aware Economic Data

**Objective:** Test whether Saudi macro-financial, commodity and trade indicators improve one-month-ahead TASI return forecasts beyond historical market information.

**Stack:** Python, pandas, NumPy, scikit-learn, statsmodels, XGBoost, Matplotlib, Jupyter.

**Methods:** lag and rolling-volatility features, release-aware alignment, chronological holdout, AutoReg(12), Random Forest, XGBoost, directional classification, nested feature groups and expanding-window walk-forward validation.

**Data:** 136 complete monthly modelling observations from March 2015 to June 2026; 16 predictors; 106/30 fixed train-test split; 76 sequential walk-forward forecasts.

**Headline results:**

- XGBoost: best fixed-test MAE, **3.049**.
- Random Forest: best fixed-test RMSE, **4.044**, and directional accuracy, **63.33%**.
- Full Economic feature set: best fixed-test regression specification for both tree models.
- Walk-forward: Historical Mean achieved the lowest MAE, **3.879**, and Zero Return the lowest RMSE, **4.751**.

**Key insight:** A favourable recent holdout did not translate into stable performance over the longer sequential test. The project therefore demonstrates why financial forecasting claims should be benchmarked and stress-tested across time rather than based on a single test period.

**Recruiter value:** Demonstrates data preparation, leakage-aware modelling, time-series validation, machine-learning comparison, metric interpretation, reproducibility and transparent communication of limitations.
