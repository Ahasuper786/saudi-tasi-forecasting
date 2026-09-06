# Recruiter-Facing Portfolio Copy

## GitHub bio

**MSc Data Analytics | Python • Machine Learning • Time Series • Financial Data**

Alternative, slightly more specific:

**MSc Data Analytics @ Queen Mary University of London | Python • ML • Time-Series Forecasting**

## LinkedIn project title

**Saudi TASI Return Forecasting with Release-Aware Economic Data**

## LinkedIn project description

Built a release-aware time-series forecasting pipeline to test whether Saudi macro-financial, commodity and trade indicators improve one-month-ahead TASI return forecasts beyond market history alone.

Compared Zero-Return and Historical-Mean benchmarks, AutoReg(12), Random Forest and XGBoost using a chronological 2024–2026 holdout and 76 expanding-window walk-forward forecasts. XGBoost achieved the lowest fixed-test MAE (3.049), while Random Forest achieved the lowest fixed-test RMSE (4.044) and 63.33% directional accuracy. In the longer walk-forward evaluation, simple benchmarks remained more robust, showing that the recent machine-learning advantage was period-specific rather than stable over time.

**Skills:** Python, pandas, scikit-learn, XGBoost, statsmodels, feature engineering, leakage-aware time-series validation, walk-forward evaluation, classification, model interpretation.

Repository: https://github.com/Ahasuper786/saudi-tasi-forecasting

## CV version — concise

**Saudi TASI Return Forecasting | Python, scikit-learn, XGBoost, statsmodels**

- Built a release-aware time-series forecasting pipeline using 16 market, commodity, macroeconomic and Saudi trade predictors.
- Compared AutoReg, Random Forest and XGBoost against simple benchmarks using chronological holdout and 76-step expanding-window walk-forward validation.
- Achieved fixed-test MAE of 3.049 with XGBoost and 63.33% directional accuracy with Random Forest, while identifying that ML gains were not robust over the longer walk-forward period.

## CV version — one bullet

Developed a leakage-aware TASI return-forecasting pipeline in Python using AutoReg, Random Forest and XGBoost; evaluated models with chronological and 76-step walk-forward validation and showed that recent ML gains did not remain robust over time.

## 30-second interview explanation

I built a forecasting project around the Saudi TASI index. The main question was whether macroeconomic, commodity and trade information added predictive value beyond historical market returns. I designed the features around approximate publication availability so future information would not leak into the models, then compared simple benchmarks, AutoReg, Random Forest and XGBoost. The tree models looked slightly better on a recent fixed test period, but that advantage disappeared under longer walk-forward validation. The main lesson was that realistic validation and strong baselines matter more than simply choosing a more complex model.

## 60-second interview explanation

The project forecasts one-month-ahead Saudi TASI returns using a release-aware monthly dataset. I created four nested feature groups: historical TASI behaviour, commodities, macro and monetary variables, and Saudi trade indicators. I used chronological rather than random splitting, because a random split would leak future market structure into training. I compared Zero-Return and Historical-Mean baselines with AutoReg(12), Random Forest and XGBoost, and I also evaluated directional classification.

In the fixed 2024–2026 test, XGBoost had the lowest MAE at 3.049 and Random Forest had the lowest RMSE at 4.044 with 63.33% directional accuracy. But across 76 expanding-window forecasts from 2020–2026, the simple benchmarks performed better numerically. So I did not present the ML models as universally superior. The project demonstrates feature engineering, leakage prevention, benchmark discipline, time-series validation and cautious model interpretation.

## Suggested GitHub topics

`python` `machine-learning` `time-series` `forecasting` `xgboost` `random-forest` `financial-data` `saudi-arabia` `tasi` `walk-forward-validation`
