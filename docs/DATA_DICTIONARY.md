# Data dictionary

This file documents the fields used by the final forecasting specification. Column names refer to `data/final/Saudi_TASI_Release_Aware_Model_Data.csv`.

| Field | Meaning | Role / availability treatment |
|---|---|---|
| `Date` | Month-end observation date | Time index |
| `Target_Next_Month_Return` | TASI return for the following month | Regression target |
| `Target_Direction` | 1 if next-month return is positive, otherwise 0 | Classification target |
| `TASI_Lag_1` | TASI monthly return lagged 1 month | Market-history predictor |
| `TASI_Lag_3` | TASI monthly return lagged 3 months | Market-history predictor |
| `TASI_Lag_6` | TASI monthly return lagged 6 months | Market-history predictor |
| `TASI_Lag_12` | TASI monthly return lagged 12 months | Market-history predictor |
| `TASI_Rolling_3M` | Sample SD of recent 3-month returns | Market-volatility predictor |
| `TASI_Rolling_6M` | Sample SD of recent 6-month returns | Market-volatility predictor |
| `Brent_Available` | Brent commodity series | Shifted 1 month |
| `Gold_Available` | Gold commodity series | Shifted 1 month |
| `Inflation_Available` | Saudi annual inflation (YoY) | Shifted 1 month |
| `Policy_Rate_Available` | Policy Rate | Current/effective at month end |
| `Repo_Rate_Available` | Repo Rate | Current/effective at month end |
| `Trade_Balance_Available` | Saudi trade balance | Shifted 2 months |
| `Total_Exports_Available` | Saudi total exports | Shifted 2 months |
| `Non_Oil_Exports_Available` | Saudi non-oil exports | Shifted 2 months |
| `Oil_Exports_Available` | Saudi oil exports | Shifted 2 months |
| `Imports_Available` | Saudi imports | Shifted 2 months |

## Verified inflation corrections

- May 2024: 1.6%
- May 2025: 2.2%
- June 2025: 2.3%

## Provenance note

The archived raw TASI file does not retain metadata identifying its original provider. The repository intentionally records this limitation rather than assigning an unverified source.
