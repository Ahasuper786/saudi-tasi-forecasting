# Verified model outputs

The CSVs in this directory are the verified outputs used by the portfolio case study. Running `python src/run_analysis.py` regenerates the main tables from the final release-aware modelling data.

## Headline values

| Evaluation | Model | Metric | Value |
|---|---|---|---:|
| Fixed | Random Forest | RMSE | 4.044166 |
| Fixed | Random Forest | Directional accuracy | 0.633333 |
| Fixed | XGBoost | MAE | 3.048709 |
| Walk-forward | Historical Mean | MAE | 3.879051 |
| Walk-forward | Zero Return | RMSE | 4.750962 |
| Walk-forward | Random Forest | RMSE | 4.950911 |
| Walk-forward | XGBoost | RMSE | 5.287203 |
