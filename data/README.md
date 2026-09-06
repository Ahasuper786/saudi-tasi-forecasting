# Data availability and provenance

This public portfolio repository does **not** redistribute the archived raw market or economic source files.

The project was built from archived TASI market data plus World Bank commodity data, IMF/GASTAT trade and inflation information, and Saudi monetary-policy series. The archived TASI raw file no longer preserves verifiable source-provider metadata, so this repository does not guess or assign a provider that cannot be independently confirmed.

For that reason, the public repository focuses on the forecasting code, verified result tables, model-evaluation design and documentation. To rerun `src/run_analysis.py`, place the final release-aware modelling table at:

`data/final/Saudi_TASI_Release_Aware_Model_Data.csv`

The expected final specification contains 136 complete modelling observations from March 2015 to June 2026 and 16 predictors. Availability assumptions used in the analysis were:

- Brent and Gold: lag 1 month
- Inflation YoY: lag 1 month
- Policy Rate and Repo Rate: current/effective at month end
- Trade Balance, Total Exports, Non-Oil Exports, Oil Exports and Imports: lag 2 months

Verified inflation corrections applied in the project were May 2024 = 1.6%, May 2025 = 2.2%, and June 2025 = 2.3%.

The repository intentionally documents missing provenance instead of reconstructing it from memory.
