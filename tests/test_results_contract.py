from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def test_static_headline_metrics():
    static = pd.read_csv(RESULTS / "Table_4_2_Static_Regression.csv", index_col=0)
    assert np.isclose(static.loc["Random Forest", "RMSE"], 4.044166, atol=5e-4)
    assert np.isclose(static.loc["Random Forest", "Directional_Accuracy"], 0.633333, atol=5e-4)
    assert np.isclose(static.loc["XGBoost", "MAE"], 3.048709, atol=5e-4)


def test_walk_forward_headline_metrics():
    wf = pd.read_csv(RESULTS / "Table_4_7_Walk_Forward_Performance.csv", index_col=0)
    assert np.isclose(wf.loc["Historical Mean", "MAE"], 3.879051, atol=5e-4)
    assert np.isclose(wf.loc["Zero Return", "RMSE"], 4.750962, atol=5e-4)
    assert np.isclose(wf.loc["Random Forest", "RMSE"], 4.950911, atol=5e-4)
    assert np.isclose(wf.loc["XGBoost", "RMSE"], 5.287203, atol=5e-4)
