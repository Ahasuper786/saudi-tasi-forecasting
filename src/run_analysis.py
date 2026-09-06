# Forecasting Saudi TASI Returns under Vision 2030
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from statsmodels.tsa.ar_model import AutoReg
from xgboost import XGBRegressor, XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "final" / "Saudi_TASI_Release_Aware_Model_Data.csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["Date"])

FEATURE_A = [
    "TASI_Lag_1", "TASI_Lag_3", "TASI_Lag_6", "TASI_Lag_12",
    "TASI_Rolling_3M", "TASI_Rolling_6M"
]
FEATURE_B = FEATURE_A + ["Brent_Available", "Gold_Available"]
FEATURE_C = FEATURE_B + [
    "Inflation_Available", "Policy_Rate_Available", "Repo_Rate_Available"
]
FEATURE_D = FEATURE_C + [
    "Trade_Balance_Available", "Total_Exports_Available",
    "Non_Oil_Exports_Available", "Oil_Exports_Available", "Imports_Available"
]
TARGET = "Target_Next_Month_Return"

period = df[
    (df["Date"] >= "2015-01-31") &
    (df["Date"] <= "2026-06-30") &
    df[TARGET].notna()
].copy()

model = period.dropna(subset=FEATURE_D + [TARGET]).reset_index(drop=True)

assert len(model) == 136
assert model["Date"].min() == pd.Timestamp("2015-03-31")
assert model["Date"].max() == pd.Timestamp("2026-06-30")

train = model[model["Date"] <= "2023-12-31"].copy()
test = model[
    (model["Date"] >= "2024-01-31") &
    (model["Date"] <= "2026-06-30")
].copy()

assert len(train) == 106
assert len(test) == 30

XGB_REG_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    objective="reg:squarederror", tree_method="hist",
    random_state=42, n_jobs=-1
)
XGB_CLF_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, eval_metric="logloss",
    tree_method="hist"
)

def reg_metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": mean_squared_error(y, pred) ** 0.5,
        "R2": r2_score(y, pred),
        "Directional_Accuracy": accuracy_score(
            (y > 0).astype(int), (pred > 0).astype(int)
        ),
    }

def cls_metrics(y, pred, prob=None):
    out = {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC_AUC": np.nan,
    }
    if prob is not None and len(np.unique(y)) > 1:
        out["ROC_AUC"] = roc_auc_score(y, prob)
    return out

# Static regression
ytr = train[TARGET]
yte = test[TARGET]

static = {}
static["Zero Return"] = reg_metrics(yte, np.zeros(len(yte)))
static["Historical Mean"] = reg_metrics(yte, np.full(len(yte), ytr.mean()))

ar = AutoReg(ytr.to_numpy(), lags=12, trend="c", old_names=False).fit()
pred_ar = np.asarray(
    ar.predict(start=len(ytr), end=len(ytr) + len(yte) - 1, dynamic=False)
)
static["AutoReg(12)"] = reg_metrics(yte, pred_ar)

rf = RandomForestRegressor(
    n_estimators=500, max_depth=6, min_samples_leaf=3,
    random_state=42, n_jobs=-1
)
rf.fit(train[FEATURE_D], ytr)
pred_rf = rf.predict(test[FEATURE_D])
static["Random Forest"] = reg_metrics(yte, pred_rf)

xgb = XGBRegressor(**XGB_REG_PARAMS)
xgb.fit(train[FEATURE_D], ytr)
pred_xgb = xgb.predict(test[FEATURE_D])
static["XGBoost"] = reg_metrics(yte, pred_xgb)

static_df = pd.DataFrame(static).T
static_df.to_csv(OUT / "Table_4_2_Static_Regression.csv")

static_predictions = pd.DataFrame({
    "Date": test["Date"],
    "Actual": yte,
    "AutoReg_12": pred_ar,
    "Random_Forest": pred_rf,
    "XGBoost": pred_xgb,
})
static_predictions.to_csv(OUT / "Static_Test_Predictions.csv", index=False)

# Static directional classification
yc_tr = (train[TARGET] > 0).astype(int)
yc_te = (test[TARGET] > 0).astype(int)
classification = {}
majority_class = int(yc_tr.mode().iloc[0])
majority_pred = np.full(len(yc_te), majority_class, dtype=int)
classification["Majority Baseline"] = cls_metrics(yc_te, majority_pred, None)

logit = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(random_state=42, max_iter=1000))
])
logit.fit(train[FEATURE_D], yc_tr)
classification["Logistic Regression"] = cls_metrics(
    yc_te,
    logit.predict(test[FEATURE_D]),
    logit.predict_proba(test[FEATURE_D])[:, 1]
)

rfc = RandomForestClassifier(
    n_estimators=500, max_depth=6, min_samples_leaf=3,
    random_state=42, n_jobs=-1
)
rfc.fit(train[FEATURE_D], yc_tr)
classification["Random Forest"] = cls_metrics(
    yc_te,
    rfc.predict(test[FEATURE_D]),
    rfc.predict_proba(test[FEATURE_D])[:, 1]
)

xgc = XGBClassifier(**XGB_CLF_PARAMS)
xgc.fit(train[FEATURE_D], yc_tr)
classification["XGBoost"] = cls_metrics(
    yc_te,
    xgc.predict(test[FEATURE_D]),
    xgc.predict_proba(test[FEATURE_D])[:, 1]
)

class_df = pd.DataFrame(classification).T
class_df.to_csv(OUT / "Table_4_4_Static_Classification.csv")

GROUPS = {
    "A: Market Only": FEATURE_A,
    "B: Market + Commodities": FEATURE_B,
    "C: Market + Commodities + Macro": FEATURE_C,
    "D: Full Economic": FEATURE_D,
}

group_class_rows = []
for group_name, features in GROUPS.items():
    clf = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(random_state=42, max_iter=1000))
    ])
    clf.fit(train[features], yc_tr)
    pred = clf.predict(test[features])
    prob = clf.predict_proba(test[features])[:, 1]
    group_class_rows.append({
        "Feature_Group": group_name,
        **cls_metrics(yc_te, pred, prob)
    })

pd.DataFrame(group_class_rows).to_csv(
    OUT / "Table_4_5_Feature_Group_Classification.csv", index=False
)

group_reg_rows = []
for group_name, features in GROUPS.items():
    rf_g = RandomForestRegressor(
        n_estimators=500, max_depth=6, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    )
    rf_g.fit(train[features], ytr)
    group_reg_rows.append({
        "Model": "Random Forest",
        "Feature_Group": group_name,
        **reg_metrics(yte, rf_g.predict(test[features]))
    })

    xg_g = XGBRegressor(**XGB_REG_PARAMS)
    xg_g.fit(train[features], ytr)
    group_reg_rows.append({
        "Model": "XGBoost",
        "Feature_Group": group_name,
        **reg_metrics(yte, xg_g.predict(test[features]))
    })

pd.DataFrame(group_reg_rows).to_csv(
    OUT / "Table_4_6_Feature_Group_Regression.csv", index=False
)

importance_df = pd.DataFrame({
    "Feature": FEATURE_D,
    "Importance": xgb.feature_importances_,
}).sort_values("Importance", ascending=False).reset_index(drop=True)
importance_df.index = importance_df.index + 1
importance_df.index.name = "Rank"
importance_df.to_csv(OUT / "Table_4_3_XGBoost_Feature_Importance.csv")

# Expanding-window walk-forward validation
wf = model.reset_index(drop=True)
wf_rows = []
INITIAL_WINDOW = 60

for i in range(INITIAL_WINDOW, len(wf)):
    tr = wf.iloc[:i]
    te = wf.iloc[[i]]
    ytr_i = tr[TARGET].to_numpy()
    actual = float(te[TARGET].iloc[0])

    rf_w = RandomForestRegressor(
        n_estimators=500, max_depth=6, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    )
    rf_w.fit(tr[FEATURE_D], ytr_i)

    xg_w = XGBRegressor(**XGB_REG_PARAMS)
    xg_w.fit(tr[FEATURE_D], ytr_i)

    wf_rows.append({
        "Date": te["Date"].iloc[0],
        "Actual": actual,
        "Zero_Return": 0.0,
        "Historical_Mean": float(np.mean(ytr_i)),
        "Random_Forest": float(rf_w.predict(te[FEATURE_D])[0]),
        "XGBoost": float(xg_w.predict(te[FEATURE_D])[0]),
    })

wf_df = pd.DataFrame(wf_rows)
assert len(wf_df) == 76
assert wf_df["Date"].min() == pd.Timestamp("2020-03-31")
assert wf_df["Date"].max() == pd.Timestamp("2026-06-30")

wf_metrics = {
    "Zero Return": reg_metrics(wf_df["Actual"], wf_df["Zero_Return"]),
    "Historical Mean": reg_metrics(wf_df["Actual"], wf_df["Historical_Mean"]),
    "Random Forest": reg_metrics(wf_df["Actual"], wf_df["Random_Forest"]),
    "XGBoost": reg_metrics(wf_df["Actual"], wf_df["XGBoost"]),
}

wf_metrics_df = pd.DataFrame(wf_metrics).T
wf_metrics_df.to_csv(OUT / "Table_4_7_Walk_Forward_Performance.csv")
wf_df.to_csv(OUT / "Walk_Forward_Predictions.csv", index=False)

rf_cm = confusion_matrix(
    (wf_df["Actual"] > 0).astype(int),
    (wf_df["Random_Forest"] > 0).astype(int)
)
pd.DataFrame(
    rf_cm,
    index=["Actual_Non_Positive", "Actual_Positive"],
    columns=["Pred_Non_Positive", "Pred_Positive"]
).to_csv(OUT / "Walk_Forward_RF_Confusion_Matrix.csv")

checks = {
    "static_rf_mae": (static_df.loc["Random Forest", "MAE"], 3.111876),
    "static_rf_rmse": (static_df.loc["Random Forest", "RMSE"], 4.044166),
    "static_rf_r2": (static_df.loc["Random Forest", "R2"], 0.020616),
    "static_xgb_mae": (static_df.loc["XGBoost", "MAE"], 3.048709),
    "walk_rf_mae": (wf_metrics_df.loc["Random Forest", "MAE"], 4.021521),
    "walk_rf_rmse": (wf_metrics_df.loc["Random Forest", "RMSE"], 4.950911),
    "walk_xgb_mae": (wf_metrics_df.loc["XGBoost", "MAE"], 4.212516),
    "walk_xgb_rmse": (wf_metrics_df.loc["XGBoost", "RMSE"], 5.287203),
}
for name, (actual, expected) in checks.items():
    if not np.isclose(actual, expected, atol=5e-4):
        raise AssertionError(f"{name}: got {actual}, expected approximately {expected}")

print("Reproduction checks passed.")
print(f"Final modelling observations: {len(model)}")
print(f"Static train/test: {len(train)}/{len(test)}")
print(f"Walk-forward forecasts: {len(wf_df)}")
