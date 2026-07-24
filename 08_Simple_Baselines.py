"""
08_Simple_Baselines.py
========================
Phase 5: Simple Baseline Models (Linear Regression, Polynomial Regression)
- Shows why advanced ML is necessary
- Same preprocessing pipeline as all other scripts
- 5-Fold CV for fair comparison
"""

import os
import glob
import random
import warnings
import time

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import (
    make_scorer, r2_score, mean_squared_error,
    mean_absolute_error, explained_variance_score
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RESULT_FOLDER = "Results/Phase5_Baselines"
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ============================================================
# LOAD & CLEAN DATA (identical pipeline)
# ============================================================
excel_files = glob.glob("*.xlsx")
FILE_NAME = excel_files[0]
print(f"Reading: {FILE_NAME}")

df = pd.read_excel(FILE_NAME)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.dropna(axis=1, how="all").dropna()
print(f"Dataset Shape: {df.shape}")

X_raw = df.drop(columns=["FOS"]).copy()
y = df["FOS"].copy().reset_index(drop=True)

feature_names = list(X_raw.columns)

for col in X_raw.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X_raw[col] = le.fit_transform(X_raw[col].astype(str))

X_raw = X_raw.reset_index(drop=True)

# ============================================================
# SCORERS
# ============================================================

def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape_score(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

SCORERS = {
    'r2':   make_scorer(r2_score),
    'rmse': make_scorer(rmse_score, greater_is_better=False),
    'mae':  make_scorer(mean_absolute_error, greater_is_better=False),
    'mape': make_scorer(mape_score, greater_is_better=False),
}

# ============================================================
# MODELS
# ============================================================

MODELS = {
    "Linear Regression": Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', LinearRegression())
    ]),
    "Polynomial (deg=2)": Pipeline([
        ('scaler', MinMaxScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('model', LinearRegression())
    ]),
    "Polynomial (deg=3)": Pipeline([
        ('scaler', MinMaxScaler()),
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('model', LinearRegression())
    ]),
}

# ============================================================
# 5-FOLD CV
# ============================================================

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
all_results = {}

print("\n5-Fold CV Results:")
print("="*70)

for name, model in MODELS.items():
    t0 = time.time()
    cv_res = cross_validate(
        model, X_raw.values, y.values,
        cv=kf, scoring=SCORERS,
        return_train_score=False, n_jobs=1
    )
    elapsed = time.time() - t0

    r2   =  cv_res['test_r2']
    rmse = -cv_res['test_rmse']
    mae  = -cv_res['test_mae']
    mape = -cv_res['test_mape']

    all_results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

    print(f"\n  {name}")
    print(f"    R2  : {r2.mean():.4f} ± {r2.std():.4f}")
    print(f"    RMSE: {rmse.mean():.4f} ± {rmse.std():.4f}")
    print(f"    MAE : {mae.mean():.4f} ± {mae.std():.4f}")
    print(f"    MAPE: {mape.mean():.2f}% ± {mape.std():.2f}%")
    print(f"    CV Time: {elapsed:.2f}s")

# ============================================================
# SINGLE TRAIN/TEST SPLIT METRICS (for direct comparison)
# ============================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_raw.values, y.values, test_size=0.20, random_state=SEED
)

print("\n\nHold-Out Test Set Metrics (80/20 split):")
print("="*70)

holdout_rows = []
for name, model in MODELS.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    preds = model.predict(X_test)
    pred_time = time.time() - t0

    R2   = r2_score(y_test, preds)
    n, p = len(y_test), X_test.shape[1]
    Adj_R2 = 1 - ((1 - R2) * (n - 1) / (n - p - 1)) if n > p+1 else np.nan
    RMSE = np.sqrt(mean_squared_error(y_test, preds))
    MAE  = mean_absolute_error(y_test, preds)
    MAPE = np.mean(np.abs((y_test - preds) / y_test)) * 100
    ExpVar = explained_variance_score(y_test, preds)

    holdout_rows.append({
        "Model": name, "R2": round(R2, 5), "Adj_R2": round(Adj_R2, 5),
        "RMSE": round(RMSE, 5), "MAE": round(MAE, 5),
        "MAPE": round(MAPE, 5), "Explained_Var": round(ExpVar, 5),
        "Train_Time_s": round(train_time, 4), "Pred_Time_s": round(pred_time, 6)
    })
    print(f"\n  {name}")
    print(f"    R2={R2:.5f}  RMSE={RMSE:.5f}  MAE={MAE:.5f}  MAPE={MAPE:.2f}%")

    # Save metrics.txt for comparison aggregation later
    model_folder = os.path.join(RESULT_FOLDER, name.replace(" ", "_").replace("(","").replace(")","").replace("=",""))
    os.makedirs(model_folder, exist_ok=True)
    with open(os.path.join(model_folder, "metrics.txt"), "w") as f:
        f.write(f"R2               : {R2}\n")
        f.write(f"Adjusted R2      : {Adj_R2}\n")
        f.write(f"RMSE             : {RMSE}\n")
        f.write(f"MAE              : {MAE}\n")
        f.write(f"MAPE             : {MAPE}\n")
        f.write(f"Explained Var    : {ExpVar}\n")
        f.write(f"Training Time (s): {train_time}\n")
        f.write(f"Prediction Time (s): {pred_time}\n")

    # Save predictions
    pd.DataFrame({"Actual": y_test, "Predicted": preds}).to_excel(
        os.path.join(model_folder, "Predictions.xlsx"), index=False
    )

    # Actual vs Predicted plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_test, preds, alpha=0.6, color='steelblue', s=20)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel("Actual FOS"); axes[0].set_ylabel("Predicted FOS")
    axes[0].set_title(f"Actual vs Predicted\n{name}"); axes[0].grid(True, alpha=0.4)
    ann_text = f"R²={R2:.4f}\nRMSE={RMSE:.4f}\nMAE={MAE:.4f}"
    axes[0].annotate(ann_text, xy=(0.05, 0.92), xycoords='axes fraction',
                     fontsize=10, va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    residuals = y_test - preds
    axes[1].scatter(preds, residuals, alpha=0.5, color='seagreen', s=20)
    axes[1].axhline(0, color='red', linestyle='--', lw=2)
    axes[1].set_xlabel("Predicted FOS"); axes[1].set_ylabel("Residuals")
    axes[1].set_title(f"Residual Plot\n{name}"); axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, "Actual_vs_Predicted.png"), dpi=300)
    plt.close()

holdout_df = pd.DataFrame(holdout_rows)
holdout_df.to_excel(os.path.join(RESULT_FOLDER, "Baselines_Holdout_Metrics.xlsx"), index=False)
holdout_df.to_csv(os.path.join(RESULT_FOLDER, "Baselines_Holdout_Metrics.csv"), index=False)
print(f"\n\nAll Phase 5 results saved in: {RESULT_FOLDER}")

# ============================================================
# CV SUMMARY TABLE
# ============================================================

cv_rows = []
for name, metrics in all_results.items():
    row = {"Model": name}
    for metric, scores in metrics.items():
        row[f"{metric}_Mean"] = round(scores.mean(), 5)
        row[f"{metric}_Std"]  = round(scores.std(), 5)
    cv_rows.append(row)

cv_df = pd.DataFrame(cv_rows)
cv_df.to_excel(os.path.join(RESULT_FOLDER, "Baselines_CV_Summary.xlsx"), index=False)
print("\n5-Fold CV Summary:")
print(cv_df.to_string(index=False))
