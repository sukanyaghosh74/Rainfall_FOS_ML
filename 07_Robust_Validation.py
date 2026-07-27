"""
07_Robust_Validation.py
========================
Phase 1: Robust Validation & Statistical Significance
- 5-Fold Cross-Validation
- Repeated K-Fold Cross-Validation (5x3)
- Grouped Validation (Leave-One-MatrixType-Out, Leave-One-VBP-Out)
- Friedman + Nemenyi post-hoc tests
- Box plots of CV fold scores

All models use IDENTICAL preprocessing pipeline as the baseline scripts.
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
import matplotlib.ticker as ticker

from sklearn.model_selection import (
    KFold, RepeatedKFold, LeaveOneGroupOut, cross_validate
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
import scipy.stats as stats
import scikit_posthocs as sp

from catboost import CatBoostRegressor
import lightgbm as lgb

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RESULT_FOLDER = "Results/Phase1_Validation"
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

# Label-encode categorical columns
le_encoders = {}
for col in X_raw.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X_raw[col] = le.fit_transform(X_raw[col].astype(str))
    le_encoders[col] = le

X_raw = X_raw.reset_index(drop=True)

# Groups for GroupedCV
matrix_type_groups = df["Matrix Type"].astype("category").cat.codes.values
vbp_groups = pd.cut(df["VBP (%)"], bins=3, labels=False)  # 3 bins

# ============================================================
# DEFINE MODELS (stateless factories - fresh instance each fold)
# ============================================================

def make_catboost():
    return CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6,
        loss_function='RMSE', random_seed=SEED, verbose=0
    )

def make_lgbm():
    return lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        random_state=SEED, n_jobs=-1, verbosity=-1
    )

# ============================================================
# CUSTOM SCORERS
# ============================================================

def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape_score(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

rmse_scorer = make_scorer(rmse_score, greater_is_better=False)
mape_scorer = make_scorer(mape_score, greater_is_better=False)
r2_scorer   = make_scorer(r2_score)
mae_scorer  = make_scorer(mean_absolute_error, greater_is_better=False)

SCORERS = {
    'r2': r2_scorer,
    'rmse': rmse_scorer,
    'mae': mae_scorer,
    'mape': mape_scorer
}

# ============================================================
# SCALED PIPELINE WRAPPER
# ============================================================

class ScaledModel(BaseEstimator, RegressorMixin):
    """Wraps a sklearn-compatible model with MinMaxScaler."""
    def __init__(self, model_fn):
        self.model_fn = model_fn

    def fit(self, X, y):
        self.scaler_ = MinMaxScaler()
        X_s = self.scaler_.fit_transform(X)
        self.model_ = self.model_fn()
        self.model_.fit(X_s, y)
        return self

    def predict(self, X):
        X_s = self.scaler_.transform(X)
        return self.model_.predict(X_s)

# ============================================================
# SKLEARN ANN WRAPPER (avoids TF session issues in CV)
# ============================================================

from sklearn.neural_network import MLPRegressor

class SKLearnANN(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    def fit(self, X, y):
        self.scaler_ = MinMaxScaler()
        X_s = self.scaler_.fit_transform(X)
        self.model_ = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=30,
            random_state=SEED
        )
        self.model_.fit(X_s, y)
        return self
    def predict(self, X):
        X_s = self.scaler_.transform(X)
        return self.model_.predict(X_s)

MODELS = {
    "ANN (MLP)":   SKLearnANN(),
    "CatBoost":    ScaledModel(make_catboost),
    "LightGBM":    ScaledModel(make_lgbm),
}

# ============================================================
# HELPER: Collect per-fold scores
# ============================================================

def run_cv(cv_strategy, label, groups=None):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    all_results = {}
    for name, model in MODELS.items():
        cv_res = cross_validate(
            model, X_raw.values, y.values,
            cv=cv_strategy,
            scoring=SCORERS,
            groups=groups,
            return_train_score=False,
            n_jobs=1
        )
        r2_scores   =  cv_res['test_r2']
        rmse_scores = -cv_res['test_rmse']
        mae_scores  = -cv_res['test_mae']
        mape_scores = -cv_res['test_mape']

        all_results[name] = {
            'R2':   r2_scores,
            'RMSE': rmse_scores,
            'MAE':  mae_scores,
            'MAPE': mape_scores,
        }
        print(f"  {name:<15} R2: {r2_scores.mean():.4f}±{r2_scores.std():.4f} | "
              f"RMSE: {rmse_scores.mean():.4f}±{rmse_scores.std():.4f} | "
              f"MAPE: {mape_scores.mean():.2f}%±{mape_scores.std():.2f}%")
    return all_results


# ============================================================
# STRATEGY 1: 5-Fold CV
# ============================================================
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
results_5fold = run_cv(kf, "5-Fold Cross-Validation")

# ============================================================
# STRATEGY 2: Repeated K-Fold (5x3)
# ============================================================
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
results_rkf = run_cv(rkf, "Repeated K-Fold CV (5x3)")

# ============================================================
# STRATEGY 3a: Leave-One-MatrixType-Out
# ============================================================
logo = LeaveOneGroupOut()
results_logo_matrix = run_cv(logo, "Leave-One-MatrixType-Out", groups=matrix_type_groups)

# ============================================================
# STRATEGY 3b: Leave-One-VBP-Out (binned)
# ============================================================
results_logo_vbp = run_cv(logo, "Leave-One-VBP-Bin-Out", groups=vbp_groups)

# ============================================================
# SUMMARY TABLE
# ============================================================

def make_summary_table(results, label):
    rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        for metric, scores in metrics.items():
            row[f"{metric} Mean"] = round(scores.mean(), 5)
            row[f"{metric} Std"]  = round(scores.std(),  5)
        rows.append(row)
    df_out = pd.DataFrame(rows)
    df_out.to_excel(os.path.join(RESULT_FOLDER, f"{label}.xlsx"), index=False)
    df_out.to_csv(os.path.join(RESULT_FOLDER, f"{label}.csv"), index=False)
    return df_out

t1 = make_summary_table(results_5fold,        "CV_5Fold_Summary")
t2 = make_summary_table(results_rkf,          "CV_Repeated_Summary")
t3 = make_summary_table(results_logo_matrix,  "CV_LOGO_Matrix_Summary")
t4 = make_summary_table(results_logo_vbp,     "CV_LOGO_VBP_Summary")

print("\n5-Fold CV Summary:")
print(t1.to_string(index=False))

# ============================================================
# STATISTICAL SIGNIFICANCE — FRIEDMAN + NEMENYI
# ============================================================

print("\n--- Friedman Test (5-Fold R2 scores) ---")
r2_matrix = np.column_stack([results_5fold[m]['R2'] for m in MODELS])
stat, p_val = stats.friedmanchisquare(*[r2_matrix[:, i] for i in range(r2_matrix.shape[1])])
print(f"  Friedman chi2 = {stat:.4f}, p-value = {p_val:.4f}")

with open(os.path.join(RESULT_FOLDER, "Statistical_Tests.txt"), "w", encoding="utf-8") as f:
    f.write("FRIEDMAN TEST ON 5-FOLD R2 SCORES\n")
    f.write(f"  Models: {list(MODELS.keys())}\n")
    f.write(f"  Friedman chi2 = {stat:.6f}\n")
    f.write(f"  p-value       = {p_val:.6f}\n\n")

    if p_val < 0.05:
        f.write("  Result: SIGNIFICANT differences exist (p < 0.05)\n")
        f.write("  Proceeding with Nemenyi post-hoc test...\n\n")
        print("  Significant differences found. Running Nemenyi post-hoc...")
        r2_df = pd.DataFrame(r2_matrix, columns=list(MODELS.keys()))
        nemenyi_res = sp.posthoc_nemenyi_friedman(r2_df)
        print(nemenyi_res)
        f.write("NEMENYI POST-HOC PAIRWISE p-VALUES:\n")
        f.write(nemenyi_res.to_string())
        nemenyi_res.to_excel(os.path.join(RESULT_FOLDER, "Nemenyi_PostHoc.xlsx"))
    else:
        f.write("  Result: No significant differences found (p >= 0.05)\n")

# ============================================================
# BOX PLOTS — R2 per Fold
# ============================================================

metrics_to_plot = ['R2', 'RMSE', 'MAPE']
cv_sources = [
    ("5-Fold CV",       results_5fold),
    ("Repeated KFold",  results_rkf),
]

for cv_label, cv_res in cv_sources:
    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        data   = [cv_res[m][metric] for m in MODELS]
        labels = list(MODELS.keys())

        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color='black', linewidth=2))

        colors = ['#4C72B0', '#55A868', '#C44E52']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{metric} Distribution — {cv_label}", fontsize=13, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()

        safe_cv = cv_label.replace(" ", "_")
        plt.savefig(os.path.join(RESULT_FOLDER, f"BoxPlot_{safe_cv}_{metric}.png"), dpi=300)
        plt.close()

print(f"\nAll Phase 1 results saved in: {RESULT_FOLDER}")
