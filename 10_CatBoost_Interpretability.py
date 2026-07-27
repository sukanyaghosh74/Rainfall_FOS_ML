"""
10_CatBoost_Interpretability.py
=================================
Phase 2: Deep Model Interpretation for CatBoost (best model)
- SHAP Summary, Beeswarm, Bar, Dependence plots
- Permutation Feature Importance
- Partial Dependence Plots (PDP) + ICE plots
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
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import r2_score, mean_squared_error
import shap

from catboost import CatBoostRegressor

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RESULT_FOLDER = "Results/Phase2_Interpretation"
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ============================================================
# LOAD & PREPROCESS (identical pipeline)
# ============================================================

excel_files = glob.glob("*.xlsx")
FILE_NAME   = excel_files[0]
print(f"Reading: {FILE_NAME}")

df = pd.read_excel(FILE_NAME)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.dropna(axis=1, how="all").dropna()
print(f"Dataset Shape: {df.shape}")

X_raw = df.drop(columns=["FOS"]).copy()
y     = df["FOS"].copy().reset_index(drop=True)
feature_names = list(X_raw.columns)

# Store original Matrix Type labels for interpretation
if "Matrix Type" in X_raw.columns:
    matrix_labels = df["Matrix Type"].astype("category")
    matrix_codes  = matrix_labels.cat.codes

for col in X_raw.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X_raw[col] = le.fit_transform(X_raw[col].astype(str))

X_raw = X_raw.reset_index(drop=True)

scaler   = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y.values, test_size=0.20, random_state=SEED
)

# ============================================================
# TRAIN CATBOOST
# ============================================================

print("\nTraining CatBoost...")
model = CatBoostRegressor(
    iterations=1000, learning_rate=0.05, depth=6,
    loss_function='RMSE', random_seed=SEED,
    early_stopping_rounds=50, verbose=100
)
model.fit(X_train, y_train, eval_set=(X_test, y_test))

preds  = model.predict(X_test)
R2     = r2_score(y_test, preds)
RMSE   = np.sqrt(mean_squared_error(y_test, preds))
print(f"CatBoost -> R2={R2:.5f}, RMSE={RMSE:.5f}")

# ============================================================
# SHAP ANALYSIS
# ============================================================

print("\nCalculating SHAP values...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    sv = shap_values[0]
else:
    sv = shap_values

X_test_df = pd.DataFrame(X_test, columns=feature_names)

# --- SHAP Summary Plot ---
plt.figure(figsize=(9, 6))
shap.summary_plot(sv, X_test_df, feature_names=feature_names, show=False, max_display=len(feature_names))
plt.title("SHAP Summary Plot — CatBoost", fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_FOLDER, "SHAP_Summary.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ SHAP Summary Plot saved")

# --- SHAP Beeswarm Plot ---
shap_exp = shap.Explanation(
    values=sv,
    base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[0],
    data=X_test_df.values,
    feature_names=feature_names
)
plt.figure(figsize=(9, 6))
shap.plots.beeswarm(shap_exp, show=False, max_display=len(feature_names))
plt.title("SHAP Beeswarm Plot — CatBoost", fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_FOLDER, "SHAP_Beeswarm.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ SHAP Beeswarm Plot saved")

# --- SHAP Bar Plot (global mean |SHAP|) ---
plt.figure(figsize=(8, 5))
shap.plots.bar(shap_exp, show=False, max_display=len(feature_names))
plt.title("SHAP Feature Importance (Mean |SHAP|) — CatBoost", fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_FOLDER, "SHAP_Bar.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ SHAP Bar Plot saved")

# --- SHAP Dependence Plots for top features ---
# Identify top features by mean |SHAP|
mean_abs_shap = np.abs(sv).mean(axis=0)
top_indices   = np.argsort(mean_abs_shap)[::-1][:4]
top_features  = [feature_names[i] for i in top_indices]

for feat in top_features:
    feat_idx = feature_names.index(feat)
    plt.figure(figsize=(7, 5))
    shap.dependence_plot(
        feat_idx, sv, X_test_df,
        feature_names=feature_names,
        show=False, alpha=0.7
    )
    plt.title(f"SHAP Dependence — {feat}", fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    safe = feat.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    plt.savefig(os.path.join(RESULT_FOLDER, f"SHAP_Dependence_{safe}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ SHAP Dependence Plot saved: {feat}")

# ============================================================
# PERMUTATION FEATURE IMPORTANCE
# ============================================================

print("\nCalculating Permutation Feature Importance...")

# Wrap model for sklearn's permutation_importance (must implement fit)
class CatBoostWrapper:
    def fit(self, X, y):
        return self  # already trained externally
    def predict(self, X):
        return model.predict(X)

perm_res = permutation_importance(
    CatBoostWrapper(), X_test, y_test,
    n_repeats=30, random_state=SEED,
    scoring='r2'
)

perm_means = perm_res.importances_mean
perm_stds  = perm_res.importances_std
sorted_idx = np.argsort(perm_means)[::-1]

fig, ax = plt.subplots(figsize=(9, 6))
y_pos = np.arange(len(feature_names))
ax.barh(
    y_pos,
    perm_means[sorted_idx],
    xerr=perm_stds[sorted_idx],
    align='center',
    color='steelblue', alpha=0.8, capsize=4
)
ax.set_yticks(y_pos)
ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=11)
ax.invert_yaxis()
ax.set_xlabel("Mean Decrease in R² (±1 SD)", fontsize=11)
ax.set_title("Permutation Feature Importance — CatBoost", fontsize=13, fontweight='bold', pad=12)
ax.grid(axis='x', linestyle='--', alpha=0.5)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_FOLDER, "Permutation_Importance.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Permutation Feature Importance saved")

# Save to CSV
perm_df = pd.DataFrame({
    "Feature":    [feature_names[i] for i in sorted_idx],
    "Mean_Decrease_R2": perm_means[sorted_idx].round(5),
    "Std":        perm_stds[sorted_idx].round(5)
})
perm_df.to_csv(os.path.join(RESULT_FOLDER, "Permutation_Importance.csv"), index=False)

# ============================================================
# PDP + ICE PLOTS
# ============================================================
# PDP + ICE PLOTS — Manual Implementation (avoids sklearn wrapper issues)
# ============================================================

print("\nGenerating PDP + ICE Plots...")

def compute_pdp_ice(model, X, feat_idx, n_grid=50):
    """
    Manually compute PDP and ICE curves for a given feature.
    Returns: grid_vals, pdp_vals, ice_matrix (n_samples x n_grid)
    """
    feat_min = X[:, feat_idx].min()
    feat_max = X[:, feat_idx].max()
    grid_vals = np.linspace(feat_min, feat_max, n_grid)

    ice_matrix = np.zeros((X.shape[0], n_grid))
    for j, val in enumerate(grid_vals):
        X_mod = X.copy()
        X_mod[:, feat_idx] = val
        ice_matrix[:, j] = model.predict(X_mod)

    pdp_vals = ice_matrix.mean(axis=0)
    return grid_vals, pdp_vals, ice_matrix

for feat in top_features:
    feat_idx = feature_names.index(feat)
    grid_vals, pdp_vals, ice_matrix = compute_pdp_ice(model, X_test, feat_idx)

    fig, ax = plt.subplots(figsize=(8, 5))

    # ICE lines
    for i in range(ice_matrix.shape[0]):
        ax.plot(grid_vals, ice_matrix[i], color='steelblue', alpha=0.06, linewidth=0.8)

    # PDP line
    ax.plot(grid_vals, pdp_vals, color='red', linewidth=2.5, label='PDP (mean)', zorder=5)

    ax.set_xlabel(f"{feat} (normalised 0–1)", fontsize=11)
    ax.set_ylabel("Predicted FOS", fontsize=11)
    ax.set_title(f"PDP + ICE — {feat}", fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    safe = feat.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    plt.savefig(os.path.join(RESULT_FOLDER, f"PDP_ICE_{safe}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] PDP+ICE saved: {feat}")

print(f"\nAll Phase 2 results saved in: {RESULT_FOLDER}")
