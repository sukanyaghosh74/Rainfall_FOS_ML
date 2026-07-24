"""
11_Error_and_Physics_Analysis.py
=================================
Phase 3 & 4: Physical Consistency & Error Analysis
- Residual analysis by Matrix Type, VBP, and FOS range
- Critical region (FOS ≈ 1) analysis
- Physical consistency checks (trend analysis)
- Geotechnical interpretation report
"""

import os
import glob
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from catboost import CatBoostRegressor

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RESULT_FOLDER = "Results/Phase3_4_Physics_Errors"
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ============================================================
# LOAD & PREPROCESS (identical pipeline)
# ============================================================

excel_files = glob.glob("*.xlsx")
FILE_NAME   = excel_files[0]
print(f"Reading: {FILE_NAME}")

df_orig = pd.read_excel(FILE_NAME)
df = df_orig.loc[:, ~df_orig.columns.str.contains('^Unnamed')]
df = df.dropna(axis=1, how="all").dropna().reset_index(drop=True)
print(f"Dataset Shape: {df.shape}")

X_raw = df.drop(columns=["FOS"]).copy()
y     = df["FOS"].copy().reset_index(drop=True)
feature_names = list(X_raw.columns)

# Keep original string labels for grouping
matrix_type_orig = df["Matrix Type"].astype(str) if "Matrix Type" in df.columns else None
vbp_orig         = df["VBP (%)"].copy()           if "VBP (%)" in df.columns      else None

for col in X_raw.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X_raw[col] = le.fit_transform(X_raw[col].astype(str))

X_raw = X_raw.reset_index(drop=True)
scaler   = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled, y.values, np.arange(len(y)), test_size=0.20, random_state=SEED
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

preds     = model.predict(X_test)
residuals = y_test - preds

R2   = r2_score(y_test, preds)
RMSE = np.sqrt(mean_squared_error(y_test, preds))
print(f"CatBoost → R²={R2:.5f}, RMSE={RMSE:.5f}")

# Build a dataframe with test results including metadata
test_df = pd.DataFrame({
    "Actual":    y_test,
    "Predicted": preds,
    "Residual":  residuals,
    "Abs_Error": np.abs(residuals),
})
if matrix_type_orig is not None:
    test_df["Matrix Type"] = matrix_type_orig.iloc[idx_test].values
if vbp_orig is not None:
    test_df["VBP (%)"]     = vbp_orig.iloc[idx_test].values

# FOS range category
bins   = [0, 0.9, 1.0, 1.1, 1.5, 2.0, 999]
labels = ["< 0.9", "0.9–1.0", "1.0–1.1", "1.1–1.5", "1.5–2.0", "> 2.0"]
test_df["FOS_Range"] = pd.cut(test_df["Actual"], bins=bins, labels=labels)

print("\n--- Test Set Summary by Group ---")
print(test_df.describe())

# ============================================================
# PHASE 4: ERROR ANALYSIS
# ============================================================

# ---- 4a. Residuals by Matrix Type ----
if "Matrix Type" in test_df.columns:
    matrix_groups = test_df.groupby("Matrix Type")["Residual"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Box plot
    groups = [g for _, g in matrix_groups]
    names  = [name for name, _ in matrix_groups]
    bp = axes[0].boxplot(groups, patch_artist=True, notch=False,
                         medianprops=dict(color='black', linewidth=2))
    colors = sns.color_palette("husl", len(names))
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    axes[0].set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    axes[0].axhline(0, color='red', linestyle='--', lw=1.5)
    axes[0].set_ylabel("Residuals (Actual − Predicted)")
    axes[0].set_title("Residuals by Matrix Type")
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # MAE per group
    mae_per_group = test_df.groupby("Matrix Type")["Abs_Error"].mean()
    axes[1].bar(mae_per_group.index, mae_per_group.values,
                color=colors, alpha=0.8, edgecolor='white')
    axes[1].set_xlabel("Matrix Type"); axes[1].set_ylabel("Mean Absolute Error")
    axes[1].set_title("MAE by Matrix Type")
    for i, (name, val) in enumerate(mae_per_group.items()):
        axes[1].text(i, val + 0.001, f"{val:.4f}", ha='center', va='bottom', fontsize=9)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_FOLDER, "Error_by_MatrixType.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("\n  ✓ Error by Matrix Type saved")

    # Statistics table
    matrix_stats = test_df.groupby("Matrix Type").agg(
        MAE=("Abs_Error", "mean"),
        RMSE=("Residual", lambda x: np.sqrt((x**2).mean())),
        Mean_Residual=("Residual", "mean"),
        Std_Residual=("Residual", "std"),
        Count=("Residual", "count")
    ).round(5)
    matrix_stats.to_excel(os.path.join(RESULT_FOLDER, "Error_Stats_MatrixType.xlsx"))
    print(matrix_stats)

# ---- 4b. Residuals by VBP ----
if "VBP (%)" in test_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    vbp_groups = test_df.groupby("VBP (%)")["Residual"]
    groups = [g.values for _, g in vbp_groups]
    names  = [str(name) for name, _ in vbp_groups]
    colors = sns.color_palette("muted", len(names))

    bp = axes[0].boxplot(groups, patch_artist=True,
                         medianprops=dict(color='black', linewidth=2))
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    axes[0].set_xticklabels(names, fontsize=10)
    axes[0].axhline(0, color='red', linestyle='--', lw=1.5)
    axes[0].set_xlabel("VBP (%)"); axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals by VBP (%)")
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    mae_vbp = test_df.groupby("VBP (%)")["Abs_Error"].mean()
    axes[1].bar(mae_vbp.index.astype(str), mae_vbp.values,
                color=colors, alpha=0.8, edgecolor='white')
    axes[1].set_xlabel("VBP (%)"); axes[1].set_ylabel("MAE")
    axes[1].set_title("MAE by VBP (%)")
    for i, val in enumerate(mae_vbp.values):
        axes[1].text(i, val + 0.001, f"{val:.4f}", ha='center', va='bottom', fontsize=9)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_FOLDER, "Error_by_VBP.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Error by VBP saved")

    vbp_stats = test_df.groupby("VBP (%)").agg(
        MAE=("Abs_Error", "mean"),
        RMSE=("Residual", lambda x: np.sqrt((x**2).mean())),
        Count=("Residual", "count")
    ).round(5)
    vbp_stats.to_excel(os.path.join(RESULT_FOLDER, "Error_Stats_VBP.xlsx"))

# ---- 4c. Residuals by FOS Range (Critical Zone Analysis) ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fos_groups  = test_df.groupby("FOS_Range", observed=False)["Residual"]
group_data  = [g.values for _, g in fos_groups]
group_names = [str(n) for n, _ in fos_groups]

bp = axes[0].boxplot(group_data, patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
crit_colors = ['#d62728' if '0.9–1.0' in n or '1.0–1.1' in n else '#4878d0' for n in group_names]
for patch, col in zip(bp['boxes'], crit_colors):
    patch.set_facecolor(col); patch.set_alpha(0.75)
axes[0].set_xticklabels(group_names, rotation=30, ha='right', fontsize=9)
axes[0].axhline(0, color='red', linestyle='--', lw=1.5)
axes[0].set_ylabel("Residuals"); axes[0].set_title("Residuals by FOS Range")
axes[0].grid(axis='y', linestyle='--', alpha=0.5)

# Annotate critical zone
for i, name in enumerate(group_names):
    if '0.9–1.0' in name or '1.0–1.1' in name:
        axes[0].get_xticklabels()[i].set_color('red')
        axes[0].get_xticklabels()[i].set_fontweight('bold')

mae_fos = test_df.groupby("FOS_Range", observed=False)["Abs_Error"].mean()
count_fos = test_df.groupby("FOS_Range", observed=False)["Abs_Error"].count()
axes[1].bar(range(len(mae_fos)), mae_fos.values, color=crit_colors, alpha=0.8, edgecolor='white')
axes[1].set_xticks(range(len(mae_fos)))
axes[1].set_xticklabels(group_names, rotation=30, ha='right', fontsize=9)
axes[1].set_ylabel("MAE"); axes[1].set_title("MAE by FOS Range (Red = Critical Zone)")
for i, (val, cnt) in enumerate(zip(mae_fos.values, count_fos.values)):
    axes[1].text(i, val + 0.0005, f"{val:.4f}\n(n={cnt})", ha='center', va='bottom', fontsize=8)
axes[1].grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_FOLDER, "Error_by_FOS_Range.png"), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Error by FOS Range saved")

fos_range_stats = test_df.groupby("FOS_Range", observed=False).agg(
    Count=("Residual", "count"),
    MAE=("Abs_Error", "mean"),
    RMSE=("Residual", lambda x: np.sqrt((x**2).mean())),
    Mean_Residual=("Residual", "mean"),
    Std_Residual=("Residual", "std")
).round(5)
fos_range_stats.to_excel(os.path.join(RESULT_FOLDER, "Error_Stats_FOS_Range.xlsx"))
print("\nFOS Range Error Statistics:")
print(fos_range_stats)

# Critical zone (FOS ≈ 1) detail
critical = test_df[test_df["Actual"].between(0.9, 1.1)]
print(f"\n  Critical Zone (FOS 0.9–1.1): {len(critical)} samples")
if len(critical) > 0:
    print(f"    MAE  = {critical['Abs_Error'].mean():.5f}")
    print(f"    RMSE = {np.sqrt((critical['Residual']**2).mean()):.5f}")
    print(f"    Max Error = {critical['Abs_Error'].max():.5f}")

# ============================================================
# PHASE 3: PHYSICAL CONSISTENCY CHECKS
# ============================================================

print("\n\n--- PHASE 3: Physical Consistency Checks ---")

# We predict FOS over a sweep of each feature while holding others at median
median_vals = np.median(X_scaled, axis=0)
report_lines = ["PHYSICAL CONSISTENCY REPORT\n", "="*60 + "\n\n"]

for i, feat in enumerate(feature_names):
    feat_range = np.linspace(X_scaled[:, i].min(), X_scaled[:, i].max(), 100)
    X_sweep = np.tile(median_vals, (100, 1))
    X_sweep[:, i] = feat_range

    fos_sweep = model.predict(X_sweep)
    correlation, p_val = stats.spearmanr(feat_range, fos_sweep)

    direction = "POSITIVE ↑" if correlation > 0.1 else ("NEGATIVE ↓" if correlation < -0.1 else "NEUTRAL ~")
    report_lines.append(f"Feature: {feat}\n")
    report_lines.append(f"  Spearman rho = {correlation:.4f}  (p={p_val:.4f})\n")
    report_lines.append(f"  Trend direction: {direction}\n\n")

    # Individual trend plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(feat_range, fos_sweep, color='steelblue', lw=2.5)
    ax.set_xlabel(f"{feat} (normalised 0–1)", fontsize=11)
    ax.set_ylabel("Predicted FOS", fontsize=11)
    ax.set_title(f"Physical Consistency — {feat}", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.annotate(f"Spearman ρ = {correlation:.3f}", xy=(0.02, 0.92),
                xycoords='axes fraction', fontsize=10,
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    plt.tight_layout()
    safe = feat.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("%", "pct")
    plt.savefig(os.path.join(RESULT_FOLDER, f"Physical_Trend_{safe}.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Geotechnical interpretation notes
report_lines.append("\nGEOTECHNICAL INTERPRETATION\n")
report_lines.append("="*60 + "\n")
report_lines.append("""
1. RAINFALL INTENSITY & TIME:
   Expected: Increasing rainfall intensity and duration should
   DECREASE FOS (more pore water pressure, reduced shear strength).
   A negative Spearman correlation for these features is physically
   consistent with slope stability mechanics (Terzaghi, Bishop).

2. VBP (Vegetation Base Percentage):
   Expected: Higher VBP should INCREASE FOS — vegetation roots
   provide mechanical reinforcement (root cohesion) and promote
   drainage (reduced pore pressures). A positive correlation is
   physically meaningful.

3. MATRIX TYPE:
   Expected: The ordering of matrix types (e.g., sandy vs. clay-rich)
   should reflect their known shear strength and hydraulic conductivity
   properties. Sandy loam typically drains faster → higher FOS for
   equivalent rainfall; clay-rich matrices retain water → lower FOS.

4. DISPLACEMENT & PWP:
   Expected: Higher displacement and pore water pressure (PWP) are
   indicators of reduced slope stability, hence a negative correlation
   with FOS is physically consistent.

NOTE: Discrepancies from expected trends may indicate data-driven
nuances (e.g., interaction effects captured by the model) rather
than physical inconsistencies.
""")

report_path = os.path.join(RESULT_FOLDER, "Physical_Consistency_Report.txt")
with open(report_path, "w") as f:
    f.writelines(report_lines)

print(f"\n  Physical Consistency Report saved: {report_path}")
print(f"\nAll Phase 3 & 4 results saved in: {RESULT_FOLDER}")
