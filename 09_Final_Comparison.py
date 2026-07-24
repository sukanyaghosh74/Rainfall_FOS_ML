"""
09_Final_Comparison.py
========================
Phase 6: Final Comprehensive Comparison
- Aggregates hold-out metrics from ALL models (ANN, CatBoost, LGBM, TabNet, FTTransformer, LinReg, PolyReg)
- Generates publication-quality comparison table
- Generates grouped bar charts for R2, RMSE, MAE, MAPE, Training Time
- Generates a CD (critical difference) style summary
"""

import os
import glob
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# CONFIG
# ============================================================

BASE_RESULTS = "Results"
OUTPUT_FOLDER = "Results/Phase6_Final"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Map model folder name -> display name
MODEL_MAP = {
    "ANN":           "ANN",
    "CatBoost":      "CatBoost",
    "LightGBM":      "LightGBM",
    "TabNet":        "TabNet",
    "FTTransformer": "FT-Transformer",
    "Phase5_Baselines/Linear_Regression":          "Linear Regression",
    "Phase5_Baselines/Polynomial_deg2":            "Poly (deg=2)",
    "Phase5_Baselines/Polynomial_deg3":            "Poly (deg=3)",
}

# ============================================================
# COLLECT METRICS
# ============================================================

all_metrics = []

for folder_suffix, display_name in MODEL_MAP.items():
    metric_path = os.path.join(BASE_RESULTS, folder_suffix, "metrics.txt")
    if not os.path.exists(metric_path):
        print(f"  [SKIP] metrics.txt not found: {metric_path}")
        continue

    with open(metric_path) as f:
        lines = f.readlines()

    row = {"Model": display_name}
    for line in lines:
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip().replace("%", "")
            try:
                row[k] = float(v)
            except ValueError:
                pass
    all_metrics.append(row)

df = pd.DataFrame(all_metrics)

# Canonical column renames (handle spaces in keys)
rename_map = {
    "R2":                 "R²",
    "Adjusted R2":        "Adj. R²",
    "RMSE":               "RMSE",
    "MAE":                "MAE",
    "MAPE":               "MAPE (%)",
    "Explained Var":      "Exp. Var",
    "Training Time (s)":  "Train Time (s)",
    "Prediction Time (s)":"Pred Time (s)",
}
df = df.rename(columns=rename_map)

# Define column order
cols_order = ["Model", "R²", "Adj. R²", "RMSE", "MAE", "MAPE (%)",
              "Exp. Var", "Train Time (s)", "Pred Time (s)"]
cols_order = [c for c in cols_order if c in df.columns]
df = df[cols_order].copy()
df_display = df.round(5)

print("\n" + "="*90)
print("FINAL COMPREHENSIVE COMPARISON TABLE")
print("="*90)
print(df_display.to_string(index=False))

# Save table
df_display.to_excel(os.path.join(OUTPUT_FOLDER, "Final_Comparison_Table.xlsx"), index=False)
df_display.to_csv(os.path.join(OUTPUT_FOLDER, "Final_Comparison_Table.csv"), index=False)

# LaTeX table
df_display.to_latex(
    os.path.join(OUTPUT_FOLDER, "Final_Comparison_Table.tex"),
    index=False, escape=True
)

# ============================================================
# PUBLICATION-QUALITY PLOTS
# ============================================================

PALETTE = [
    '#E64B35', '#4DBBD5', '#00A087', '#3C5488',
    '#F39B7F', '#8491B4', '#91D1C2', '#DC0000'
]

# Helper
def bar_with_labels(ax, x, heights, colors, xtick_labels, ylabel, title, ylim=None):
    bars = ax.bar(x, heights, color=colors[:len(heights)], width=0.6, edgecolor='white', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    if ylim:
        ax.set_ylim(ylim)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.4f}',
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

models = df["Model"].tolist()
x = np.arange(len(models))
colors = PALETTE[:len(models)]

# --- R² ---
fig, ax = plt.subplots(figsize=(11, 5))
bar_with_labels(ax, x, df["R²"].tolist(), colors, models, "R²", "R² Score — All Models", ylim=(0, 1.12))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "Comparison_R2.png"), dpi=300, bbox_inches='tight')
plt.close()

# --- RMSE & MAE side-by-side ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bar_with_labels(axes[0], x, df["RMSE"].tolist(), colors, models, "RMSE", "RMSE — All Models")
bar_with_labels(axes[1], x, df["MAE"].tolist(),  colors, models, "MAE",  "MAE — All Models")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "Comparison_RMSE_MAE.png"), dpi=300, bbox_inches='tight')
plt.close()

# --- MAPE ---
fig, ax = plt.subplots(figsize=(11, 5))
bar_with_labels(ax, x, df["MAPE (%)"].tolist(), colors, models, "MAPE (%)", "MAPE (%) — All Models")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "Comparison_MAPE.png"), dpi=300, bbox_inches='tight')
plt.close()

# --- Training Time (log scale) ---
if "Train Time (s)" in df.columns:
    fig, ax = plt.subplots(figsize=(11, 5))
    times = df["Train Time (s)"].tolist()
    bars = ax.bar(x, times, color=colors[:len(models)], width=0.6, edgecolor='white')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("Training Time (s) — Log Scale", fontsize=11)
    ax.set_title("Training Time — All Models", fontsize=12, fontweight='bold', pad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}s',
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "Comparison_TrainingTime.png"), dpi=300, bbox_inches='tight')
    plt.close()

# --- Radar Chart (normalised metrics) ---
def make_radar(df_in, metrics, labels, title, filename):
    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=10)

    col_arrays = {}
    for m in metrics:
        col_arrays[m] = df_in[m].values.astype(float)

    # Normalise each metric 0-1 (higher = better for R²; invert for errors)
    norm = {}
    for m in metrics:
        mn, mx = col_arrays[m].min(), col_arrays[m].max()
        if mx == mn:
            norm[m] = np.zeros_like(col_arrays[m])
        elif m == "R²":
            norm[m] = (col_arrays[m] - mn) / (mx - mn)
        else:  # error metrics: lower is better → invert
            norm[m] = 1 - (col_arrays[m] - mn) / (mx - mn)

    for i, row in df_in.iterrows():
        values = [norm[m][i] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row["Model"],
                color=PALETTE[i % len(PALETTE)])
        ax.fill(angles, values, alpha=0.08, color=PALETTE[i % len(PALETTE)])

    ax.set_title(title, size=13, fontweight='bold', y=1.12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

radar_metrics = ["R²", "RMSE", "MAE", "MAPE (%)"]
radar_metrics = [m for m in radar_metrics if m in df.columns]
make_radar(df.reset_index(drop=True), radar_metrics, models,
           "Model Performance Radar Chart",
           os.path.join(OUTPUT_FOLDER, "Comparison_Radar.png"))

print(f"\nAll Phase 6 plots and tables saved in: {OUTPUT_FOLDER}")

# ============================================================
# RANK TABLE
# ============================================================

rank_df = df[["Model", "R²", "RMSE", "MAE", "MAPE (%)"]].copy()
rank_df["Rank_R2"]   = rank_df["R²"].rank(ascending=False).astype(int)
rank_df["Rank_RMSE"] = rank_df["RMSE"].rank(ascending=True).astype(int)
rank_df["Rank_MAE"]  = rank_df["MAE"].rank(ascending=True).astype(int)
rank_df["Rank_MAPE"] = rank_df["MAPE (%)"].rank(ascending=True).astype(int)
rank_df["Average_Rank"] = rank_df[["Rank_R2","Rank_RMSE","Rank_MAE","Rank_MAPE"]].mean(axis=1)
rank_df = rank_df.sort_values("Average_Rank")
rank_df.to_excel(os.path.join(OUTPUT_FOLDER, "Model_Ranking.xlsx"), index=False)

print("\n--- MODEL RANKING (lower average rank = better) ---")
print(rank_df[["Model","R²","RMSE","Average_Rank"]].to_string(index=False))
