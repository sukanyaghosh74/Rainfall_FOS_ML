"""
12_Additional_Plots.py
========================
Adds the two analyses flagged by reviewer:
1. Taylor Diagram — comparing all models simultaneously
2. Predicted vs Actual Density Plot (hexbin + KDE) for CatBoost

Run this AFTER all other scripts.
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
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RESULT_FOLDER = "Results/Phase6_Final"
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ============================================================
# LOAD & PREPROCESS (identical pipeline)
# ============================================================

excel_files = glob.glob("*.xlsx")
FILE_NAME   = excel_files[0]
print(f"Reading: {FILE_NAME}")

df = pd.read_excel(FILE_NAME)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.dropna(axis=1, how="all").dropna().reset_index(drop=True)
print(f"Dataset Shape: {df.shape}")

X_raw = df.drop(columns=["FOS"]).copy()
y     = df["FOS"].copy().reset_index(drop=True)
feature_names = list(X_raw.columns)

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
# TRAIN ALL MODELS
# ============================================================

print("\nTraining all models for additional plots...")

models_preds = {}

# CatBoost
cb = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6,
                       loss_function='RMSE', random_seed=SEED,
                       early_stopping_rounds=50, verbose=0)
cb.fit(X_train, y_train, eval_set=(X_test, y_test))
models_preds["CatBoost"]     = cb.predict(X_test)

# LightGBM
lgbm = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6,
                          random_state=SEED, n_jobs=-1, verbosity=-1)
lgbm.fit(X_train, y_train,
         eval_set=[(X_test, y_test)],
         callbacks=[lgb.early_stopping(50, verbose=False)])
models_preds["LightGBM"]     = lgbm.predict(X_test)

# ANN (MLP)
ann = MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation='relu',
                   max_iter=500, early_stopping=True,
                   validation_fraction=0.2, n_iter_no_change=30,
                   random_state=SEED)
ann.fit(X_train, y_train)
models_preds["ANN"]          = ann.predict(X_test)

# Linear Regression
lr = Pipeline([('scaler', MinMaxScaler()), ('model', LinearRegression())])
lr.fit(X_train, y_train)
models_preds["Linear Reg."]  = lr.predict(X_test)

# Polynomial deg=2
poly2 = Pipeline([('scaler', MinMaxScaler()),
                  ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                  ('model', LinearRegression())])
poly2.fit(X_train, y_train)
models_preds["Poly (d=2)"]   = poly2.predict(X_test)

print("  All models trained.")

# ============================================================
# 1. TAYLOR DIAGRAM
# ============================================================
# Taylor diagram plots each model as a point in polar space:
#   - Angular axis  = arccos(Pearson correlation)
#   - Radial axis   = standard deviation of predictions
# The "reference point" is the observed std on the x-axis.
# RMSE is shown as arcs centered on the reference point.

def taylor_diagram(y_obs, preds_dict, title, savepath):
    obs_std = np.std(y_obs)

    fig = plt.figure(figsize=(8, 7))
    ax  = fig.add_subplot(111, projection='polar')

    # Only use the first quadrant (0 to pi/2)
    ax.set_thetamax(90)
    ax.set_thetamin(0)
    ax.set_theta_zero_location('W')   # 0 correlation on left, 1.0 on x-axis
    ax.set_theta_direction(-1)

    # Correlation labels on arc
    corr_ticks = [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    theta_ticks = [np.arccos(c) for c in corr_ticks]
    ax.set_thetagrids(np.degrees(theta_ticks),
                      labels=[str(c) for c in corr_ticks], fontsize=9)
    ax.set_xlabel("Correlation", labelpad=20)

    # Radial axis = std dev
    max_std = obs_std * 1.6
    ax.set_rlim(0, max_std)
    ax.set_ylabel("Standard Deviation", labelpad=40)

    # RMSE arcs centered on the reference point (obs_std, 0)
    for rmse_arc in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]:
        arc_theta = np.linspace(0, np.pi/2, 200)
        arc_r     = []
        for t in arc_theta:
            # r^2 + obs_std^2 - 2*r*obs_std*cos(t) = rmse^2
            # quadratic in r
            a_c = 1
            b_c = -2 * obs_std * np.cos(t)
            c_c = obs_std**2 - rmse_arc**2
            disc = b_c**2 - 4*a_c*c_c
            if disc < 0:
                arc_r.append(np.nan)
            else:
                r_pos = (-b_c + np.sqrt(disc)) / (2*a_c)
                if r_pos >= 0:
                    arc_r.append(r_pos)
                else:
                    arc_r.append((-b_c - np.sqrt(disc)) / (2*a_c))
        arc_r = np.array(arc_r)
        mask  = (arc_r >= 0) & (arc_r <= max_std)
        if mask.sum() > 2:
            ax.plot(arc_theta[mask], arc_r[mask],
                    color='lightgray', linestyle='--', linewidth=0.8, zorder=1)
            # Label arc
            mid = mask.sum() // 2
            ax.annotate(f"RMSE={rmse_arc:.2f}",
                        xy=(arc_theta[mask][mid], arc_r[mask][mid]),
                        fontsize=6.5, color='gray', ha='center')

    # Reference point (observations)
    ax.plot(0, obs_std, 'k*', markersize=12, label='Observed', zorder=5)
    ax.annotate("OBS", (0, obs_std), textcoords="offset points",
                xytext=(5, 5), fontsize=9, fontweight='bold')

    # Plot each model
    palette = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F']
    for i, (name, preds) in enumerate(preds_dict.items()):
        corr    = np.corrcoef(y_obs, preds)[0, 1]
        std_p   = np.std(preds)
        theta   = np.arccos(np.clip(corr, -1, 1))
        color   = palette[i % len(palette)]
        ax.plot(theta, std_p, 'o', markersize=9, color=color,
                label=f"{name}", zorder=5)
        ax.annotate(name, (theta, std_p),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=color)

    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1),
              fontsize=8, frameon=True)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Taylor Diagram saved: {savepath}")

taylor_diagram(y_test, models_preds,
               "Taylor Diagram — FOS Prediction (All Models)",
               os.path.join(RESULT_FOLDER, "Taylor_Diagram.png"))

# ============================================================
# 2. PREDICTED VS ACTUAL DENSITY PLOTS (CatBoost only)
# ============================================================

cb_preds = models_preds["CatBoost"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("CatBoost — Predicted vs Actual Density", fontsize=13,
             fontweight='bold', y=1.01)

# --- Panel A: Hexbin ---
ax = axes[0]
hb = ax.hexbin(y_test, cb_preds, gridsize=20, cmap='YlOrRd',
               mincnt=1, linewidths=0.3)
cb_bar = fig.colorbar(hb, ax=ax)
cb_bar.set_label("Count", fontsize=10)
lo = min(y_test.min(), cb_preds.min()) - 0.05
hi = max(y_test.max(), cb_preds.max()) + 0.05
ax.plot([lo, hi], [lo, hi], 'b--', linewidth=2, label='Perfect fit')
ax.set_xlabel("Actual FOS", fontsize=11)
ax.set_ylabel("Predicted FOS", fontsize=11)
ax.set_title("Hexbin Density Plot", fontsize=12)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.legend(fontsize=9)
r2 = r2_score(y_test, cb_preds)
rmse = np.sqrt(mean_squared_error(y_test, cb_preds))
ax.annotate(f"R²={r2:.4f}\nRMSE={rmse:.4f}",
            xy=(0.05, 0.92), xycoords='axes fraction',
            fontsize=10, va='top',
            bbox=dict(boxstyle='round', fc='white', alpha=0.85))
ax.grid(True, linestyle='--', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

# --- Panel B: KDE (2D) ---
ax2 = axes[1]
sns.kdeplot(x=y_test, y=cb_preds, ax=ax2, fill=True,
            cmap='Blues', levels=12, thresh=0.02)
ax2.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='Perfect fit')
ax2.set_xlabel("Actual FOS", fontsize=11)
ax2.set_ylabel("Predicted FOS", fontsize=11)
ax2.set_title("KDE Density Plot", fontsize=12)
ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi)
ax2.legend(fontsize=9)
ax2.annotate(f"R²={r2:.4f}\nRMSE={rmse:.4f}",
             xy=(0.05, 0.92), xycoords='axes fraction',
             fontsize=10, va='top',
             bbox=dict(boxstyle='round', fc='white', alpha=0.85))
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(RESULT_FOLDER, "CatBoost_Density_ActualVsPredicted.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] Density plots saved.")

# ============================================================
# 3. All-models KDE scatter (compact panel)
# ============================================================

model_names_ordered = ["CatBoost", "LightGBM", "ANN", "Linear Reg.", "Poly (d=2)"]
n = len(model_names_ordered)
fig, axes = plt.subplots(1, n, figsize=(4*n, 4.5), sharey=True)
fig.suptitle("Predicted vs Actual — All Models", fontsize=13, fontweight='bold')

colors_all = ['#E64B35', '#4DBBD5', '#00A087', '#8491B4', '#F39B7F']

for i, name in enumerate(model_names_ordered):
    ax = axes[i]
    preds = models_preds[name]
    r2  = r2_score(y_test, preds)
    rmse_v = np.sqrt(mean_squared_error(y_test, preds))

    ax.scatter(y_test, preds, alpha=0.5, s=18, color=colors_all[i], edgecolors='none')
    lo2 = min(y_test.min(), preds.min()) - 0.05
    hi2 = max(y_test.max(), preds.max()) + 0.05
    ax.plot([lo2, hi2], [lo2, hi2], 'k--', linewidth=1.5)
    ax.set_xlabel("Actual FOS", fontsize=9)
    if i == 0:
        ax.set_ylabel("Predicted FOS", fontsize=9)
    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.annotate(f"R²={r2:.3f}\nRMSE={rmse_v:.3f}",
                xy=(0.05, 0.88), xycoords='axes fraction',
                fontsize=8, va='top',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(RESULT_FOLDER, "AllModels_ActualVsPredicted_Panel.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] All-models panel saved.")
print(f"\nAll additional plots saved in: {RESULT_FOLDER}")
