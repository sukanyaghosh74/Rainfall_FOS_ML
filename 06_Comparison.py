import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONSTANTS & CONFIG
# ============================================================

RESULTS_DIR = "Results"
MODELS = ["ANN", "CatBoost", "LightGBM", "TabNet", "FTTransformer"]

# ============================================================
# DATA COLLECTION
# ============================================================

all_metrics = []

for model in MODELS:
    metric_file = os.path.join(RESULTS_DIR, model, "metrics.txt")
    if os.path.exists(metric_file):
        with open(metric_file, "r") as f:
            lines = f.readlines()
            
        metrics_dict = {"Model": model}
        for line in lines:
            if ":" in line:
                key, val = line.split(":")
                key = key.strip()
                val = val.strip().replace("%", "")
                
                try:
                    metrics_dict[key] = float(val)
                except ValueError:
                    metrics_dict[key] = val
                    
        all_metrics.append(metrics_dict)
    else:
        print(f"Warning: metrics.txt not found for {model}")

df_metrics = pd.DataFrame(all_metrics)

# Round floats to 4 decimals for cleaner table
df_metrics = df_metrics.round(4)

# Reorder columns logically
cols = ["Model", "R2", "Adjusted R2", "RMSE", "MAE", "MAPE", "Explained Var", "Training Time (s)", "Prediction Time (s)"]
# Use only columns that actually exist
cols = [c for c in cols if c in df_metrics.columns]
df_metrics = df_metrics[cols]

print("\n==============================")
print("FINAL MODEL COMPARISON")
print("==============================")
print(df_metrics.to_string(index=False))

# Save comparison table
comparison_folder = os.path.join(RESULTS_DIR, "Comparison")
os.makedirs(comparison_folder, exist_ok=True)
df_metrics.to_csv(os.path.join(comparison_folder, "Model_Comparison.csv"), index=False)
df_metrics.to_excel(os.path.join(comparison_folder, "Model_Comparison.xlsx"), index=False)

# ============================================================
# PLOT: R2 & Adjusted R2 Comparison
# ============================================================

x = np.arange(len(df_metrics["Model"]))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, df_metrics["R2"], width, label='R2', color='royalblue')
rects2 = ax.bar(x + width/2, df_metrics["Adjusted R2"], width, label='Adjusted R2', color='lightsteelblue')

ax.set_ylabel('Scores')
ax.set_title('R² and Adjusted R² by Model')
ax.set_xticks(x)
ax.set_xticklabels(df_metrics["Model"])
ax.legend(loc='lower right')
ax.set_ylim([0, 1.1])
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig(os.path.join(comparison_folder, "Comparison_R2.png"), dpi=300)
plt.close()

# ============================================================
# PLOT: RMSE & MAE Comparison
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, df_metrics["RMSE"], width, label='RMSE', color='indianred')
rects2 = ax.bar(x + width/2, df_metrics["MAE"], width, label='MAE', color='lightcoral')

ax.set_ylabel('Error Value')
ax.set_title('RMSE and MAE by Model')
ax.set_xticks(x)
ax.set_xticklabels(df_metrics["Model"])
ax.legend(loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig(os.path.join(comparison_folder, "Comparison_Error.png"), dpi=300)
plt.close()

# ============================================================
# PLOT: Timings Comparison
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x, df_metrics["Training Time (s)"], 0.5, label='Training Time', color='mediumseagreen')

ax.set_ylabel('Time (seconds)')
ax.set_title('Training Time by Model')
ax.set_xticks(x)
ax.set_xticklabels(df_metrics["Model"])
plt.grid(axis='y', linestyle='--', alpha=0.7)

autolabel(rects1)

fig.tight_layout()
plt.savefig(os.path.join(comparison_folder, "Comparison_Training_Time.png"), dpi=300)
plt.close()

# ============================================================
# Identify Best Model
# ============================================================

best_r2_model = df_metrics.loc[df_metrics["R2"].idxmax(), "Model"]
lowest_rmse_model = df_metrics.loc[df_metrics["RMSE"].idxmin(), "Model"]
fastest_model = df_metrics.loc[df_metrics["Training Time (s)"].idxmin(), "Model"]

print(f"\nBest Model by R2: {best_r2_model}")
print(f"Best Model by RMSE: {lowest_rmse_model}")
print(f"Fastest Model: {fastest_model}")

print("\nComparison successfully completed and saved in Results/Comparison/")
