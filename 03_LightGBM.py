import os
import glob
import random
import warnings
import time

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score
)
import shap

import lightgbm as lgb

# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# CREATE RESULTS FOLDER
# ============================================================

RESULT_FOLDER = "Results/LightGBM"
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ============================================================
# READ EXCEL FILE
# ============================================================

excel_files = glob.glob("*.xlsx")

if len(excel_files) == 0:
    raise FileNotFoundError("No Excel file found in this folder.")

FILE_NAME = excel_files[0]

print(f"\nReading File : {FILE_NAME}")

df = pd.read_excel(FILE_NAME)

# ============================================================
# CLEAN DATA
# ============================================================

# Remove Unnamed columns (which contain notes/artifacts)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Remove empty columns
df = df.dropna(axis=1, how="all")

# Remove rows with missing values
df = df.dropna()

print("\nDataset Shape :", df.shape)

# ============================================================
# SELECT FEATURES
# ============================================================

X = df.drop(columns=["FOS"]).copy()
y = df["FOS"].copy()

# ============================================================
# ENCODE CATEGORICAL VARIABLE
# ============================================================

feature_names = list(X.columns)
# Dynamically encode any string/object columns
for col in X.select_dtypes(include=['object', 'category']).columns:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col].astype(str))

# ============================================================
# NORMALIZATION
# ============================================================

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=SEED
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ============================================================
# BUILD & TRAIN MODEL
# ============================================================

model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=SEED,
    n_jobs=-1
)

start_time = time.time()
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=100)]
)
training_time = time.time() - start_time

# ============================================================
# PREDICTIONS
# ============================================================

start_time = time.time()
predictions = model.predict(X_test)
prediction_time = time.time() - start_time

# ============================================================
# METRICS
# ============================================================

R2 = r2_score(y_test, predictions)

# Adjusted R2
n = len(y_test)
p = X_test.shape[1]
Adj_R2 = 1 - ((1 - R2) * (n - 1) / (n - p - 1)) if n > p + 1 else np.nan

RMSE = np.sqrt(mean_squared_error(y_test, predictions))
MAE = mean_absolute_error(y_test, predictions)
MAPE = np.mean(np.abs((y_test - predictions) / y_test)) * 100
Exp_Var = explained_variance_score(y_test, predictions)

print("\n==============================")
print("LIGHTGBM RESULTS")
print("==============================")

print(f"R2               : {R2:.5f}")
print(f"Adjusted R2      : {Adj_R2:.5f}")
print(f"RMSE             : {RMSE:.5f}")
print(f"MAE              : {MAE:.5f}")
print(f"MAPE             : {MAPE:.5f}%")
print(f"Explained Var    : {Exp_Var:.5f}")
print(f"Training Time (s): {training_time:.5f}")
print(f"Prediction Time (s): {prediction_time:.5f}")

# ============================================================
# SAVE METRICS
# ============================================================

with open(os.path.join(RESULT_FOLDER, "metrics.txt"), "w") as f:
    f.write(f"R2               : {R2}\n")
    f.write(f"Adjusted R2      : {Adj_R2}\n")
    f.write(f"RMSE             : {RMSE}\n")
    f.write(f"MAE              : {MAE}\n")
    f.write(f"MAPE             : {MAPE}\n")
    f.write(f"Explained Var    : {Exp_Var}\n")
    f.write(f"Training Time (s): {training_time}\n")
    f.write(f"Prediction Time (s): {prediction_time}\n")

# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({"Actual": y_test.values, "Predicted": predictions})
prediction_df.to_excel(os.path.join(RESULT_FOLDER, "Predictions.xlsx"), index=False)

# ============================================================
# PLOTS
# ============================================================

# Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, predictions, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual FOS")
plt.ylabel("Predicted FOS")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_FOLDER, "Actual_vs_Predicted.png"), dpi=300)
plt.close()

# Residual Plot
residuals = y_test - predictions
plt.figure(figsize=(7,5))
plt.scatter(predictions, residuals, color="green")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted FOS")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_FOLDER, "Residuals.png"), dpi=300)
plt.close()

# Residual Histogram
plt.figure(figsize=(7,5))
plt.hist(residuals, bins=20, color='purple', edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Histogram')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_FOLDER, "Residuals_Histogram.png"), dpi=300)
plt.close()

# Feature Importance
importances = model.feature_importances_
plt.figure(figsize=(8,6))
plt.barh(feature_names, importances, color='teal')
plt.xlabel("Feature Importance")
plt.title("LightGBM Feature Importance")
plt.grid(True, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_FOLDER, "Feature_Importance.png"), dpi=300)
plt.close()

# SHAP Values
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_FOLDER, "SHAP_Summary.png"), dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"Could not generate SHAP values: {e}")

# ============================================================
# SAVE MODEL
# ============================================================
model.booster_.save_model(os.path.join(RESULT_FOLDER, "LightGBM_Model.txt"))
print("\nAll results saved inside", RESULT_FOLDER)
