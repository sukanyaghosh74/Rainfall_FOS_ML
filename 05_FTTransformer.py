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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tab_transformer_pytorch import FTTransformer

# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# CREATE RESULTS FOLDER
# ============================================================

RESULT_FOLDER = "Results/FTTransformer"
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

feature_names = list(X.columns)

# Separate categorical and continuous columns for FT-Transformer
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
continuous_cols = X.select_dtypes(exclude=['object', 'category']).columns

# Encode categorical variables and get num categories
categories = []
for col in categorical_cols:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col].astype(str))
    categories.append(X[col].nunique())

categories = tuple(categories)
num_continuous = len(continuous_cols)

# Normalize continuous variables
scaler = MinMaxScaler()
X[continuous_cols] = scaler.fit_transform(X[continuous_cols])

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=SEED
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# Prepare tensors
def prep_tensors(X_split, y_split):
    if len(categorical_cols) > 0:
        x_categ = torch.tensor(X_split[categorical_cols].values, dtype=torch.long)
    else:
        x_categ = torch.empty((X_split.shape[0], 0), dtype=torch.long)
    
    if num_continuous > 0:
        x_cont = torch.tensor(X_split[continuous_cols].values, dtype=torch.float32)
    else:
        x_cont = torch.empty((X_split.shape[0], 0), dtype=torch.float32)
        
    y_tensor = torch.tensor(y_split.values, dtype=torch.float32).unsqueeze(1)
    return x_categ, x_cont, y_tensor

x_categ_train, x_cont_train, y_train_t = prep_tensors(X_train, y_train)
x_categ_test, x_cont_test, y_test_t = prep_tensors(X_test, y_test)

train_dataset = TensorDataset(x_categ_train, x_cont_train, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ============================================================
# BUILD & TRAIN MODEL
# ============================================================

model = FTTransformer(
    categories = categories,
    num_continuous = num_continuous,
    dim = 32,
    dim_out = 1,
    depth = 6,
    heads = 8,
    attn_dropout = 0.1,
    ff_dropout = 0.1
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 200
patience = 20
best_val_loss = float('inf')
patience_counter = 0

train_losses = []
val_losses = []

start_time = time.time()

for epoch in range(epochs):
    model.train()
    batch_losses = []
    for x_cat, x_con, y_batch in train_loader:
        x_cat, x_con, y_batch = x_cat.to(device), x_con.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        # FTTransformer signature expects x_categ, x_cont
        preds = model(x_cat, x_con)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        
        batch_losses.append(loss.item())
        
    train_loss = np.mean(batch_losses)
    train_losses.append(train_loss)
    
    model.eval()
    with torch.no_grad():
        preds_val = model(x_categ_test.to(device), x_cont_test.to(device))
        val_loss = criterion(preds_val, y_test_t.to(device)).item()
        val_losses.append(val_loss)
        
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(RESULT_FOLDER, "best_model.pt"))
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

training_time = time.time() - start_time

# Load best model
model.load_state_dict(torch.load(os.path.join(RESULT_FOLDER, "best_model.pt")))
model.eval()

# ============================================================
# PREDICTIONS
# ============================================================

start_time = time.time()
with torch.no_grad():
    predictions = model(x_categ_test.to(device), x_cont_test.to(device)).cpu().numpy().flatten()
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
print("FT-TRANSFORMER RESULTS")
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
residuals = y_test.values - predictions
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

# Loss Curve
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_FOLDER, "Loss_Curve.png"), dpi=300)
plt.close()

print("\nAll results saved inside", RESULT_FOLDER)
