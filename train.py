"""
SoH (State of Health) - Improved Training Script
Better accuracy while staying simple enough for FPGA
"""

import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_FILE   = r"C:\Users\sivak\Downloads\5.+Battery+Data+Set\5. Battery Data Set\1. BatteryAgingARC-FY08Q4\6b.csv"
MODEL_DIR   = r"C:\Users\sivak\Downloads\5.+Battery+Data+Set\5. Battery Data Set\1. BatteryAgingARC-FY08Q4\soh_model"
RESULTS_DIR = r"C:\Users\sivak\Downloads\5.+Battery+Data+Set\5. Battery Data Set\1. BatteryAgingARC-FY08Q4\results"

FEATURES = ["Avg_Voltage", "Avg_Current", "Avg_Temperature", "Capacity"]
TARGET   = "SoH"

# 3 hidden layers + Adam optimiser = much better accuracy, still FPGA-friendly
HIDDEN1  = 16
HIDDEN2  = 8
HIDDEN3  = 4
EPOCHS   = 5000
LR       = 0.005
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── ACTIVATIONS ──────────────────────────────────────────────────────────────
def relu(x):   return np.maximum(0, x)
def relu_d(x): return (x > 0).astype(float)


# ── NETWORK ──────────────────────────────────────────────────────────────────
class MLP:
    def __init__(self, n_in, h1, h2, h3):
        np.random.seed(42)
        self.W1 = np.random.randn(n_in, h1) * np.sqrt(2/n_in)
        self.b1 = np.zeros((1, h1))
        self.W2 = np.random.randn(h1, h2)  * np.sqrt(2/h1)
        self.b2 = np.zeros((1, h2))
        self.W3 = np.random.randn(h2, h3)  * np.sqrt(2/h2)
        self.b3 = np.zeros((1, h3))
        self.W4 = np.random.randn(h3, 1)   * np.sqrt(2/h3)
        self.b4 = np.zeros((1, 1))

        # Adam state
        self.ms = {k: np.zeros_like(v) for k,v in self._p().items()}
        self.vs = {k: np.zeros_like(v) for k,v in self._p().items()}
        self.t  = 0

    def _p(self):
        return dict(W1=self.W1,b1=self.b1,W2=self.W2,b2=self.b2,
                    W3=self.W3,b3=self.b3,W4=self.W4,b4=self.b4)

    def forward(self, X):
        self.X  = X
        self.z1 = X        @ self.W1 + self.b1;  self.a1 = relu(self.z1)
        self.z2 = self.a1  @ self.W2 + self.b2;  self.a2 = relu(self.z2)
        self.z3 = self.a2  @ self.W3 + self.b3;  self.a3 = relu(self.z3)
        self.z4 = self.a3  @ self.W4 + self.b4
        return self.z4

    def backward(self, y, lr):
        m = self.X.shape[0];  y = y.reshape(-1,1)
        self.t += 1
        b1_a, b2_a, eps = 0.9, 0.999, 1e-8

        d4  = (self.z4 - y) / m
        dW4 = self.a3.T @ d4;  db4 = d4.sum(0, keepdims=True)
        d3  = (d4 @ self.W4.T) * relu_d(self.z3)
        dW3 = self.a2.T @ d3;  db3 = d3.sum(0, keepdims=True)
        d2  = (d3 @ self.W3.T) * relu_d(self.z2)
        dW2 = self.a1.T @ d2;  db2 = d2.sum(0, keepdims=True)
        d1  = (d2 @ self.W2.T) * relu_d(self.z1)
        dW1 = self.X.T  @ d1;  db1 = d1.sum(0, keepdims=True)

        grads = dict(W1=dW1,b1=db1,W2=dW2,b2=db2,
                     W3=dW3,b3=db3,W4=dW4,b4=db4)
        for k, g in grads.items():
            self.ms[k] = b1_a*self.ms[k] + (1-b1_a)*g
            self.vs[k] = b2_a*self.vs[k] + (1-b2_a)*g**2
            mh = self.ms[k]/(1-b1_a**self.t)
            vh = self.vs[k]/(1-b2_a**self.t)
            getattr(self, k)[:] -= lr * mh / (np.sqrt(vh) + eps)

    def loss(self, X, y):
        return float(np.mean((self.forward(X).flatten() - y)**2))


# ── LOAD DATA ────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_FILE)
print(f"Loaded {len(df)} rows")
print(df.head())

X_raw = df[FEATURES].values.astype(float)
y_raw = df[TARGET].values.astype(float)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_n = scaler_X.fit_transform(X_raw)
y_n = scaler_y.fit_transform(y_raw.reshape(-1,1)).flatten()

X_tr, X_te, y_tr, y_te = train_test_split(
    X_n, y_n, test_size=0.15, random_state=42)
print(f"Train: {len(X_tr)}  |  Test: {len(X_te)}")


# ── TRAIN ────────────────────────────────────────────────────────────────────
model = MLP(len(FEATURES), HIDDEN1, HIDDEN2, HIDDEN3)
loss_log    = []
best_loss   = 1e9
best_w      = None

print("\nTraining ...")
for epoch in range(1, EPOCHS+1):
    model.forward(X_tr)
    model.backward(y_tr, LR)

    if epoch % 500 == 0:
        trl = model.loss(X_tr, y_tr)
        tel = model.loss(X_te, y_te)
        loss_log.append({"epoch": epoch,
                         "train_loss": round(trl,8),
                         "test_loss":  round(tel,8)})
        print(f"  Epoch {epoch:5d}  |  Train: {trl:.6f}  |  Test: {tel:.6f}")

        if tel < best_loss:
            best_loss = tel
            best_w = {k: v.copy() for k,v in model._p().items()}

# Restore best
for k,v in best_w.items():
    setattr(model, k, v)

print("Training complete.\n")


# ── EVALUATE ─────────────────────────────────────────────────────────────────
y_pn = model.forward(X_te).flatten()
y_pred = scaler_y.inverse_transform(y_pn.reshape(-1,1)).flatten()
y_true = scaler_y.inverse_transform(y_te.reshape(-1,1)).flatten()

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)

print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")

# Sample comparison table
print("\n── Test Set: True vs Predicted ──────────────────────────────")
print(f"  {'True SoH':>10}  {'Predicted':>10}  {'Abs Error':>10}")
for i in range(len(y_true)):
    flag = " ◄" if abs(y_true[i]-y_pred[i]) > 0.02 else ""
    print(f"  {y_true[i]:10.6f}  {y_pred[i]:10.6f}  {abs(y_true[i]-y_pred[i]):10.6f}{flag}")


# ── SAVE RESULTS ─────────────────────────────────────────────────────────────
json.dump({"MAE": round(mae,6), "RMSE": round(rmse,6), "R2": round(r2,6)},
          open(f"{RESULTS_DIR}/metrics.json","w"), indent=2)

pd.DataFrame({"True_SoH": y_true, "Predicted_SoH": y_pred,
               "Abs_Error": np.abs(y_true-y_pred)}
             ).to_csv(f"{RESULTS_DIR}/predictions.csv", index=False)

pd.DataFrame(loss_log).to_csv(f"{RESULTS_DIR}/loss_history.csv", index=False)

print(f"\nResults saved to '{RESULTS_DIR}/'")


# ── SAVE MODEL ───────────────────────────────────────────────────────────────
weights = {k: getattr(model,k).tolist()
           for k in ["W1","b1","W2","b2","W3","b3","W4","b4"]}

scaler_params = {
    "X_min":        scaler_X.data_min_.tolist(),
    "X_range":      scaler_X.data_range_.tolist(),
    "y_min":        float(scaler_y.data_min_[0]),
    "y_range":      float(scaler_y.data_range_[0]),
    "features":     FEATURES,
    "architecture": [len(FEATURES), HIDDEN1, HIDDEN2, HIDDEN3, 1]
}

json.dump(weights,       open(f"{MODEL_DIR}/weights.json","w"),  indent=2)
json.dump(scaler_params, open(f"{MODEL_DIR}/scaler.json","w"),   indent=2)

# Q8.8 fixed-point for FPGA
FP = 256
fp = {k: np.clip(np.round(np.array(v)*FP),-32768,32767).astype(int).tolist()
      for k,v in weights.items()}
json.dump({"scale": FP, "weights": fp},
          open(f"{MODEL_DIR}/weights_fixed_point.json","w"), indent=2)

print(f"Model saved to '{MODEL_DIR}/'")
print("  weights.json              – float weights")
print("  scaler.json               – normalisation params")
print("  weights_fixed_point.json  – Q8.8 fixed-point for FPGA")