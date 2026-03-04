"""
SoH (State of Health) - Prediction Script
"""

import numpy as np
import json
import os

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_DIR = r"C:\Users\sivak\Downloads\5.+Battery+Data+Set\5. Battery Data Set\1. BatteryAgingARC-FY08Q4\soh_model"
# ─────────────────────────────────────────────────────────────────────────────

# ── LOAD MODEL ───────────────────────────────────────────────────────────────
weights_path = os.path.join(MODEL_DIR, "weights.json")   # <-- FIXED: no double path
scaler_path  = os.path.join(MODEL_DIR, "scaler.json")    # <-- FIXED: no double path

if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Model not found at '{MODEL_DIR}'. Run train.py first.")

with open(weights_path, "r") as f:
    w = json.load(f)

with open(scaler_path, "r") as f:
    sc = json.load(f)

# ── LOAD ALL 4 LAYERS ────────────────────────────────────────────────────────
W1 = np.array(w["W1"]); b1 = np.array(w["b1"])
W2 = np.array(w["W2"]); b2 = np.array(w["b2"])
W3 = np.array(w["W3"]); b3 = np.array(w["b3"])
W4 = np.array(w["W4"]); b4 = np.array(w["b4"])   # <-- FIXED: was missing

X_min    = np.array(sc["X_min"])
X_range  = np.array(sc["X_range"])
y_min    = sc["y_min"]
y_range  = sc["y_range"]
features = sc["features"]

# Quick sanity check — print scaler range so you can verify it's correct
print(f"  Scaler y_min={y_min:.4f}  y_range={y_range:.4f}")
print(f"  (y_min should be ~0.56, y_range should be ~0.44)")


# ── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
def relu(x):
    return np.maximum(0, x)

def predict(raw_values):
    x   = ((np.array(raw_values, dtype=float) - X_min) / X_range).reshape(1, -1)
    a1  = relu(x  @ W1 + b1)
    a2  = relu(a1 @ W2 + b2)
    a3  = relu(a2 @ W3 + b3)          # <-- FIXED: 3 relu layers
    out = (a3 @ W4 + b4).flatten()[0] # <-- FIXED: final linear layer
    return float(out * y_range + y_min)


# ── PREDICT ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 50)
    print("  SoH Prediction")
    print(f"  Features : {features}")
    print("=" * 50)

    # Single sample (hardcoded)
    sample = {
        "Avg_Voltage":     3.556946062,
        "Avg_Current":    -1.990533351,
        "Avg_Temperature": 32.14277786,
        "Capacity":        2.035337591,
    }
    input_vals = [sample[f] for f in features]
    soh = predict(input_vals)
    print(f"\nInput  : {sample}")
    print(f"SoH    : {soh:.6f}  ({soh*100:.2f}%)")

    # Batch prediction from CSV (optional)
    BATCH_FILE = r"C:\Users\sivak\Downloads\5.+Battery+Data+Set\5. Battery Data Set\1. BatteryAgingARC-FY08Q4\new_data.csv"
    if os.path.exists(BATCH_FILE):
        import pandas as pd
        df = pd.read_csv(BATCH_FILE)
        results = []
        for _, row in df.iterrows():
            vals = [row[f] for f in features]
            results.append({**row.to_dict(), "Predicted_SoH": round(predict(vals), 6)})
        out_df = pd.DataFrame(results)
        out_path = os.path.join(MODEL_DIR.replace("soh_model", "results"), "batch_predictions.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"\nBatch predictions saved to '{out_path}'")

    # Interactive input
    print("\n" + "-" * 50)
    print("Enter custom values (press Enter to skip):")
    try:
        vals = []
        for feat in features:
            v = input(f"  {feat}: ").strip()
            if v == "":
                print("Skipped.")
                break
            vals.append(float(v))
        if len(vals) == len(features):
            soh = predict(vals)
            print(f"\n  Predicted SoH : {soh:.6f}  ({soh*100:.2f}%)")
    except (EOFError, KeyboardInterrupt):
        pass