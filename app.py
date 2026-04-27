from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# --- LOAD PRODUCTION ARTIFACTS ---
def load_artifacts():
    try:
        model = pickle.load(open("model/regression_model.pkl", "rb"))
        scaler = pickle.load(open("model/scaler.pkl", "rb"))
        features = pickle.load(open("model/features.pkl", "rb"))
        threshold = pickle.load(open("model/threshold_norm.pkl", "rb"))
        return model, scaler, features, threshold
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return None, None, None, None

model, scaler, features, threshold = load_artifacts()
API_KEY = "my-secret-key-123" # Should be moved to env var in production

@app.route("/predict", methods=["POST"])
def predict():
    if request.headers.get("x-api-key") != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    if not data or "invoice_features" not in data:
        return jsonify({"error": "Missing invoice_features"}), 400

    # Prepare input
    df_in = pd.DataFrame([data["invoice_features"]])
    
    # Ensure feature alignment (excluding target 'debit')
    X_cols = [f for f in features if f != 'debit']
    X = df_in[X_cols].fillna(0)
    X_scaled = scaler.transform(X)

    y_actual = float(df_in['debit'].values[0])
    y_pred = float(model.predict(X_scaled)[0])

    # Stabilized Deviation Formula
    norm_score = (abs(y_actual - y_pred) / (y_actual + 5.0)) * 100

    return jsonify({
        "anomaly_score_percentage": round(norm_score, 4),
        "is_anomaly": bool(norm_score > threshold),
        "predicted_debit": round(y_pred, 2),
        "actual_debit": y_actual,
        "threshold_used": round(threshold, 2)
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
