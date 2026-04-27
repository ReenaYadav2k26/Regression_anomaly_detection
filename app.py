from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# -----------------------------
# 🔹 BASE DIRECTORY (IMPORTANT)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# 🔹 LOAD ARTIFACTS SAFELY
# -----------------------------
def load_artifacts():
    try:
        model_path = os.path.join(BASE_DIR, "model", "regression_model.pkl")
        scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")
        features_path = os.path.join(BASE_DIR, "model", "features.pkl")
        threshold_path = os.path.join(BASE_DIR, "model", "threshold_norm.pkl")

        print("📂 Loading artifacts...")
        print("Model exists:", os.path.exists(model_path))

        model = pickle.load(open(model_path, "rb"))
        scaler = pickle.load(open(scaler_path, "rb"))
        features = pickle.load(open(features_path, "rb"))
        threshold = pickle.load(open(threshold_path, "rb"))

        print("✅ Artifacts loaded successfully")

        return model, scaler, features, threshold

    except Exception as e:
        print(f"🔥 ERROR loading artifacts: {e}")
        return None, None, None, None


model, scaler, features, threshold = load_artifacts()

# -----------------------------
# 🔐 API KEY (ENV BASED)
# -----------------------------
API_KEY = os.getenv("API_KEY", "my-secret-key-123")

# -----------------------------
# 🔹 ROOT ROUTE (ADD THIS)
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "API is running 🚀",
        "endpoints": {
            "/health": "GET",
            "/predict": "POST"
        }
    })


# -----------------------------
# 🔹 HEALTH CHECK
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy" if model else "error",
        "model_loaded": model is not None
    })

# -----------------------------
# 🔹 PREDICT ROUTE
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 🔐 API KEY CHECK
        if request.headers.get("x-api-key") != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401

        # 🚨 MODEL LOADED CHECK
        if model is None or scaler is None or features is None:
            return jsonify({"error": "Model not loaded properly"}), 500

        data = request.get_json()

        if not data or "invoice_features" not in data:
            return jsonify({"error": "Missing invoice_features"}), 400

        if not isinstance(data["invoice_features"], dict):
            return jsonify({"error": "invoice_features must be an object"}), 400

        # -----------------------------
        # 🔹 INPUT PREPARATION
        # -----------------------------
        df_in = pd.DataFrame([data["invoice_features"]])

        # Ensure all expected features exist
        for col in features:
            if col not in df_in.columns:
                df_in[col] = 0

        df_in = df_in.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Separate X (exclude target)
        X_cols = [f for f in features if f != 'debit']
        X = df_in[X_cols]

        # 🚨 SAFE SCALING
        try:
            X_scaled = scaler.transform(X)
        except Exception as e:
            return jsonify({"error": f"Scaling failed: {str(e)}"}), 500

        # 🚨 SAFE TARGET ACCESS
        if 'debit' not in df_in.columns:
            return jsonify({"error": "Missing 'debit' in input"}), 400

        y_actual = float(df_in['debit'].values[0])

        # 🚨 SAFE PREDICTION
        try:
            y_pred = float(model.predict(X_scaled)[0])
        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

        # -----------------------------
        # 🔹 ANOMALY SCORE
        # -----------------------------
        norm_score = (abs(y_actual - y_pred) / (y_actual + 5.0)) * 100

        return jsonify({
            "anomaly_score_percentage": round(norm_score, 4),
            "is_anomaly": bool(norm_score > threshold),
            "predicted_debit": round(y_pred, 2),
            "actual_debit": y_actual,
            "threshold_used": round(float(threshold), 2)
        })

    except Exception as e:
        print(f"🔥 UNEXPECTED ERROR: {e}")
        return jsonify({"error": str(e)}), 500


# -----------------------------
# 🔹 MAIN
# -----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
