from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
import joblib

app = Flask(__name__)

# Cargar modelo y columnas
modelo = xgb.XGBClassifier()
modelo.load_model("modelo_xgboost_v5.json")
features = joblib.load("xgboost_features_v3.pkl")

@app.route("/")
def home():
    return "✅ API de predicción activa"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        # Codificación one-hot
        df_encoded = pd.get_dummies(df)

        # Asegurar que tenga todas las columnas necesarias
        for col in features:
            if col not in df_encoded:
                df_encoded[col] = 0

        df_encoded = df_encoded[features]

        # Predicción
        pred = modelo.predict(df_encoded)[0]

        return jsonify({"rentable_predicha": int(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render usa PORT
    app.run(host="0.0.0.0", port=port, debug=True)
