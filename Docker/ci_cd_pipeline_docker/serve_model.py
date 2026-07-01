"""
API Flask pentru servirea modelului Random Forest ENHANCED (joblib).

Endpoint-uri:
  GET  /health      - verificare disponibilitate
  GET  /model-info  - metadata model
  POST /predict     - predictie timeOnSite (secunde)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from flask import Flask, jsonify, request

from feature_engineering import engineer_features_from_input
from metadata_utils import (
    get_features_all,
    get_hyperparameters,
    get_model_type,
    get_model_version,
    get_features_total,
    get_pageviews_median,
    get_performance,
)
from model_loader import load_artifacts

app = Flask(__name__)

# Incarcam artefactele la pornirea containerului / procesului
MODEL, SCALER, METADATA = load_artifacts()
FEATURES_ALL = get_features_all(METADATA)
PAGE_VIEWS_MEDIAN = get_pageviews_median(METADATA)
MODEL_VERSION = get_model_version(METADATA)


@app.route("/health", methods=["GET"])
def health():
    """Health check folosit de Docker si GitHub Actions."""
    return jsonify({"status": "ok", "model": "RandomForestRegressor_ENHANCED"}), 200


@app.route("/model-info", methods=["GET"])
def model_info():
    """Returneaza informatii despre modelul incarcat (din metadata JSON)."""
    performance = get_performance(METADATA)
    return jsonify(
        {
            "model_type": get_model_type(METADATA),
            "model_version": MODEL_VERSION,
            "features_total": get_features_total(METADATA),
            "features_all": FEATURES_ALL,
            "hyperparameters": get_hyperparameters(METADATA),
            "performance": performance,
        }
    ), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Primeste JSON cu campurile de baza si returneaza timp estimat pe site.

    Exemplu body:
    {
        "pageviews": 5,
        "visitNumber": 2,
        "hits": 12,
        "device_category": "desktop",
        "country": "United States"
    }
    """
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "JSON body required"}), 400

    required = ("pageviews", "visitNumber", "hits")
    missing = [field for field in required if field not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        features_df = engineer_features_from_input(
            payload,
            features_all=FEATURES_ALL,
            pageviews_median=PAGE_VIEWS_MEDIAN,
        )
        features_scaled = SCALER.transform(features_df)
        prediction = float(MODEL.predict(features_scaled)[0])
        prediction = max(0.0, prediction)

        return jsonify(
            {
                "predicted_time_on_site_seconds": round(prediction, 2),
                "predicted_time_on_site_minutes": round(prediction / 60, 2),
                "model_version": MODEL_VERSION,
                "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ), 200
    except Exception as exc:  # noqa: BLE001 - mesaj clar catre client API
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8501"))
    app.run(host="0.0.0.0", port=port)
