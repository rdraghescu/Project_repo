"""Teste incarcare joblib + predictie pe features scalate."""

import numpy as np

from feature_engineering import engineer_features_from_input
from metadata_utils import get_features_all, get_model_type
from model_loader import load_artifacts


def test_load_artifacts_from_joblib(models_dir, metadata):
    model, scaler, loaded_meta = load_artifacts(models_dir)

    assert get_model_type(loaded_meta) == "RandomForestRegressor"
    assert len(get_features_all(loaded_meta)) == 20
    assert hasattr(model, "predict")
    assert hasattr(scaler, "transform")


def test_model_predict_positive_output(models_dir, metadata, sample_payload):
    model, scaler, _ = load_artifacts(models_dir)

    features_df = engineer_features_from_input(
        sample_payload,
        features_all=get_features_all(metadata),
    )
    scaled = scaler.transform(features_df)
    prediction = float(model.predict(scaled)[0])

    assert np.isfinite(prediction)
    assert prediction >= 0
