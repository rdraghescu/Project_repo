"""
Helpers pentru citirea metadata JSON (suporta schema veche si cea noua).
"""

from __future__ import annotations

from typing import Any


def get_features_all(metadata: dict[str, Any]) -> list[str]:
    """Returneaza lista completa de features, indiferent de structura JSON."""
    if "features_all" in metadata:
        return metadata["features_all"]
    features = metadata.get("features")
    if isinstance(features, dict) and "features_all" in features:
        return features["features_all"]
    raise KeyError("features_all not found in metadata")


def get_model_type(metadata: dict[str, Any]) -> str:
    if "model_type" in metadata:
        return metadata["model_type"]
    return metadata.get("model_info", {}).get("model_type", "RandomForestRegressor")


def get_model_version(metadata: dict[str, Any]) -> str:
    if "model_version" in metadata:
        return metadata["model_version"]
    model_info = metadata.get("model_info", {})
    return model_info.get("version", model_info.get("model_version", "2.0_Enhanced_20features_BEST"))


def get_features_total(metadata: dict[str, Any]) -> int:
    if "features_total" in metadata:
        return int(metadata["features_total"])
    features = metadata.get("features")
    if isinstance(features, dict) and "n_features" in features:
        return int(features["n_features"])
    return len(get_features_all(metadata))


def get_pageviews_median(metadata: dict[str, Any], default: float = 4.0) -> float:
    stats = metadata.get("feature_statistics")
    if isinstance(stats, dict) and "pageviews" in stats:
        return float(stats["pageviews"]["median"])
    return default


def get_hyperparameters(metadata: dict[str, Any]) -> dict[str, Any]:
    return metadata.get("hyperparameters", {})


def get_performance(metadata: dict[str, Any]) -> dict[str, Any]:
    return metadata.get("performance", {})


def country_column_prefix(features_all: list[str]) -> str:
    if any(name.startswith("country_grouped_") for name in features_all):
        return "country_grouped_"
    return "country_"
