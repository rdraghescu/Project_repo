"""
Transformare input API -> matrice cu cele 20 features ENHANCED
folosite de Random Forest (aceeasi ordine ca in model_metadata_v2.json).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from metadata_utils import country_column_prefix

# Tarile din metadata (One-Hot, fara drop_first pe country -> 10 coloane)
TOP_COUNTRIES = [
    "Brazil",
    "Canada",
    "France",
    "Germany",
    "India",
    "Japan",
    "Other",
    "United Kingdom",
    "United States",
    "Vietnam",
]


def engineer_features_from_input(
    data: dict[str, Any],
    features_all: list[str],
    pageviews_median: float = 4.0,
) -> pd.DataFrame:
    """
    Construieste un rand cu exact coloanele asteptate de model.

    Input minim: pageviews, visitNumber, hits, device_category, country
    (formulele sunt aliniate cu mlflow_ml_model_web_traffic.py).
    """
    pageviews = float(data["pageviews"])
    visit_number = float(data["visitNumber"])
    hits = float(data["hits"])
    device_category = str(data.get("device_category", "desktop")).lower()
    country = str(data.get("country", "Other"))

    # One-Hot device (desktop = referinta -> ambele 0)
    device_mobile = 1 if device_category == "mobile" else 0
    device_tablet = 1 if device_category == "tablet" else 0

    # One-Hot country (grupare: tari din top sau Other)
    country_grouped = country if country in TOP_COUNTRIES else "Other"
    country_prefix = country_column_prefix(features_all)
    country_flags = {
        f"{country_prefix}{name}": (1 if country_grouped == name else 0)
        for name in TOP_COUNTRIES
    }

    # Feature engineering (5 coloane derivate)
    engineered = {
        "pageviews_per_visit": pageviews / (visit_number + 1),
        "engagement_score": pageviews * hits,
        "hits_per_pageview": hits / (pageviews + 1),
        "high_pageviews": 1 if pageviews > pageviews_median else 0,
        "is_returning": 1 if visit_number > 1 else 0,
    }

    row = {
        "pageviews": pageviews,
        "visitNumber": visit_number,
        "hits": hits,
        "device_category_mobile": device_mobile,
        "device_category_tablet": device_tablet,
        **country_flags,# ** inseamna unpacking (desfasurarea) a dictionarului country_flags in linia de mai jos.
        **engineered,
    }

    # Reordonam strict dupa lista din metadata (ordinea conteaza la predict)
    frame = pd.DataFrame([row])
    return frame[features_all]
