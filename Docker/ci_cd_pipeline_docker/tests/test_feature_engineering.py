"""Teste pentru transformarea celor 20 features ENHANCED."""

from feature_engineering import engineer_features_from_input


def test_engineer_features_returns_20_columns(metadata, sample_payload):
    features_df = engineer_features_from_input(
        sample_payload,
        features_all=metadata["features_all"],
        pageviews_median=metadata["feature_statistics"]["pageviews"]["median"],
    )
    assert features_df.shape == (1, 20)
    assert list(features_df.columns) == metadata["features_all"]


def test_engineered_formulas(metadata, sample_payload):
    features_df = engineer_features_from_input(
        sample_payload,
        features_all=metadata["features_all"],
        pageviews_median=4.0,
    )
    row = features_df.iloc[0]

    assert row["pageviews_per_visit"] == sample_payload["pageviews"] / (
        sample_payload["visitNumber"] + 1
    )
    assert row["engagement_score"] == sample_payload["pageviews"] * sample_payload["hits"]
    assert row["is_returning"] == 1
    assert row["device_category_mobile"] == 0
    assert row["country_grouped_United States"] == 1


def test_unknown_country_maps_to_other(metadata):
    payload = {
        "pageviews": 3,
        "visitNumber": 1,
        "hits": 5,
        "device_category": "mobile",
        "country": "Romania",
    }
    features_df = engineer_features_from_input(
        payload,
        features_all=metadata["features_all"],
    )
    assert features_df.iloc[0]["country_grouped_Other"] == 1
    assert features_df.iloc[0]["device_category_mobile"] == 1
