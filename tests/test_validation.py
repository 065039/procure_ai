import pandas as pd
from utils.validation import validate_dataset

REQUIRED = [
    "vendor_id", "vendor_name", "category", "price_lakhs",
    "quality_score", "lead_time_days", "on_time_delivery_pct",
    "reliability_score", "defect_rate_pct", "sustainability_score",
    "years_in_business", "esg_certified"
]


def valid_df():
    return pd.DataFrame({
        "vendor_id": ["A"],
        "vendor_name": ["A"],
        "category": ["X"],
        "price_lakhs": [4.0],
        "quality_score": [8.0],
        "lead_time_days": [7],
        "on_time_delivery_pct": [95],
        "reliability_score": [9.0],
        "defect_rate_pct": [1.0],
        "sustainability_score": [8.0],
        "years_in_business": [10],
        "esg_certified": ["Yes"],
    })


def test_valid_dataset():
    result = validate_dataset(valid_df(), REQUIRED)
    assert result["valid"] is True


def test_missing_column():
    df = valid_df().drop(columns=["quality_score"])
    result = validate_dataset(df, REQUIRED)
    assert result["valid"] is False
    assert any("quality_score" in e for e in result["errors"])


def test_negative_price():
    df = valid_df()
    df.loc[0, "price_lakhs"] = -1
    result = validate_dataset(df, REQUIRED)
    assert result["valid"] is False
