import pandas as pd
from utils.scoring import (
    normalize_higher_is_better,
    normalize_lower_is_better,
    evaluate_vendors,
)

WEIGHTS = {
    "Cost": 30,
    "Quality": 30,
    "Delivery": 20,
    "Reliability": 15,
    "Sustainability": 5,
}


def sample_df():
    return pd.DataFrame({
        "vendor_id": ["A", "B", "C"],
        "vendor_name": ["A", "B", "C"],
        "category": ["X", "X", "X"],
        "price_lakhs": [3.0, 5.0, 4.0],
        "quality_score": [8.0, 10.0, 9.0],
        "lead_time_days": [10, 5, 7],
        "on_time_delivery_pct": [90, 98, 95],
        "reliability_score": [8.0, 9.8, 9.0],
        "defect_rate_pct": [2.0, 0.8, 1.2],
        "sustainability_score": [7.0, 9.0, 8.0],
        "years_in_business": [10, 20, 15],
        "esg_certified": ["No", "Yes", "Yes"],
    })


def test_higher_normalization():
    s = pd.Series([1, 2, 3])
    result = normalize_higher_is_better(s)
    assert result.iloc[0] == 0
    assert result.iloc[-1] == 100


def test_lower_normalization():
    s = pd.Series([1, 2, 3])
    result = normalize_lower_is_better(s)
    assert result.iloc[0] == 100
    assert result.iloc[-1] == 0


def test_zero_variance_is_safe():
    s = pd.Series([5, 5, 5])
    result = normalize_higher_is_better(s)
    assert all(result == 100)


def test_evaluation_ranks_vendors():
    result = evaluate_vendors(sample_df(), WEIGHTS)
    assert len(result) == 3
    assert list(result["rank"]) == [1, 2, 3]
    assert result["overall_score"].between(0, 100).all()


def test_invalid_weights_raise():
    bad_weights = WEIGHTS.copy()
    bad_weights["Cost"] = 40
    try:
        evaluate_vendors(sample_df(), bad_weights)
        assert False
    except ValueError:
        assert True
