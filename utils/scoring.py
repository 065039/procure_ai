import numpy as np
import pandas as pd


def normalize_higher_is_better(series: pd.Series) -> pd.Series:
    """Normalize a numeric series to 0-100 where higher values are better."""
    s = pd.to_numeric(series, errors="coerce")
    min_v, max_v = s.min(), s.max()
    if pd.isna(min_v) or pd.isna(max_v):
        return pd.Series(np.nan, index=series.index)
    if max_v == min_v:
        return pd.Series(100.0, index=series.index)
    return ((s - min_v) / (max_v - min_v) * 100).clip(0, 100)


def normalize_lower_is_better(series: pd.Series) -> pd.Series:
    """Normalize a numeric series to 0-100 where lower values are better."""
    s = pd.to_numeric(series, errors="coerce")
    min_v, max_v = s.min(), s.max()
    if pd.isna(min_v) or pd.isna(max_v):
        return pd.Series(np.nan, index=series.index)
    if max_v == min_v:
        return pd.Series(100.0, index=series.index)
    return ((max_v - s) / (max_v - min_v) * 100).clip(0, 100)


def _prepare_scoring_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert required numeric fields and exclude rows missing scoring inputs."""
    result = df.copy()

    numeric_cols = [
        "price_lakhs", "quality_score", "lead_time_days",
        "on_time_delivery_pct", "reliability_score",
        "defect_rate_pct", "sustainability_score", "years_in_business"
    ]
    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    scoring_inputs = [
        "price_lakhs", "quality_score", "lead_time_days",
        "on_time_delivery_pct", "reliability_score",
        "sustainability_score"
    ]
    result = result.dropna(subset=scoring_inputs).copy()
    return result


def evaluate_vendors(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Calculate criterion scores, weighted contributions and ranking."""
    if abs(sum(weights.values()) - 100.0) >= 0.01:
        raise ValueError("Weights must sum to 100%.")

    result = _prepare_scoring_data(df)
    if result.empty:
        raise ValueError("No vendors remain after removing rows with missing scoring fields.")

    result["cost_score"] = normalize_lower_is_better(result["price_lakhs"])
    result["quality_score_norm"] = normalize_higher_is_better(result["quality_score"])

    lead_time_score = normalize_lower_is_better(result["lead_time_days"])
    on_time_score = normalize_higher_is_better(result["on_time_delivery_pct"])
    result["delivery_score"] = 0.4 * lead_time_score + 0.6 * on_time_score

    result["reliability_score_norm"] = normalize_higher_is_better(
        result["reliability_score"]
    )
    result["sustainability_score_norm"] = normalize_higher_is_better(
        result["sustainability_score"]
    )

    # Keep user-facing score names consistent.
    result["quality_score"] = result["quality_score_norm"]
    result["reliability_score"] = result["reliability_score_norm"]
    result["sustainability_score"] = result["sustainability_score_norm"]

    w = {
        "cost": weights["Cost"] / 100,
        "quality": weights["Quality"] / 100,
        "delivery": weights["Delivery"] / 100,
        "reliability": weights["Reliability"] / 100,
        "sustainability": weights["Sustainability"] / 100,
    }

    result["cost_contribution"] = result["cost_score"] * w["cost"]
    result["quality_contribution"] = result["quality_score"] * w["quality"]
    result["delivery_contribution"] = result["delivery_score"] * w["delivery"]
    result["reliability_contribution"] = result["reliability_score"] * w["reliability"]
    result["sustainability_contribution"] = result["sustainability_score"] * w["sustainability"]

    result["overall_score"] = (
        result["cost_contribution"]
        + result["quality_contribution"]
        + result["delivery_contribution"]
        + result["reliability_contribution"]
        + result["sustainability_contribution"]
    )

    result = result.sort_values(
        ["overall_score", "vendor_name"],
        ascending=[False, True]
    ).reset_index(drop=True)
    result["rank"] = result.index + 1

    return result


def compare_rankings(base: pd.DataFrame, scenario: pd.DataFrame) -> pd.DataFrame:
    """Compare ranks and scores between two evaluations."""
    left = base[["vendor_name", "rank", "overall_score"]].rename(
        columns={"rank": "current_rank", "overall_score": "current_score"}
    )
    right = scenario[["vendor_name", "rank", "overall_score"]].rename(
        columns={"rank": "scenario_rank", "overall_score": "scenario_score"}
    )
    comparison = left.merge(right, on="vendor_name", how="outer")
    comparison["current_rank"] = comparison["current_rank"].fillna(len(base) + 1)
    comparison["scenario_rank"] = comparison["scenario_rank"].fillna(len(scenario) + 1)
    comparison["rank_change"] = (
        comparison["current_rank"] - comparison["scenario_rank"]
    )
    comparison["rank_change_label"] = comparison["rank_change"].apply(
        lambda x: "↑" if x > 0 else ("↓" if x < 0 else "—")
    )
    return comparison.sort_values("scenario_rank").reset_index(drop=True)
