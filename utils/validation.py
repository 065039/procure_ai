import pandas as pd

NUMERIC_RANGES = {
    "price_lakhs": (0, None),
    "quality_score": (0, 10),
    "lead_time_days": (0, None),
    "on_time_delivery_pct": (0, 100),
    "reliability_score": (0, 10),
    "defect_rate_pct": (0, 100),
    "sustainability_score": (0, 10),
    "years_in_business": (0, None),
}


def validate_dataset(df: pd.DataFrame, required_columns: list[str]) -> dict:
    """Validate structure, missingness and basic numeric ranges."""
    result = {"valid": True, "errors": [], "warnings": []}

    if df is None or df.empty:
        return {"valid": False, "errors": ["The dataset is empty."], "warnings": []}

    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        result["valid"] = False
        result["errors"].append(
            "Missing required columns: " + ", ".join(missing_columns)
        )
        return result

    if df["vendor_id"].isna().any() or df["vendor_name"].isna().any():
        result["valid"] = False
        result["errors"].append("vendor_id and vendor_name cannot contain missing values.")

    duplicate_ids = df["vendor_id"].duplicated().sum()
    if duplicate_ids:
        result["valid"] = False
        result["errors"].append(
            f"Found {duplicate_ids} duplicate vendor_id value(s). Vendor IDs must be unique."
        )

    for col, (lower, upper) in NUMERIC_RANGES.items():
        converted = pd.to_numeric(df[col], errors="coerce")
        invalid_type_count = converted.isna().sum() - df[col].isna().sum()
        if invalid_type_count > 0:
            result["valid"] = False
            result["errors"].append(
                f"{col} contains {invalid_type_count} non-numeric value(s)."
            )

        if lower is not None:
            count = (converted.dropna() < lower).sum()
            if count:
                result["valid"] = False
                result["errors"].append(
                    f"{col} contains {count} value(s) below the allowed minimum of {lower}."
                )

        if upper is not None:
            count = (converted.dropna() > upper).sum()
            if count:
                result["valid"] = False
                result["errors"].append(
                    f"{col} contains {count} value(s) above the allowed maximum of {upper}."
                )

    missing_required = int(df[required_columns].isna().sum().sum())
    if missing_required:
        result["warnings"].append(
            f"The dataset contains {missing_required} missing value(s) in required fields. "
            "Rows with missing scoring fields will be excluded from ranking."
        )

    if len(df) < 3:
        result["warnings"].append(
            "Fewer than 3 vendors are present; ranking comparisons may be less informative."
        )

    return result
