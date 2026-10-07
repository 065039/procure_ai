from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "vendors.csv"


def load_data_source(uploaded_file=None, use_sample=False) -> pd.DataFrame:
    """Load the built-in sample dataset or an uploaded CSV/XLSX file."""
    if use_sample:
        return pd.read_csv(SAMPLE_PATH)

    if uploaded_file is None:
        raise ValueError("No file was provided.")

    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file type. Please upload CSV or XLSX.")
