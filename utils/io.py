"""Path constants and simple I/O helpers shared across the pipeline."""

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TRANSACTIONS_CSV = DATA_RAW_DIR / "transactions.csv"
ACCOUNTS_CSV = DATA_RAW_DIR / "accounts.csv"
CUSTOMERS_CSV = DATA_RAW_DIR / "customers.csv"


def load_raw_csvs():
    """Load the three raw CSVs exactly as provided (no cleaning here)."""
    tx = pd.read_csv(TRANSACTIONS_CSV)
    acc = pd.read_csv(ACCOUNTS_CSV)
    cust = pd.read_csv(CUSTOMERS_CSV)
    return tx, acc, cust


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def save_dataframe(df: pd.DataFrame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
