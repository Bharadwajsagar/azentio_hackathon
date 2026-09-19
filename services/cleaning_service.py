"""Column-level cleaning for transactions / accounts / customers.

Every rule here was validated against the real data in dev scratch scripts
before being ported in (see conversation history): amount parsing, timestamp
parsing, categorical normalization, and boolean normalization.
"""

import re

import numpy as np
import pandas as pd
from utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Amount cleaning
# ---------------------------------------------------------------------------
# Real defects found in transactions.csv: blank amounts, and 6 rows where a
# currency code leaked into the amount field (e.g. "INR 62146.26" instead of
# 62146.26), plus 11 rows where a quoted thousands-separator comma
# (e.g. "41,677.41") makes naive int(",") splitting look like an extra column.
_CURRENCY_PREFIX_RE = re.compile(r"^[A-Za-z]{2,4}\s+")


def clean_amount(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    s = _CURRENCY_PREFIX_RE.sub("", s)
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# Timestamp cleaning
# ---------------------------------------------------------------------------
# Real formats found: ISO "YYYY-MM-DD HH:MM:SS", "DD/MM/YYYY HH:MM" (day-first,
# consistent with the Indian-locale data elsewhere in the file), blank, and
# the literal string "NOT_AVAILABLE".
def parse_timestamp(x):
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    if s == "" or s.upper() == "NOT_AVAILABLE":
        return pd.NaT
    ts = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if pd.isna(ts):
        ts = pd.to_datetime(s, format="%d/%m/%Y %H:%M", errors="coerce")
    if pd.isna(ts):
        ts = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return ts


# ---------------------------------------------------------------------------
# Categorical normalization
# ---------------------------------------------------------------------------
# Real defects found: mixed case ("Pos"/"POS"/"pos terminal"), stray
# whitespace (" MOBILE_APP "), and -- found only once this was tested against
# real data -- the SAME category spelled with a space vs. an underscore
# ("INTERNET BANKING" vs "INTERNET_BANKING"), which naive strip+upper does not
# unify. Collapsing runs of whitespace/underscore to a single underscore fixes
# that without changing what "normalize categoricals" was meant to do.
def normalize_category(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "":
        return None
    s = re.sub(r"[\s_]+", "_", s)
    return s.upper()


TX_CATEGORICAL_COLUMNS = [
    "transaction_type",
    "channel",
    "status",
    "merchant_category",
    "merchant_city",
    "merchant_country",
    "device_type",
    "auth_method",
]

# ---------------------------------------------------------------------------
# Boolean normalization
# ---------------------------------------------------------------------------
# Real defect: the SAME boolean concept is encoded differently across columns
# and even within one column (0/1, TRUE/FALSE, yes/no, Y/N all appear).
BOOLEAN_MAP = {
    "true": True,
    "false": False,
    "yes": True,
    "no": False,
    "y": True,
    "n": False,
    "1": True,
    "0": False,
}


def normalize_boolean(x):
    if pd.isna(x):
        return None
    xs = str(x).strip().lower()
    return BOOLEAN_MAP.get(xs)


TX_BOOLEAN_COLUMNS = ["is_foreign_transaction", "is_card_present", "is_new_device", "is_weekend"]


def clean_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx.copy()
    tx["amount_clean"] = tx["amount"].apply(clean_amount)
    tx["transaction_timestamp_clean"] = tx["transaction_timestamp"].apply(parse_timestamp)
    tx["is_timestamp_suspect"] = tx["transaction_timestamp_clean"].isna()

    for c in TX_CATEGORICAL_COLUMNS:
        tx[c] = tx[c].apply(normalize_category)

    for c in TX_BOOLEAN_COLUMNS:
        tx[c + "_norm"] = tx[c].apply(normalize_boolean)

    n_bad_amount = tx["amount_clean"].isna().sum()
    n_bad_ts = tx["is_timestamp_suspect"].sum()
    logger.info(
        "clean_transactions: %d/%d rows have unparseable amount, "
        "%d/%d rows have a suspect/missing timestamp",
        n_bad_amount,
        len(tx),
        n_bad_ts,
        len(tx),
    )
    return tx


ACC_CATEGORICAL_COLUMNS = ["account_type", "account_status", "card_type", "account_tier"]


def clean_accounts(acc: pd.DataFrame) -> pd.DataFrame:
    acc = acc.copy()
    for c in ACC_CATEGORICAL_COLUMNS:
        acc[c] = acc[c].apply(normalize_category)
    return acc


CUST_CATEGORICAL_COLUMNS = [
    "gender",
    "kyc_status",
    "risk_rating",
    "employment_status",
    "marital_status",
]


def clean_customers(cust: pd.DataFrame) -> pd.DataFrame:
    cust = cust.copy()
    for c in CUST_CATEGORICAL_COLUMNS:
        cust[c] = cust[c].apply(normalize_category)
    return cust
