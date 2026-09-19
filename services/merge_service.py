"""Relational merge: transactions -> accounts -> customers.

Real defect found in testing: 8 transactions reference account_ids
(ACC_900000-ACC_900007) that do not exist anywhere in accounts.csv (real
accounts only run ACC_000001-ACC_000178). These are not recoverable by any
join -- they are fabricated/orphaned foreign keys. Rather than silently
dropping those rows (which would violate "one JSON output per transaction"),
we keep them and flag account_reference_broken=True, which is also treated as
a fraud signal downstream (a transaction against a nonexistent account is
itself suspicious).
"""

import pandas as pd
from utils.logging import get_logger

logger = get_logger(__name__)


def merge_transactions(tx: pd.DataFrame, acc: pd.DataFrame, cust: pd.DataFrame):
    acc_indexed = acc.set_index("account_id")
    merged = tx.merge(
        acc_indexed,
        left_on="account_id",
        right_index=True,
        how="left",
        suffixes=("", "_acc"),
        indicator="_acc_merge",
    )

    # account join failed => the referenced account_id doesn't exist at all.
    # NOTE: do not use e.g. account_type.isna() for this -- accounts.csv has
    # its own genuine nulls in account_type (6 rows) independent of whether
    # the join matched at all, which would overcount broken references.
    merged["account_reference_broken"] = merged["_acc_merge"] == "left_only"
    merged = merged.drop(columns=["_acc_merge"])

    # recover missing customer_id via the account -> customer_id link where possible
    customer_id_from_account = (
        merged["customer_id_acc"]
        if "customer_id_acc" in merged.columns
        else merged.get("customer_id")
    )
    n_missing_before = merged["customer_id"].isna().sum()
    merged["customer_id_resolved"] = merged["customer_id"].where(
        merged["customer_id"].notna(), customer_id_from_account
    )
    n_missing_after = merged["customer_id_resolved"].isna().sum()

    cust_indexed = cust.set_index("customer_id")
    merged = merged.merge(
        cust_indexed,
        left_on="customer_id_resolved",
        right_index=True,
        how="left",
        suffixes=("", "_cust"),
    )

    stats = {
        "rows_in": len(tx),
        "rows_out": len(merged),
        "account_reference_broken_count": int(merged["account_reference_broken"].sum()),
        "customer_id_missing_before_recovery": int(n_missing_before),
        "customer_id_missing_after_recovery": int(n_missing_after),
        "customer_id_recovered_count": int(n_missing_before - n_missing_after),
    }
    logger.info("merge_transactions: %s", stats)
    return merged, stats
