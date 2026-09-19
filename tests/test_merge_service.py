import pandas as pd
from services.cleaning_service import clean_accounts, clean_customers, clean_transactions
from services.merge_service import merge_transactions


def test_customer_id_recovered_when_account_exists(
    raw_transactions_df, raw_accounts_df, raw_customers_df
):
    tx = clean_transactions(raw_transactions_df)
    acc = clean_accounts(raw_accounts_df)
    cust = clean_customers(raw_customers_df)
    merged, stats = merge_transactions(tx, acc, cust)

    row_c = merged.set_index("transaction_id").loc["TXN_C"]
    assert row_c["customer_id_resolved"] == "CUST_2"
    assert stats["customer_id_recovered_count"] == 1


def test_orphan_account_reference_flagged_not_recovered(
    raw_transactions_df, raw_accounts_df, raw_customers_df
):
    tx = clean_transactions(raw_transactions_df)
    acc = clean_accounts(raw_accounts_df)
    cust = clean_customers(raw_customers_df)
    merged, stats = merge_transactions(tx, acc, cust)

    row_d = merged.set_index("transaction_id").loc["TXN_D"]
    assert bool(row_d["account_reference_broken"]) is True
    assert pd.isna(row_d["customer_id_resolved"])
    assert stats["account_reference_broken_count"] == 1


def test_null_account_type_on_a_valid_account_is_not_a_broken_reference(
    raw_transactions_df, raw_accounts_df, raw_customers_df
):
    """Regression test for the bug found during development: using
    account_type.isna() as the 'broken join' signal overcounts, because
    accounts.csv can have genuine nulls in account_type independent of
    whether the join matched at all. ACC_2 has account_type=None but DOES
    exist, so transactions against it must NOT be flagged as broken."""
    tx = clean_transactions(raw_transactions_df)
    acc = clean_accounts(raw_accounts_df)
    cust = clean_customers(raw_customers_df)
    merged, _ = merge_transactions(tx, acc, cust)

    row_c = merged.set_index("transaction_id").loc["TXN_C"]  # account_id=ACC_2, account_type=None
    assert bool(row_c["account_reference_broken"]) is False


def test_valid_accounts_not_flagged_broken(raw_transactions_df, raw_accounts_df, raw_customers_df):
    tx = clean_transactions(raw_transactions_df)
    acc = clean_accounts(raw_accounts_df)
    cust = clean_customers(raw_customers_df)
    merged, _ = merge_transactions(tx, acc, cust)

    for txn_id in ["TXN_A", "TXN_B"]:
        row = merged.set_index("transaction_id").loc[txn_id]
        assert bool(row["account_reference_broken"]) is False
