"""Small synthetic fixtures reproducing the real defect patterns found in the
supplied CSVs, so tests run fast and don't depend on data/raw/*.csv existing.
"""

import pandas as pd
import pytest


@pytest.fixture
def raw_transactions_df():
    return pd.DataFrame(
        [
            {  # ordinary row, ISO timestamp, underscore-separated channel
                "transaction_id": "TXN_A",
                "account_id": "ACC_1",
                "customer_id": "CUST_1",
                "transaction_timestamp": "2026-01-05 10:00:00",
                "transaction_hour": 10,
                "amount": "1,234.56",
                "channel": "INTERNET_BANKING",
                "transaction_type": "PURCHASE",
                "status": "SUCCESS",
                "merchant_category": "GROCERY",
                "merchant_city": "Pune",
                "merchant_country": "IN",
                "device_type": "Android",
                "auth_method": "PIN",
                "is_foreign_transaction": "Y",
                "is_card_present": "1",
                "is_new_device": "0",
                "is_weekend": "0",
                "distance_from_home_km": 5.0,
                "txn_count_last_24h": 1,
                "amount_to_account_avg_ratio": 1.1,
            },
            {  # currency-code-prefixed amount + DD/MM/YYYY timestamp + space-separated channel
                "transaction_id": "TXN_B",
                "account_id": "ACC_1",
                "customer_id": "CUST_1",
                "transaction_timestamp": "05/01/2026 22:15",
                "transaction_hour": 22,
                "amount": "INR 500.00",
                "channel": "INTERNET BANKING",
                "transaction_type": "PAYMENT",
                "status": "success",
                "merchant_category": "utilities",
                "merchant_city": "Pune",
                "merchant_country": "IN",
                "device_type": "android",
                "auth_method": "OTP",
                "is_foreign_transaction": "no",
                "is_card_present": "0",
                "is_new_device": "1",
                "is_weekend": "0",
                "distance_from_home_km": 3.0,
                "txn_count_last_24h": 5,
                "amount_to_account_avg_ratio": 0.5,
            },
            {  # missing customer_id (recoverable via accounts join), blank amount, NOT_AVAILABLE timestamp
                "transaction_id": "TXN_C",
                "account_id": "ACC_2",
                "customer_id": None,
                "transaction_timestamp": "NOT_AVAILABLE",
                "transaction_hour": 3,
                "amount": None,
                "channel": " mobile_app ",
                "transaction_type": "TRANSFER",
                "status": "SUCCESS",
                "merchant_category": "FUND_TRANSFER",
                "merchant_city": "Pune",
                "merchant_country": "IN",
                "device_type": "iOS",
                "auth_method": "BIOMETRIC",
                "is_foreign_transaction": "1",
                "is_card_present": "0",
                "is_new_device": "0",
                "is_weekend": "1",
                "distance_from_home_km": 700.0,
                "txn_count_last_24h": 6,
                "amount_to_account_avg_ratio": 6.0,
            },
            {  # orphaned account reference (unrecoverable) + garbage amount + blank timestamp
                "transaction_id": "TXN_D",
                "account_id": "ACC_999",
                "customer_id": None,
                "transaction_timestamp": "",
                "transaction_hour": 14,
                "amount": "not_a_number",
                "channel": "POS",
                "transaction_type": "PURCHASE",
                "status": "FAILED",
                "merchant_category": "JEWELLERY",
                "merchant_city": "Pune",
                "merchant_country": "IN",
                "device_type": "POS Terminal",
                "auth_method": "NONE",
                "is_foreign_transaction": "0",
                "is_card_present": "1",
                "is_new_device": "0",
                "is_weekend": "0",
                "distance_from_home_km": 1.0,
                "txn_count_last_24h": 0,
                "amount_to_account_avg_ratio": 0.1,
            },
        ]
    )


@pytest.fixture
def raw_accounts_df():
    return pd.DataFrame(
        [
            {
                "account_id": "ACC_1",
                "customer_id": "CUST_1",
                "account_type": "SAVINGS",
                "account_status": "ACTIVE",
                "card_type": "CLASSIC",
                "account_tier": "SILVER",
            },
            {  # genuine null account_type in a VALID account -- must not be confused with a broken join
                "account_id": "ACC_2",
                "customer_id": "CUST_2",
                "account_type": None,
                "account_status": "ACTIVE",
                "card_type": None,
                "account_tier": "BASIC",
            },
        ]
    )


@pytest.fixture
def raw_customers_df():
    return pd.DataFrame(
        [
            {
                "customer_id": "CUST_1",
                "risk_rating": "LOW",
                "is_politically_exposed": 0,
                "gender": "M",
                "kyc_status": "VERIFIED",
                "employment_status": "EMPLOYED",
                "marital_status": "Single",
            },
            {
                "customer_id": "CUST_2",
                "risk_rating": "HIGH",
                "is_politically_exposed": 1,
                "gender": "F",
                "kyc_status": "VERIFIED",
                "employment_status": "EMPLOYED",
                "marital_status": "Married",
            },
        ]
    )


@pytest.fixture
def merged_df(raw_transactions_df, raw_accounts_df, raw_customers_df):
    from services.cleaning_service import clean_accounts, clean_customers, clean_transactions
    from services.merge_service import merge_transactions

    tx = clean_transactions(raw_transactions_df)
    acc = clean_accounts(raw_accounts_df)
    cust = clean_customers(raw_customers_df)
    merged, _ = merge_transactions(tx, acc, cust)
    return merged
