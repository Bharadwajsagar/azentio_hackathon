import math

import pandas as pd
from services.cleaning_service import (
    clean_amount,
    clean_transactions,
    normalize_boolean,
    normalize_category,
    parse_timestamp,
)


class TestCleanAmount:
    def test_plain_float_string(self):
        assert clean_amount("1234.5") == 1234.5

    def test_thousands_comma(self):
        assert clean_amount("41,677.41") == 41677.41

    def test_currency_prefix(self):
        assert clean_amount("INR 62146.26") == 62146.26

    def test_currency_prefix_with_comma(self):
        assert clean_amount("INR 1,234.56") == 1234.56

    def test_blank_is_nan(self):
        assert math.isnan(clean_amount(None))
        assert math.isnan(clean_amount(float("nan")))

    def test_garbage_is_nan(self):
        assert math.isnan(clean_amount("not_a_number"))

    def test_numeric_passthrough(self):
        assert clean_amount(99.9) == 99.9


class TestParseTimestamp:
    def test_iso_format(self):
        ts = parse_timestamp("2026-08-13 06:18:18")
        assert ts == pd.Timestamp("2026-08-13 06:18:18")

    def test_day_first_slash_format(self):
        ts = parse_timestamp("29/07/2026 18:07")
        assert ts == pd.Timestamp("2026-07-29 18:07")

    def test_not_available_is_nat(self):
        assert pd.isna(parse_timestamp("NOT_AVAILABLE"))

    def test_blank_is_nat(self):
        assert pd.isna(parse_timestamp(""))
        assert pd.isna(parse_timestamp(None))


class TestNormalizeCategory:
    def test_unifies_space_and_underscore(self):
        assert normalize_category("INTERNET_BANKING") == normalize_category("INTERNET BANKING")

    def test_strips_whitespace_and_uppercases(self):
        assert normalize_category("  mobile_app  ") == "MOBILE_APP"

    def test_blank_and_none_are_none(self):
        assert normalize_category("") is None
        assert normalize_category(None) is None


class TestNormalizeBoolean:
    def test_known_encodings(self):
        for truthy in ["1", "true", "TRUE", "yes", "Y", "y"]:
            assert normalize_boolean(truthy) is True
        for falsy in ["0", "false", "FALSE", "no", "N", "n"]:
            assert normalize_boolean(falsy) is False

    def test_unmapped_value_is_none(self):
        assert normalize_boolean("maybe") is None

    def test_nan_is_none(self):
        assert normalize_boolean(float("nan")) is None


class TestCleanTransactions:
    def test_adds_expected_columns(self, raw_transactions_df):
        cleaned = clean_transactions(raw_transactions_df)
        for col in ["amount_clean", "transaction_timestamp_clean", "is_timestamp_suspect"]:
            assert col in cleaned.columns

    def test_amount_defects_from_fixture(self, raw_transactions_df):
        cleaned = clean_transactions(raw_transactions_df)
        by_id = cleaned.set_index("transaction_id")
        assert by_id.loc["TXN_A", "amount_clean"] == 1234.56
        assert by_id.loc["TXN_B", "amount_clean"] == 500.00
        assert math.isnan(by_id.loc["TXN_C", "amount_clean"])
        assert math.isnan(by_id.loc["TXN_D", "amount_clean"])

    def test_timestamp_suspect_flags(self, raw_transactions_df):
        cleaned = clean_transactions(raw_transactions_df)
        by_id = cleaned.set_index("transaction_id")
        assert by_id.loc["TXN_A", "is_timestamp_suspect"] == False  # noqa: E712
        assert by_id.loc["TXN_C", "is_timestamp_suspect"] == True  # noqa: E712 (NOT_AVAILABLE)
        assert by_id.loc["TXN_D", "is_timestamp_suspect"] == True  # noqa: E712 (blank)

    def test_channel_separator_unification(self, raw_transactions_df):
        cleaned = clean_transactions(raw_transactions_df)
        by_id = cleaned.set_index("transaction_id")
        assert by_id.loc["TXN_A", "channel"] == by_id.loc["TXN_B", "channel"]
