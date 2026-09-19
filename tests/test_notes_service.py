from services.notes_service import synthesize_transaction_notes


def test_deterministic_with_same_seed(merged_df):
    out1 = synthesize_transaction_notes(merged_df, seed=7, injection_rate=0.5)
    out2 = synthesize_transaction_notes(merged_df, seed=7, injection_rate=0.5)
    assert (
        out1["transaction_notes__SYNTHETIC"].tolist()
        == out2["transaction_notes__SYNTHETIC"].tolist()
    )
    assert out1["_gt_is_injection_attempt"].tolist() == out2["_gt_is_injection_attempt"].tolist()


def test_injection_rate_is_honored(merged_df):
    out = synthesize_transaction_notes(merged_df, seed=3, injection_rate=0.5)
    n = len(out)
    expected = round(n * 0.5)
    assert out["_gt_is_injection_attempt"].sum() == expected


def test_marker_columns_present(merged_df):
    out = synthesize_transaction_notes(merged_df)
    assert "transaction_notes__SYNTHETIC" in out.columns
    assert "_gt_is_injection_attempt" in out.columns
