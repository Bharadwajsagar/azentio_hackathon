"""Entry point: run the entire Fraud Sentinel pipeline end-to-end.

    python -m routes.run_full_pipeline

Stages: clean -> merge -> synthesize notes -> sanitize -> features ->
weak pseudo-labels -> baseline zero-shot SLM eval (subset, for the
before/after story) -> LoRA fine-tune on pseudo-labels -> fine-tuned SLM
inference (full dataset, this is the actual required deliverable) ->
strict JSON validation -> export outputs/predictions.json,
outputs/audit_log.json, outputs/evaluation_report.json.

Runtime note (measured on an RTX 3050 4GB laptop GPU, batched generation):
~1-2s/transaction. Baseline runs on a stratified subset for speed; the final
predictions.json covers every transaction as required by the brief.
"""

import argparse

import pandas as pd
from controllers.data_pipeline_controller import run_data_pipeline
from controllers.inference_controller import (
    compare_against_pseudo_labels,
    merge_audit,
    run_predictions,
)
from controllers.training_controller import run_fine_tuning
from services.inference_service import load_model_and_tokenizer
from utils.io import OUTPUTS_DIR, save_json
from utils.logging import get_logger
from utils.seed import set_global_seed

logger = get_logger(__name__)


def stratified_baseline_sample(df: pd.DataFrame, n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Sample across risk levels + injection status so the baseline eval isn't
    accidentally all-benign or all-high-risk."""
    high_risk = df[df["_weak_risk_score"] >= 3]
    injected = df[df["injection_detected"]]
    benign = df[(df["_weak_risk_score"] == 0) & (~df["injection_detected"])]

    n_each = max(1, n // 3)
    parts = [
        high_risk.sample(n=min(n_each, len(high_risk)), random_state=seed),
        injected.sample(n=min(n_each, len(injected)), random_state=seed),
        benign.sample(n=min(n_each, len(benign)), random_state=seed),
    ]
    sample = pd.concat(parts).drop_duplicates(subset="transaction_id")
    return sample.head(n)


def main(
    baseline_sample_size: int = 100,
    training_sample_size: int = 80,
    num_train_epochs: int = 1,
    inference_batch_size: int = 16,
):
    set_global_seed()

    logger.info(
        "=== Stage 1-7: data pipeline (clean, merge, notes, sanitize, features, weak labels) ==="
    )
    merged, audit_records, quality_report = run_data_pipeline()

    logger.info("=== Stage 8: load base SLM ===")
    model, tokenizer = load_model_and_tokenizer()

    logger.info(
        "=== Stage 9-10: baseline zero-shot inference (stratified subset, n=%d) ===",
        baseline_sample_size,
    )
    baseline_df = stratified_baseline_sample(merged, n=baseline_sample_size)
    baseline_preds, baseline_audit_meta = run_predictions(
        model, tokenizer, baseline_df, batch_size=inference_batch_size
    )
    baseline_stats = compare_against_pseudo_labels(
        baseline_preds, baseline_df, label="baseline_zero_shot"
    )

    logger.info("=== Stage 11-12: LoRA fine-tune on heuristic pseudo-labels (weak supervision) ===")
    peft_model, adapter_path = run_fine_tuning(
        model,
        tokenizer,
        merged,
        max_examples=training_sample_size,
        num_train_epochs=num_train_epochs,
    )

    logger.info("=== Stage 13: fine-tuned inference on the FULL dataset (n=%d) ===", len(merged))
    final_preds, final_audit_meta = run_predictions(
        peft_model, tokenizer, merged, batch_size=inference_batch_size
    )
    fine_tuned_stats = compare_against_pseudo_labels(final_preds, merged, label="fine_tuned")

    logger.info("=== Stage 16: baseline vs fine-tuned comparison ===")
    comparison = {"baseline_zero_shot": baseline_stats, "fine_tuned": fine_tuned_stats}
    logger.info("comparison: %s", comparison)

    logger.info("=== Stage 17-18: JSON validation (already enforced inline) + export ===")
    full_audit = merge_audit(audit_records, final_audit_meta, merged)

    predictions_json = [p.to_dict() for p in final_preds]
    save_json(predictions_json, OUTPUTS_DIR / "predictions.json")
    save_json([a.to_dict() for a in full_audit], OUTPUTS_DIR / "audit_log.json")
    save_json(
        {
            "data_quality_report": quality_report,
            "baseline_vs_fine_tuned": comparison,
            "note": (
                "weak_pseudo_label_positive_rate and *_pseudo_labels agreement figures are "
                "WEAK SUPERVISION heuristics, not real fraud ground truth -- no authoritative "
                "fraud labels were provided in the supplied data."
            ),
        },
        OUTPUTS_DIR / "evaluation_report.json",
    )

    logger.info("Done. Wrote %d predictions to outputs/predictions.json", len(predictions_json))
    return predictions_json, full_audit, quality_report, comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-sample-size", type=int, default=100)
    parser.add_argument("--training-sample-size", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    main(args.baseline_sample_size, args.training_sample_size, args.epochs, args.batch_size)
