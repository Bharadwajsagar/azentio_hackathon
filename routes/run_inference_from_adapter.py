"""Entry point: resume from an already-trained LoRA adapter and run inference
directly, skipping the (potentially lengthy) fine-tuning step -- useful after
an interrupted run, or to re-run inference with a fix that doesn't affect
training (e.g. the KV-cache regression this route was added to recover from).

    python -m routes.run_inference_from_adapter
"""

import argparse

from controllers.data_pipeline_controller import run_data_pipeline
from controllers.inference_controller import merge_audit, run_predictions
from controllers.training_controller import DEFAULT_ADAPTER_DIR
from services.inference_service import load_model_and_tokenizer
from utils.io import OUTPUTS_DIR, save_json
from utils.logging import get_logger
from utils.seed import set_global_seed

logger = get_logger(__name__)


def main(adapter_dir: str = str(DEFAULT_ADAPTER_DIR), inference_batch_size: int = 16):
    set_global_seed()

    logger.info("=== Data pipeline (clean, merge, notes, sanitize, features, weak labels) ===")
    merged, audit_records, quality_report = run_data_pipeline()

    logger.info("=== Loading base SLM + saved LoRA adapter from %s ===", adapter_dir)
    model, tokenizer = load_model_and_tokenizer(adapter_path=adapter_dir)

    logger.info("=== Fine-tuned inference on the FULL dataset (n=%d) ===", len(merged))
    final_preds, final_audit_meta = run_predictions(
        model, tokenizer, merged, batch_size=inference_batch_size
    )

    full_audit = merge_audit(audit_records, final_audit_meta, merged)
    predictions_json = [p.to_dict() for p in final_preds]

    save_json(predictions_json, OUTPUTS_DIR / "predictions.json")
    save_json([a.to_dict() for a in full_audit], OUTPUTS_DIR / "audit_log.json")
    save_json(
        {"data_quality_report": quality_report},
        OUTPUTS_DIR / "evaluation_report.json",
    )

    logger.info("Done. Wrote %d predictions to outputs/predictions.json", len(predictions_json))
    return predictions_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=str, default=str(DEFAULT_ADAPTER_DIR))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    main(args.adapter_dir, args.batch_size)
