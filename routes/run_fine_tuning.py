"""Entry point: run only the data pipeline + LoRA fine-tuning step, useful
for retraining the adapter independently of a full inference run.

    python -m routes.run_fine_tuning
"""

import argparse

from controllers.data_pipeline_controller import run_data_pipeline
from controllers.training_controller import run_fine_tuning
from services.inference_service import load_model_and_tokenizer
from utils.logging import get_logger
from utils.seed import set_global_seed

logger = get_logger(__name__)


def main(training_sample_size: int = 80, num_train_epochs: int = 1):
    set_global_seed()
    merged, _, _ = run_data_pipeline()
    model, tokenizer = load_model_and_tokenizer()
    _, adapter_path = run_fine_tuning(
        model,
        tokenizer,
        merged,
        max_examples=training_sample_size,
        num_train_epochs=num_train_epochs,
    )
    logger.info("Fine-tuning complete. Adapter saved to: %s", adapter_path)
    return adapter_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-sample-size", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    main(args.training_sample_size, args.epochs)
