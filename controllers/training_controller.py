"""Orchestrates LoRA fine-tuning on the heuristic pseudo-labels produced by
the data pipeline controller."""

from services.fine_tuning_service import build_training_examples, fine_tune_lora
from utils.io import PROJECT_ROOT
from utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "outputs" / "lora_adapter"


def run_fine_tuning(
    model,
    tokenizer,
    featured_df,
    output_dir=DEFAULT_ADAPTER_DIR,
    max_examples=None,
    num_train_epochs=1,
):
    df = featured_df
    if max_examples is not None:
        df = df.sample(n=min(max_examples, len(df)), random_state=42)

    logger.info(
        "run_fine_tuning: building training examples from %d rows "
        "(heuristic pseudo-labels -- weak supervision, not ground truth)",
        len(df),
    )
    examples = build_training_examples(df)
    peft_model = fine_tune_lora(
        model, tokenizer, examples, str(output_dir), num_train_epochs=num_train_epochs
    )
    return peft_model, str(output_dir)
