"""LoRA fine-tuning of the SLM on heuristic pseudo-labels (weak supervision).

IMPORTANT: the training targets here come from services/labeling_service.py's
_pseudo_is_fraud, which is a deterministic rule-based heuristic, NOT an
authoritative fraud label. No ground-truth fraud data was supplied. This
fine-tune's purpose is to teach the tiny model to (a) reliably emit the exact
required JSON schema and (b) align its judgment with the structured evidence
rules, not to claim a verified accuracy improvement against real fraud.
"""

import json

import torch
from datasets import Dataset
from models.transaction import TransactionRecord
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from utils.logging import get_logger

from services.prompt_service import build_prompt

logger = get_logger(__name__)

LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)


def _weak_label_to_completion(row) -> str:
    """Build the target completion from the heuristic pseudo-label -- explicitly
    weak supervision, never presented as ground truth."""
    is_fraud = bool(row["_pseudo_is_fraud"])
    score = int(row["_weak_risk_score"])
    confidence = min(0.6 + 0.08 * score, 0.97) if is_fraud else max(0.9 - 0.08 * score, 0.55)
    reasons = []
    if row.get("is_high_amount_ratio"):
        reasons.append("amount far above the account's average")
    if row.get("is_odd_hour"):
        reasons.append("occurred at an unusual hour")
    if row.get("is_high_velocity"):
        reasons.append("high transaction velocity in the last 24h")
    if row.get("is_far_from_home"):
        reasons.append("far from the customer's home location")
    if row.get("is_new_device_flag"):
        reasons.append("initiated from a new device")
    if row.get("is_high_risk_customer"):
        reasons.append("customer has a high risk rating")
    if row.get("is_pep"):
        reasons.append("customer is politically exposed")
    if row.get("account_reference_broken"):
        reasons.append("references an account ID that does not exist in our records")

    justification = (
        ("Flagged due to " + ", ".join(reasons) + ".")
        if (is_fraud and reasons)
        else (
            "No significant risk signals were present in the transaction evidence."
            if not is_fraud
            else "Multiple weak risk signals combined to exceed the review threshold."
        )
    )
    obj = {
        "transaction_id": row["transaction_id"],
        "is_fraud": is_fraud,
        "confidence": round(confidence, 2),
        "justification": justification,
    }
    return json.dumps(obj)


def build_training_examples(df):
    """Returns a list of {"text": full_chat_formatted_string} ready for the tokenizer."""
    examples = []
    for _, row in df.iterrows():
        record = TransactionRecord.from_row(row)
        system_prompt, user_msg = build_prompt(record)
        completion = _weak_label_to_completion(row)
        examples.append({"system": system_prompt, "user": user_msg, "completion": completion})
    logger.info(
        "build_training_examples: built %d examples from heuristic pseudo-labels", len(examples)
    )
    return examples


def fine_tune_lora(
    model,
    tokenizer,
    training_examples,
    output_dir,
    num_train_epochs=1,
    per_device_batch_size=2,
    max_length=512,
):
    def format_example(ex):
        messages = [
            {"role": "system", "content": ex["system"]},
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": ex["completion"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    ds = Dataset.from_list([format_example(ex) for ex in training_examples])

    def tokenize_fn(batch):
        # Truncate only -- no fixed-length padding here. The data collator
        # below pads dynamically to the longest sequence in each batch instead,
        # which also lets it correctly mask padding out of the loss (-100)
        # instead of training on padding tokens as if they were real targets.
        # (Measured: on this hardware most prompts are already close to
        # max_length, so this did not meaningfully change step time -- the
        # real bottleneck is backward-pass compute on a 4GB GPU. It's kept
        # because it fixes the loss-masking correctness issue regardless.)
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)

    # Standard QLoRA setup: casts norms to fp32, enables input grads on the
    # frozen 4-bit base, and turns on gradient checkpointing -- needed for
    # headroom on a 4GB GPU (measured peak ~3.9/4.0GB without this).
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    peft_model = get_peft_model(model, LORA_CONFIG)
    peft_model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )

    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("fine_tune_lora: adapter saved to %s", output_dir)
    return peft_model
