"""Load the SLM (<3B params) and run inference with retry + rule-based fallback.

Model: Qwen/Qwen2.5-1.5B-Instruct, loaded in 4-bit (bitsandbytes NF4) so it
fits comfortably on a 4GB laptop GPU (~1.1-1.9GB VRAM observed). Falls back to
fp16/CPU automatically if no CUDA device is available.

Inference is BATCHED: measured serial (one prompt at a time) generation took
15-29s/transaction, which would take 4-8 hours for 1000 rows. Batching 16-32
prompts together dropped that to ~1-2s/transaction (same model, same prompt,
same validation rules -- purely a throughput fix, not a behavior change).
"""

import time

import torch
from models.prediction import PredictionOutput
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils.logging import get_logger

from services.prompt_service import build_prompt
from services.validation_service import fallback_prediction, parse_model_output

logger = get_logger(__name__)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 100
DEFAULT_BATCH_SIZE = 16


def load_model_and_tokenizer(model_name: str = MODEL_NAME, adapter_path: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = (
        "left"  # required for correct batched generation with a decoder-only model
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        )
    else:
        logger.warning("No CUDA device found -- loading in fp32 on CPU (will be slow).")
        model = AutoModelForCausalLM.from_pretrained(model_name)

    if adapter_path:
        from peft import PeftModel

        logger.info("Loading LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def _build_chat_text(tokenizer, record) -> str:
    system_prompt, user_msg = build_prompt(record)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _generate_batch(model, tokenizer, texts: list[str], max_new_tokens: int) -> list[str]:
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_only = out[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(gen_only, skip_special_tokens=True)


def run_inference(
    model,
    tokenizer,
    records,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_new_tokens: int = MAX_NEW_TOKENS,
):
    """Same business logic as single-record inference (validate -> retry once
    on failure -> rule-based fallback), executed batch-by-batch for speed.
    Returns (predictions, audit_meta) in the same order as `records`.
    """
    n = len(records)
    predictions: list[PredictionOutput | None] = [None] * n
    audit_meta = [None] * n
    t0 = time.time()

    for start in range(0, n, batch_size):
        batch_records = records[start : start + batch_size]
        texts = [_build_chat_text(tokenizer, r) for r in batch_records]
        raw_outputs = _generate_batch(model, tokenizer, texts, max_new_tokens)

        retry_indices, retry_records, retry_texts = [], [], []
        for i, (record, raw) in enumerate(zip(batch_records, raw_outputs)):
            pred, err = parse_model_output(raw, record.transaction_id)
            idx = start + i
            if pred is not None:
                predictions[idx] = pred
                audit_meta[idx] = {
                    "json_valid": True,
                    "validation_error": None,
                    "fallback_used": False,
                }
            else:
                retry_indices.append(idx)
                retry_records.append(record)
                system_prompt, user_msg = build_prompt(record)
                retry_msg = user_msg + (
                    "\n\nYour previous answer was invalid. Respond with ONLY the JSON object, "
                    "no other text, matching exactly the required keys."
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": retry_msg},
                ]
                retry_texts.append(
                    tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                )

        if retry_texts:
            retry_outputs = _generate_batch(model, tokenizer, retry_texts, max_new_tokens)
            for idx, record, raw2 in zip(retry_indices, retry_records, retry_outputs):
                pred2, err2 = parse_model_output(raw2, record.transaction_id)
                if pred2 is not None:
                    predictions[idx] = pred2
                    audit_meta[idx] = {
                        "json_valid": True,
                        "validation_error": None,
                        "fallback_used": False,
                    }
                else:
                    predictions[idx] = fallback_prediction(
                        record.transaction_id, record.weak_risk_score
                    )
                    audit_meta[idx] = {
                        "json_valid": False,
                        "validation_error": err2,
                        "fallback_used": True,
                    }

        done = min(start + batch_size, n)
        elapsed = time.time() - t0
        logger.info(
            "run_inference: %d/%d done (%.1fs elapsed, %.2fs/txn)", done, n, elapsed, elapsed / done
        )

    return predictions, audit_meta
