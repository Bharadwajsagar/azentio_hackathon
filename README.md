# Fraud Sentinel

End-to-end pipeline that cleans and merges three messy relational banking
CSVs, neutralizes prompt-injection attempts hidden in transaction notes, and
uses a Small Language Model (<3B params, Qwen2.5-1.5B-Instruct) to produce a
strict-JSON fraud risk profile per transaction, plus a LoRA fine-tune step.

## Structure

```
models/          data schemas (dataclasses) -- NOT the SLM itself
  transaction.py   TransactionRecord: one fully cleaned/merged/featured row
  prediction.py    PredictionOutput: the required 4-field output contract
  audit.py         AuditRecord: injection/validation audit trail (kept out of predictions.json)

services/        single-responsibility business logic
  data_loader.py         load the 3 raw CSVs
  cleaning_service.py    amount/timestamp/categorical/boolean cleaning
  merge_service.py       transactions -> accounts -> customers join, orphan-account detection
  notes_service.py       synthesizes transaction_notes (see "Data decisions" below)
  sanitization_service.py  regex injection detector + sanitizer (layer 1 of defense)
  feature_service.py     structured fraud-signal features
  labeling_service.py    heuristic pseudo-labels (weak supervision, NOT ground truth)
  prompt_service.py      prompt template with untrusted-note fencing (layer 2 of defense)
  validation_service.py  strict JSON schema validation + rule-based fallback
  inference_service.py   loads the SLM (4-bit), batched generation
  fine_tuning_service.py LoRA fine-tune on the heuristic pseudo-labels

controllers/     orchestrate several services into one coherent stage
  data_pipeline_controller.py   clean -> merge -> notes -> sanitize -> features -> labels
  inference_controller.py       prompt -> SLM -> validate, baseline-vs-fine-tuned comparison
  training_controller.py        LoRA fine-tuning orchestration

routes/          runnable entry points
  run_full_pipeline.py   python -m routes.run_full_pipeline
  run_fine_tuning.py     python -m routes.run_fine_tuning

utils/           shared helpers (paths/IO, logging, seeding)

notebooks/
  hackathon_submission.ipynb   the actual submission notebook -- imports the
                               package above and runs/narrates the full flow

data/raw/        the three original CSVs (transactions/accounts/customers)
data/processed/  (reserved for intermediate cleaned output, if persisted)
outputs/         predictions.json, audit_log.json, evaluation_report.json

tests/           pytest unit tests for every deterministic service (57 tests)
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

(Omit `--extra-index-url` for CPU-only torch -- the pipeline detects CUDA
automatically and falls back gracefully, just much slower.)

## Running

```
python -m routes.run_full_pipeline        # clean -> ... -> fine-tune -> predictions.json
python -m routes.run_fine_tuning          # just the fine-tuning step
jupyter notebook notebooks/hackathon_submission.ipynb   # the narrated submission
```

## Testing & code standards

```
pip install -r requirements-dev.txt
pytest                                     # 57 tests, all deterministic services
ruff check models services controllers routes utils tests
black models services controllers routes utils tests
```

Lint/format rules live in `pyproject.toml`.

## Two explicit data decisions (documented, not hidden)

1. **Synthetic `transaction_notes`.** The supplied `transactions.csv` has no
   free-text notes/memo column at all (verified: zero hits for any
   injection-style phrase anywhere in the file). The brief explicitly
   requires neutralizing injection attempts hidden in transaction notes, so
   rather than skip that requirement, `services/notes_service.py` generates a
   deterministic, seeded `transaction_notes__SYNTHETIC` column -- clearly
   marked as hackathon robustness test data, not part of the original export.

2. **Heuristic pseudo-labels for fine-tuning.** No authoritative fraud label
   exists in the supplied data. `services/labeling_service.py` derives a
   weak, rule-based `_pseudo_is_fraud` label (>=3 independent structured risk
   signals) solely to give the LoRA fine-tune something to learn from. This
   is weak supervision, never reported as real fraud ground truth or real
   accuracy.

## Real data defects found (and fixed)

- Amounts: blank values, plus 6 rows with a leaked currency-code prefix
  (`"INR 62146.26"`), plus 11 rows with a quoted thousands-separator comma.
- Timestamps: mixed ISO / `DD/MM/YYYY HH:MM` / blank / literal
  `"NOT_AVAILABLE"`.
- Categoricals: mixed case and whitespace, and (found only via testing) the
  same category spelled with a space vs. an underscore (`INTERNET_BANKING`
  vs `INTERNET BANKING`), which naive strip+upper does not unify.
- Booleans: the same concept encoded as `0/1`, `TRUE/FALSE`, `yes/no`, `Y/N`,
  sometimes within the same column.
- 8 transactions reference `ACC_900000`-`ACC_900007`, account IDs that do not
  exist anywhere in `accounts.csv` -- fabricated/orphaned foreign keys, not
  recoverable by any join. Flagged as `account_reference_broken` and treated
  as a fraud signal itself rather than silently dropped.
- Credit limits and duplicate rows in `accounts.csv`/`customers.csv` were
  checked and found to be **clean** despite the brief's claim -- no fix was
  invented for a defect that isn't actually present.

## Hardware notes

Tested on an NVIDIA RTX 3050 4GB laptop GPU. The model loads in 4-bit
(bitsandbytes NF4), using ~1.1-2.6GB VRAM depending on batch size. Serial
(one-prompt-at-a-time) generation measured 15-29s/transaction; batched
generation (16-32 prompts at once) brought this down to ~1.8-2.0s/transaction
-- the same model and prompts, purely a throughput fix.
