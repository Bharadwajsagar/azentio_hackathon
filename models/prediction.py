"""Data schema for the required output contract -- exactly 4 fields, nothing else.

{
  "transaction_id": "TXN_00001",
  "is_fraud": true,
  "confidence": 0.92,
  "justification": "One-sentence plain-text justification based on transaction behavior."
}
"""

from dataclasses import dataclass


@dataclass
class PredictionOutput:
    transaction_id: str
    is_fraud: bool
    confidence: float
    justification: str

    def to_dict(self) -> dict:
        return {
            "transaction_id": self.transaction_id,
            "is_fraud": bool(self.is_fraud),
            "confidence": round(float(self.confidence), 4),
            "justification": self.justification,
        }
