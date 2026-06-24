"""Inference wrapper for the fine-tuned maintenance-log classifier.

This loads the DistilBERT+LoRA model we trained (and registered to the MLflow
model registry as "maintenance-fault-classifier", stage "Staging") and exposes
a single `classify_log()` call the agent's tools can use.

The model is loaded lazily and cached, so the (slow) load happens once.
"""

import os

# We import mlflow lazily inside the loader so this module can be imported even
# in environments where the model isn't available yet (the agent degrades
# gracefully instead of crashing at import time).

_REGISTRY_URI = os.getenv(
    "CLASSIFIER_MODEL_URI", "models:/maintenance-fault-classifier/Staging"
)

# Module-level cache. None = not loaded yet; False = tried and failed.
_PIPELINE = None


def _load_pipeline():
    """Load the text-classification pipeline from the MLflow registry once."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE or None  # False -> None (don't retry every call)

    try:
        import mlflow.transformers
        # The training script logged a transformers text-classification model,
        # so MLflow hands us back a ready-to-use HF pipeline.
        _PIPELINE = mlflow.transformers.load_model(_REGISTRY_URI)
    except Exception as exc:  # noqa: BLE001 - we want any failure to be soft
        print(f"[classifier] could not load model from {_REGISTRY_URI}: {exc}")
        _PIPELINE = False
        return None
    return _PIPELINE


def classify_log(log_text: str) -> dict:
    """Predict the fault category for one maintenance-log note.

    Returns {"fault_category": str, "confidence": float}. If the model is not
    available, returns a clearly-flagged fallback rather than raising, so the
    agent can still run and reason about the missing signal.
    """
    pipe = _load_pipeline()
    if pipe is None:
        return {
            "fault_category": "unknown",
            "confidence": 0.0,
            "error": "classifier model unavailable (not registered/loadable)",
        }

    # truncation/max_length mirror training (128 tokens) so inference matches.
    result = pipe(log_text, truncation=True, max_length=128)[0]
    return {
        "fault_category": result["label"],
        "confidence": round(float(result["score"]), 4),
    }
