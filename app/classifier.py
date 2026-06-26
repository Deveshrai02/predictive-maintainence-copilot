"""Inference wrapper for the fine-tuned maintenance-log classifier.

Loads the DistilBERT+LoRA model we trained from the MLflow **model registry**
and exposes a single `classify()` call.

Why load from the MLflow registry instead of a local file path
--------------------------------------------------------------
A hardcoded path like "training/checkpoints/best" freezes the agent to one
specific file on one machine. By loading from the registry by *stage*
("Production", falling back to "Staging") instead, the agent always picks up
whatever model has been **promoted** through our MLOps process — no code change
needed to ship a better model. We retrain, register a new version, promote it
to Production, and the next process restart serves it automatically. The model
lifecycle is decoupled from the application code.
"""

import os

# MLflow tracking server location. If unset, MLflow uses its local default
# (./mlruns). We only override when an explicit URI is provided.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = "maintenance-fault-classifier"

# Where to look, in priority order: prefer Production, fall back to Staging.
_STAGE_PRIORITY = ["Production", "Staging"]

# Fallback location: a fine-tuned model directory bundled in the repo. Used when
# the MLflow registry is unreachable — e.g. on Hugging Face Spaces, where no
# MLflow server runs. This lets the demo run standalone while local/production
# still load from the MLflow registry (the path above).
LOCAL_MODEL_DIR = os.getenv(
    "CLASSIFIER_LOCAL_DIR",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "maintenance-classifier",
    ),
)

# In-memory cache. None = not loaded yet; False = tried and failed (don't retry
# on every call). Anything else is the loaded HF pipeline.
_PIPELINE = None
_LOADED_STAGE = None


def _resolve_stage(client) -> str:
    """Return the first stage that actually has a registered version."""
    for stage in _STAGE_PRIORITY:
        # get_latest_versions returns the newest model version in that stage.
        versions = client.get_latest_versions(MODEL_NAME, stages=[stage])
        if versions:
            return stage
    raise RuntimeError(
        f"No 'Production' or 'Staging' version found for model '{MODEL_NAME}'."
    )


def _load_pipeline():
    """Load the text-classification pipeline from the registry, once, cached."""
    global _PIPELINE, _LOADED_STAGE
    if _PIPELINE is not None:
        return _PIPELINE or None  # cached pipeline, or False -> None

    try:
        import mlflow
        import mlflow.transformers
        from mlflow.tracking import MlflowClient

        if MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Decide which stage to serve (Production preferred), then load that
        # stage's latest version. The "models:/name/Stage" URI always points at
        # whatever version currently occupies that stage.
        client = MlflowClient()
        stage = _resolve_stage(client)
        model_uri = f"models:/{MODEL_NAME}/{stage}"

        # We logged a transformers pipeline during training, so MLflow returns a
        # ready-to-use Hugging Face text-classification pipeline.
        _PIPELINE = mlflow.transformers.load_model(model_uri)
        _LOADED_STAGE = stage
        print(f"[classifier] loaded '{MODEL_NAME}' from MLflow stage '{stage}'.")
        return _PIPELINE
    except Exception as exc:  # noqa: BLE001 - registry unreachable -> try local
        print(f"[classifier] MLflow registry unavailable ({exc}); "
              f"trying local model dir...")

    # Fallback: load the bundled fine-tuned model straight from disk with a
    # plain transformers pipeline (no MLflow needed). This is the Spaces path.
    try:
        if not os.path.isdir(LOCAL_MODEL_DIR):
            raise FileNotFoundError(f"{LOCAL_MODEL_DIR} not found")
        from transformers import pipeline
        _PIPELINE = pipeline(
            "text-classification",
            model=LOCAL_MODEL_DIR,
            tokenizer=LOCAL_MODEL_DIR,
        )
        _LOADED_STAGE = "local"
        print(f"[classifier] loaded model from local dir '{LOCAL_MODEL_DIR}'.")
    except Exception as exc:  # noqa: BLE001 - any failure should be soft
        print(f"[classifier] could not load model from MLflow or local dir: {exc}")
        _PIPELINE = False
        return None
    return _PIPELINE


def classify(log_text: str) -> dict:
    """Classify a maintenance-log note.

    Returns:
        {
          "predicted_category": str,   # most likely fault category
          "confidence": float,         # 0-1 score for that category
          "all_scores": {category: score, ...},  # full distribution
        }

    If the model can't be loaded, returns a clearly-flagged fallback instead of
    raising, so callers (e.g. the agent) keep working.
    """
    pipe = _load_pipeline()
    if pipe is None:
        return {
            "predicted_category": "unknown",
            "confidence": 0.0,
            "all_scores": {},
            "error": "classifier model unavailable (no Production/Staging version)",
        }

    # top_k=None asks the pipeline for EVERY category's score (not just the top
    # one). truncation/max_length mirror training (128 tokens) so the tokenised
    # input the model sees at inference matches what it saw while learning.
    raw = pipe(log_text, top_k=None, truncation=True, max_length=128)

    # The pipeline returns a list of {"label": ..., "score": ...} dicts.
    all_scores = {item["label"]: round(float(item["score"]), 4) for item in raw}
    best_label = max(all_scores, key=all_scores.get)

    return {
        "predicted_category": best_label,
        "confidence": all_scores[best_label],
        "all_scores": all_scores,
    }


# --------------------------------------------------------------------------- #
# Standalone test: `python app/classifier.py`
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import json

    # One terse, engineer-style note per fault type to eyeball the predictions.
    examples = [
        "DE bearing running hot at 82C, growl at 1480 rpm, vibration trending up.",
        "Gearbox oil level below min on sight glass, weep at output seal, oil dark.",
        "VFD tripped on overcurrent fault F012, phase imbalance ~6%, motor megger marginal.",
        "Flow transmitter reading 4% high vs reference, zero drifted, last cal 14 months ago.",
        "PLC comms dropout to remote IO rack, watchdog timeout, HMI froze until restart.",
    ]

    for text in examples:
        result = classify(text)
        print(f"\nNote: {text}")
        print(json.dumps(result, indent=2))
