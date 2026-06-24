"""
train_classifier.py — Fine-tune DistilBERT on synthetic maintenance logs to
classify fault categories, using LoRA (PEFT) for parameter-efficient training.

Pipeline:
  1. Load + stratified 80/20 split of the log dataset.
  2. Tokenise with DistilBERT tokeniser (max 128 tokens).
  3. Wrap the model with a LoRA adapter (PEFT) on the attention projections.
  4. Train with the HF Trainer (5 epochs, best-checkpoint on val accuracy).
  5. Log everything to MLflow + register the model as "Staging".
  6. Report final accuracy + per-category precision/recall/F1.
"""

import json
import os

import numpy as np
import mlflow
import mlflow.transformers
from mlflow.tracking import MlflowClient

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #
DATA_PATH = "data/maintenance_logs/historical_logs.json"
METRICS_PATH = "data/processed/classifier_metrics.json"
OUTPUT_DIR = "training/checkpoints"
BASE_MODEL = "distilbert-base-uncased"
MODEL_NAME = "maintenance-fault-classifier"   # MLflow registry name
MAX_TOKENS = 128
SEED = 42


# --------------------------------------------------------------------------- #
# 1. Load data + stratified split
# --------------------------------------------------------------------------- #
def load_and_split():
    with open(DATA_PATH) as f:
        records = json.load(f)

    texts = [r["log_text"] for r in records]
    labels_str = [r["fault_category"] for r in records]

    # Build a stable string<->int label mapping. Sorting makes the id
    # assignment deterministic across runs, which matters because the model
    # head, id2label, and the saved metrics all have to agree.
    categories = sorted(set(labels_str))
    label2id = {c: i for i, c in enumerate(categories)}
    id2label = {i: c for c, i in label2id.items()}
    labels = [label2id[c] for c in labels_str]

    # stratify=labels guarantees each fault category keeps the same proportion
    # in train and validation. With only ~45 examples per class this is
    # essential — a plain random split could leave a class barely represented
    # (or absent) in validation and make the accuracy meaningless.
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=0.20,
        stratify=labels,
        random_state=SEED,
    )
    return (train_texts, train_labels, val_texts, val_labels,
            label2id, id2label, categories)


# --------------------------------------------------------------------------- #
# 2. Tokenisation
# --------------------------------------------------------------------------- #
def build_datasets(tokenizer, train_texts, train_labels, val_texts, val_labels):
    # Why max 128 tokens is plenty here:
    #   These logs are terse engineer notes — typically one or two clauses,
    #   e.g. "DE bearing running hot at 82C, growl at 1480 rpm on line 3 drive."
    #   That tokenises to well under 50 tokens. 128 leaves comfortable headroom
    #   for the longest two-symptom notes while keeping the attention cost low
    #   (cost scales with sequence length). A larger window would just pad with
    #   wasted compute and no accuracy gain.
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,        # clip anything past 128 (rare here)
            max_length=MAX_TOKENS,
        )

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})

    # batched=True tokenises in chunks for speed; we drop the raw text column
    # afterwards since the Trainer only needs input_ids/attention_mask/label.
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])
    return train_ds, val_ds


# --------------------------------------------------------------------------- #
# 3. LoRA configuration
# --------------------------------------------------------------------------- #
def wrap_with_lora(num_labels, label2id, id2label):
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    # What "rank" means in LoRA:
    #   LoRA freezes the original weight matrix W and learns a low-rank update
    #   dW = B @ A, where A is (r x d) and B is (d x r). "rank" = r is the inner
    #   dimension of that bottleneck. Small r (=8) means the adapter can only
    #   express a low-dimensional change to each attention projection — few
    #   trainable params, strong regulariser, ideal for a small dataset.
    #   Larger r => more capacity but more params and more overfitting risk.
    #
    #   lora_alpha (16) is a scaling factor; the update is applied as
    #   (alpha / r) * dW, so alpha=16, r=8 gives a 2x scale on the adapter.
    #   lora_dropout (0.1) randomly zeroes adapter activations for regularisation.
    #   target_modules = the attention query/value projections, which in
    #   DistilBERT are named q_lin and v_lin — adapting attention is where LoRA
    #   gives the most task adaptation per parameter.
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,       # sequence classification head
        target_modules=["q_lin", "v_lin"],
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
    )

    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()    # shows the tiny trainable %
    return model


# --------------------------------------------------------------------------- #
# Metric fn for the Trainer (val accuracy drives best-checkpoint selection)
# --------------------------------------------------------------------------- #
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


# --------------------------------------------------------------------------- #
# 5. MLflow per-epoch logging callback
# --------------------------------------------------------------------------- #
class MLflowEpochCallback(TrainerCallback):
    """Logs training loss and validation accuracy to MLflow every epoch.

    The Trainer emits two kinds of logs: training logs (contain 'loss') and
    evaluation logs (contain 'eval_accuracy'). We forward both to MLflow with
    the epoch number as the step so the curves are per-epoch.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        epoch = int(state.epoch) if state.epoch is not None else state.global_step
        if "loss" in logs:
            mlflow.log_metric("train_loss", logs["loss"], step=epoch)
        if "eval_accuracy" in logs:
            mlflow.log_metric("val_accuracy", logs["eval_accuracy"], step=epoch)
        if "eval_loss" in logs:
            mlflow.log_metric("val_loss", logs["eval_loss"], step=epoch)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    (train_texts, train_labels, val_texts, val_labels,
     label2id, id2label, categories) = load_and_split()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_ds, val_ds = build_datasets(
        tokenizer, train_texts, train_labels, val_texts, val_labels
    )

    model = wrap_with_lora(len(categories), label2id, id2label)

    # Dynamic padding: pad each batch to its own longest sequence rather than
    # to 128 every time — cheaper than static padding to max_length.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4. Trainer configuration -------------------------------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,                 # small data -> a few passes is enough
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-4,                 # LoRA tolerates a higher LR than full
                                            # fine-tuning since only adapters train
        weight_decay=0.01,                  # mild L2 regularisation
        eval_strategy="epoch",              # evaluate once per epoch
        save_strategy="epoch",              # checkpoint once per epoch...
        load_best_model_at_end=True,        # ...and reload the best at the end
        metric_for_best_model="accuracy",   # "best" = highest validation accuracy
        greater_is_better=True,
        logging_strategy="epoch",
        report_to="none",                   # we drive MLflow manually via callback
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[MLflowEpochCallback()],
    )

    mlflow.set_experiment("maintenance-fault-classifier")
    with mlflow.start_run() as run:
        # --- log hyperparameters up front ---
        mlflow.log_params({
            "base_model": BASE_MODEL,
            "max_tokens": MAX_TOKENS,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "lora_targets": "q_lin,v_lin",
            "epochs": 5,
            "batch_size": 16,
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "num_labels": len(categories),
        })

        # --- train (per-epoch loss/accuracy logged by the callback) ---
        trainer.train()

        # 6. Final evaluation + classification report -------------------------
        preds_output = trainer.predict(val_ds)
        y_pred = np.argmax(preds_output.predictions, axis=-1)
        y_true = preds_output.label_ids

        final_acc = accuracy_score(y_true, y_pred)
        report = classification_report(
            y_true, y_pred,
            target_names=categories,        # human-readable category names
            output_dict=True,
            zero_division=0,
        )

        mlflow.log_metric("final_val_accuracy", final_acc)
        # Flatten per-category F1 into MLflow for easy comparison across runs.
        for cat in categories:
            mlflow.log_metric(f"f1_{cat}", report[cat]["f1-score"])

        # --- register the best model to the MLflow registry as "Staging" ---
        # Merge the LoRA adapter into the base weights so the logged artifact is
        # a standalone model (no PEFT needed to load it for inference).
        merged = trainer.model.merge_and_unload()
        pipe_components = {"model": merged, "tokenizer": tokenizer}
        mlflow.transformers.log_model(
            transformers_model=pipe_components,
            artifact_path="model",
            task="text-classification",
            registered_model_name=MODEL_NAME,
        )
        # Transition the just-registered version to the "Staging" stage.
        client = MlflowClient()
        latest = client.get_latest_versions(MODEL_NAME, stages=["None"])[0]
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest.version,
            stage="Staging",
        )

        # --- persist metrics to disk for the rest of the project ---
        os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
        with open(METRICS_PATH, "w") as f:
            json.dump(
                {"final_val_accuracy": final_acc,
                 "classification_report": report,
                 "mlflow_run_id": run.info.run_id},
                f, indent=2,
            )

    # --- console summary ---
    print("\n" + "=" * 60)
    print(f"Final validation accuracy: {final_acc:.4f}")
    print("=" * 60)
    print(classification_report(
        y_true, y_pred, target_names=categories, zero_division=0
    ))
    print(f"Metrics written to {METRICS_PATH}")


if __name__ == "__main__":
    main()