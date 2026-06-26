# Predictive Maintenance Copilot

An AI copilot for industrial predictive maintenance. It combines a fine-tuned
text classifier over maintenance logs, an anomaly detector over NASA C-MAPSS
turbofan sensor data, a vector-store knowledge base, and a LangGraph agent that
ties the tools together behind a FastAPI service and a Streamlit UI.

## Project structure

```
predictive-maintenance-copilot/
├── data/
│   ├── cmapss/              # NASA C-MAPSS turbofan data (downloaded separately)
│   ├── maintenance_logs/    # Raw / synthetic maintenance logs
│   └── processed/           # Cleaned, feature-engineered datasets
├── training/
│   ├── generate_logs.py     # Generate synthetic maintenance logs
│   ├── train_classifier.py  # Fine-tune the log classifier (PEFT/LoRA)
│   └── evaluate_classifier.py
├── ingest/
│   └── load_weaviate.py     # Embed and load docs into Weaviate
├── app/
│   ├── agent.py             # LangGraph agent orchestration
│   ├── tools.py             # Agent tools
│   ├── classifier.py        # Inference wrapper for the fine-tuned classifier
│   ├── anomaly_detector.py  # Sensor anomaly detection
│   └── streamlit_app.py     # Streamlit front end
├── eval/
│   ├── test_cases.json
│   └── run_eval.py
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Getting started

Requires **Python 3.11**.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then fill in your credentials
```

## Environment variables

See `.env.example`:

- `ANTHROPIC_API_KEY`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `WEAVIATE_URL`
- `MLFLOW_TRACKING_URI`

## Data

The NASA C-MAPSS turbofan dataset is **not** committed to the repo. Download it
separately and place it under `data/cmapss/`.

## Deploy to Hugging Face Spaces

The app is built to run as a **single free Streamlit Space** — no AWS, no
separate Weaviate container, no MLflow server. It self-initialises its vector
store on startup, so the demo works reliably even after the Space sleeps and
cold-starts.

**Steps:**

1. **Create a new Space** at <https://huggingface.co/new-space>. For the SDK,
   select **Streamlit**.
2. **Set the app file** to `app/streamlit_app.py` (Space *Settings → App file*,
   or the `app_file:` field in the Space `README.md` front-matter).
3. **Add your Anthropic key as a secret**: Space *Settings → Variables and
   secrets → New secret* → name `ANTHROPIC_API_KEY`, value `sk-ant-...`.
4. **Add a public variable**: same screen → *New variable* → name
   `DEPLOYMENT_MODE`, value `anthropic`. (This selects the Anthropic-API LLM
   backend instead of AWS Bedrock.)
5. **Use the Spaces requirements file**: copy `requirements_spaces.txt` to
   `requirements.txt` in the Space (CPU-only torch, no GPU/AWS deps).
6. **Push the repo** — Spaces builds and deploys automatically on every push.
   Make sure `data/maintenance_logs/historical_logs.json` is committed; the app
   re-ingests it into the vector store on each cold start.

**How cold start works:** the Space runs Weaviate in **embedded** mode
(in-process, no container). That data lives on ephemeral disk and is wiped when
the Space sleeps, so on startup the app checks whether the `MaintenanceLog`
collection is populated and, if not, re-ingests the bundled
`historical_logs.json` behind an "Initialising knowledge base..." spinner before
accepting queries. This keeps the agent's retrieval grounded after every wake.

**Embedded vs. production:** embedded mode is for the single-container Space.
The production architecture (see `docker-compose.yml` / `k8s/`) instead uses a
**managed Weaviate cluster** and can run the LLM via **AWS Bedrock**
(`DEPLOYMENT_MODE=bedrock`, `WEAVIATE_MODE=cluster`) — intentionally different,
and kept so the full cloud architecture stays demonstrable.

**Classifier on Spaces:** the fine-tuned classifier normally loads from the
**MLflow model registry** (local/production). Since no MLflow server runs on
Spaces, `app/classifier.py` falls back to a fine-tuned model bundled at
`models/maintenance-classifier/`. (Train and export that directory first; note
model weight files are git-ignored by default, so commit them explicitly or use
the MLflow path locally.)
