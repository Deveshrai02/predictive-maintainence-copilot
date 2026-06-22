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
