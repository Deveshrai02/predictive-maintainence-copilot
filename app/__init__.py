"""Predictive Maintenance Copilot application package.

Loads environment variables from a project-root .env file on import, so every
entry point (Streamlit UI, FastAPI service, the agent, the tools) picks up the
AWS / Weaviate / MLflow configuration without each module having to call
load_dotenv() itself. Importing any app.* submodule runs this first.
"""

import os

try:
    from dotenv import load_dotenv
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Only loads variables that aren't already set in the real environment,
    # so container-provided env (docker-compose) still takes precedence.
    load_dotenv(os.path.join(_ROOT, ".env"))
except Exception:
    # python-dotenv missing or no .env present — fall back to the shell env.
    pass
