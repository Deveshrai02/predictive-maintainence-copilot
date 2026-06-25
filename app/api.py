"""FastAPI service exposing the Predictive Maintenance Copilot over REST.

This is what the Docker `api` container launches:
    uvicorn app.api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health                              -> liveness check
    GET  /equipment/{id}/anomaly              -> current anomaly snapshot
    GET  /equipment/{id}/trend?cycles=10      -> recent sensor trend
    POST /diagnose  {equipment_id, log_entry} -> full agent diagnosis
"""

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from app.anomaly_detector import check_anomaly, get_recent_trend
from app.agent import run_agent

app = FastAPI(
    title="Predictive Maintenance Copilot API",
    description="Anomaly monitoring + self-reflective fault diagnosis. "
                "All data is simulated for portfolio purposes.",
    version="1.0.0",
)


class DiagnoseRequest(BaseModel):
    equipment_id: str
    log_entry: Optional[str] = None


@app.get("/health")
def health():
    """Simple liveness probe."""
    return {"status": "ok"}


@app.get("/equipment/{equipment_id}/anomaly")
def equipment_anomaly(equipment_id: str):
    """Current anomaly signal, severity, RUL and sensor summary for a machine."""
    return check_anomaly(equipment_id)


@app.get("/equipment/{equipment_id}/trend")
def equipment_trend(equipment_id: str, cycles: int = 10):
    """Last N cycles of sensor readings, to judge degrading vs stable."""
    return get_recent_trend(equipment_id, cycles=cycles)


@app.post("/diagnose")
def diagnose(req: DiagnoseRequest):
    """Run the full diagnostic agent and return its structured verdict."""
    return run_agent(equipment_id=req.equipment_id, log_entry=req.log_entry)
