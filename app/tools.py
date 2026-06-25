"""Tools exposed to the LangGraph agent.

Each tool is a thin, well-documented wrapper around one capability of the
system (classifier, vector store, anomaly detector). They are decorated with
LangChain's @tool so the agent can bind them and call them by name. The
docstrings matter: the LLM reads them to decide when to use each tool.
"""

import os
from typing import Optional
from urllib.parse import urlparse

from langchain_core.tools import tool

from app import classifier
from app import anomaly_detector

# --------------------------------------------------------------------------- #
# Weaviate connection helper (v4 client)
# --------------------------------------------------------------------------- #
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
COLLECTION_NAME = "MaintenanceLog"


def _connect_weaviate():
    """Open a short-lived connection to the local Weaviate instance."""
    import weaviate
    parsed = urlparse(WEAVIATE_URL)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8080
    return weaviate.connect_to_local(host=host, port=port, grpc_port=50051)


# --------------------------------------------------------------------------- #
# Tool 1 — fault classification
# --------------------------------------------------------------------------- #
@tool
def classify_fault(log_text: str) -> dict:
    """Classify a maintenance-log note into one of the known fault categories.

    Use this whenever a free-text maintenance log entry is available. Returns
    the predicted fault_category and a confidence score (0-1).
    """
    # classifier.classify() returns predicted_category/confidence/all_scores;
    # we surface it as fault_category for consistency with the rest of the agent.
    result = classifier.classify(log_text)
    return {
        "fault_category": result.get("predicted_category"),
        "confidence": result.get("confidence"),
        "all_scores": result.get("all_scores", {}),
    }


# --------------------------------------------------------------------------- #
# Tool 2 — semantic retrieval of past incidents
# --------------------------------------------------------------------------- #
@tool
def retrieve_similar_incidents(
    query: str, fault_category: Optional[str] = None, k: int = 3
) -> list:
    """Find past maintenance incidents semantically similar to `query`.

    Use this to ground a diagnosis in real historical events and their
    resolutions. Optionally restrict results to a single fault_category.
    Returns a list of dicts: log_id, equipment_id, fault_category, log_text,
    resolution, similarity_score. Returns [] if nothing relevant is found.
    """
    from weaviate.classes.query import Filter, MetadataQuery

    client = _connect_weaviate()
    try:
        collection = client.collections.get(COLLECTION_NAME)

        # If a category was given, add a server-side WHERE filter so we only
        # search within that fault type — sharper, more relevant results.
        filters = None
        if fault_category:
            filters = Filter.by_property("fault_category").equal(fault_category)

        response = collection.query.near_text(
            query=query,
            limit=k,
            filters=filters,
            return_metadata=MetadataQuery(certainty=True),
        )

        incidents = []
        for obj in response.objects:
            p = obj.properties
            incidents.append({
                "log_id": p.get("log_id"),
                "equipment_id": p.get("equipment_id"),
                "fault_category": p.get("fault_category"),
                "log_text": p.get("log_text"),
                "resolution": p.get("resolution"),
                # certainty is Weaviate's 0-1 similarity score (higher = closer).
                "similarity_score": round(obj.metadata.certainty, 4)
                if obj.metadata.certainty is not None else None,
            })
        return incidents
    except Exception as exc:  # noqa: BLE001 - degrade gracefully for the agent
        return [{"error": f"retrieval failed: {exc}"}]
    finally:
        client.close()


# --------------------------------------------------------------------------- #
# Tool 3 — live anomaly signal
# --------------------------------------------------------------------------- #
@tool
def check_equipment_anomaly(equipment_id: str) -> dict:
    """Check the current anomaly signal and RUL estimate for a machine.

    Use this FIRST for any diagnosis: it reports current_rul_estimate,
    anomaly_detected, anomaly_severity (normal/warning/critical), and the
    latest key sensor values.
    """
    return anomaly_detector.check_anomaly(equipment_id)


# --------------------------------------------------------------------------- #
# Tool 4 — recent sensor trend
# --------------------------------------------------------------------------- #
@tool
def get_sensor_trend(equipment_id: str) -> dict:
    """Return the last 10 cycles of sensor readings for a machine.

    Use this to judge whether the machine is degrading (values drifting) or
    stable (values flat) over time — direction, not just a snapshot.
    """
    return anomaly_detector.get_recent_trend(equipment_id, cycles=10)


# Convenience exports for the agent graph.
TOOLS = [
    classify_fault,
    retrieve_similar_incidents,
    check_equipment_anomaly,
    get_sensor_trend,
]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
