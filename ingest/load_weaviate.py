"""Embed maintenance documents and load them into Weaviate (embedded mode).

This populates an EMBEDDED, in-process Weaviate instance with the synthetic
maintenance logs so the agent can do semantic search over them. Embedded mode
runs Weaviate inside this Python process (no separate container) — used for the
single-container Hugging Face Spaces deployment. Production uses a managed
Weaviate cluster instead (see docker-compose.yml / k8s/, intentionally
different); set WEAVIATE_MODE=cluster to target it.

Vectors are computed in-process with all-MiniLM-L6-v2 and stored as
self-provided vectors (vectorizer: none), because embedded mode has no
text2vec-transformers sidecar to call. See app/weaviate_client.py.
"""

import json
import os
import sys

# Allow `from app...` when this file is launched directly from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weaviate.classes.config import Property, DataType, Configure  # noqa: E402
from weaviate.classes.query import MetadataQuery  # noqa: E402

from app.weaviate_client import (  # noqa: E402
    get_client, embed, embed_one, collection_is_populated, COLLECTION_NAME,
)

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "maintenance_logs", "historical_logs.json",
)
BATCH_SIZE = 50


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_records() -> list:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run training/generate_logs.py first."
        )
    with open(DATA_PATH) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def create_collection(client):
    """Create the MaintenanceLog collection with a self-provided vectoriser.

    vectorizer = none: we compute embeddings ourselves and pass them in, since
    embedded mode has no text2vec-transformers sidecar (see module docstring).
    """
    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="log_id", data_type=DataType.TEXT),
            Property(name="equipment_id", data_type=DataType.TEXT),
            Property(name="fault_category", data_type=DataType.TEXT),
            Property(name="log_text", data_type=DataType.TEXT),
            Property(name="resolution", data_type=DataType.TEXT),
            Property(name="downtime_minutes", data_type=DataType.NUMBER),
            Property(name="timestamp", data_type=DataType.TEXT),
        ],
    )


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def ingest(client, records):
    """Insert all records in batches, attaching a locally-computed vector each."""
    collection = client.collections.get(COLLECTION_NAME)
    total = len(records)

    # Embed every log_text up front (one batched model call is far faster than
    # embedding one row at a time).
    vectors = embed([r["log_text"] for r in records])

    with collection.batch.fixed_size(batch_size=BATCH_SIZE) as batch:
        for i, (rec, vec) in enumerate(zip(records, vectors), start=1):
            batch.add_object(
                properties={
                    "log_id": rec["log_id"],
                    "equipment_id": rec["equipment_id"],
                    "fault_category": rec["fault_category"],
                    "log_text": rec["log_text"],
                    "resolution": rec["resolution"],
                    "downtime_minutes": rec["downtime_minutes"],
                    "timestamp": rec["timestamp"],
                },
                vector=vec,  # self-provided vector
            )
            if i % BATCH_SIZE == 0 or i == total:
                print(f"  queued {i}/{total} records")

    failed = collection.batch.failed_objects
    if failed:
        print(f"  WARNING: {len(failed)} objects failed. First: {failed[0].message}")
    else:
        print(f"  All {total} records ingested successfully.")


def run_ingestion() -> int:
    """(Re)create the collection and load every record. Returns the count.

    Drops any existing collection first so a re-ingest is always clean.
    """
    client = get_client()
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)
    print(f"Creating collection '{COLLECTION_NAME}'...")
    create_collection(client)
    records = load_records()
    print(f"Ingesting {len(records)} records in batches of {BATCH_SIZE}...")
    ingest(client, records)
    return len(records)


def ensure_populated() -> int:
    """Ingest only if the collection is empty. Returns the record count.

    Called on app startup (incl. Spaces cold start) so retrieval always has
    data. Cheap no-op when the store is already populated.
    """
    client = get_client()
    if collection_is_populated(client):
        return (
            client.collections.get(COLLECTION_NAME)
            .aggregate.over_all(total_count=True)
            .total_count
        )
    return run_ingestion()


# --------------------------------------------------------------------------- #
# Test query
# --------------------------------------------------------------------------- #
def test_query(client):
    """Run one semantic search to prove the vector index works."""
    collection = client.collections.get(COLLECTION_NAME)
    query_text = "bearing noise at high speed on drive shaft"

    results = collection.query.near_vector(
        near_vector=embed_one(query_text),
        limit=3,
        return_metadata=MetadataQuery(certainty=True),
    )

    print(f"\nTop 3 logs similar to: \"{query_text}\"")
    for rank, obj in enumerate(results.objects, start=1):
        p = obj.properties
        cert = obj.metadata.certainty
        print(f"  {rank}. [{p['fault_category']}] similarity={cert:.3f}")
        print(f"     {p['log_text']}")


# --------------------------------------------------------------------------- #
# Main (standalone)
# --------------------------------------------------------------------------- #
def main():
    client = get_client()
    if collection_is_populated(client):
        count = (
            client.collections.get(COLLECTION_NAME)
            .aggregate.over_all(total_count=True)
            .total_count
        )
        print(f"Collection '{COLLECTION_NAME}' already has {count} objects — "
              f"skipping ingestion.")
    else:
        run_ingestion()
    test_query(client)
    # Standalone run: stop the embedded server we started.
    client.close()


if __name__ == "__main__":
    main()
