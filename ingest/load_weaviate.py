"""Embed maintenance documents and load them into Weaviate.

This script populates a local Weaviate vector database with the synthetic
maintenance logs so the agent can do *semantic* search over them — i.e.
"find past incidents that read like this new one" rather than exact keyword
matching.

Written for the weaviate-client **v4** API (the modern client).
"""

import json
import os
from datetime import datetime, timezone
from urllib.parse import urlparse

import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType, Configure

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "maintenance_logs", "historical_logs.json",
)
COLLECTION_NAME = "MaintenanceLog"
BATCH_SIZE = 50

# Weaviate location. Defaults to the local Docker instance. We read it from an
# env var so the same script works locally and in other environments.
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")


# --------------------------------------------------------------------------- #
# Connection helper
# --------------------------------------------------------------------------- #
def connect():
    """Open a connection to the Weaviate instance named in WEAVIATE_URL.

    The v4 client talks over both HTTP (REST) and gRPC. For a standard local
    Docker setup the HTTP port is 8080 and gRPC is 50051, which is what our
    docker-compose exposes.
    """
    parsed = urlparse(WEAVIATE_URL)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8080
    return weaviate.connect_to_local(host=host, port=port, grpc_port=50051)


# --------------------------------------------------------------------------- #
# Schema definition
# --------------------------------------------------------------------------- #
def create_collection(client):
    """Create the MaintenanceLog collection with its vectoriser.

    Why we do NOT generate embeddings ourselves:
      We attach the `text2vec-transformers` vectoriser (all-MiniLM-L6-v2) to
      the collection. Because of this, Weaviate runs the embedding model
      *server-side* every time we insert an object: it reads the log_text,
      calls the transformer model, and stores the resulting vector for us.
      So we just send plain JSON — no need to load a model in this script,
      no manual embedding step, and queries are embedded the same way for a
      consistent vector space. The configuration below is what makes that
      automatic.

    Which field gets vectorised:
      By default text2vec would embed every text property. We only want the
      free-text engineer note (log_text) to drive semantic search, so the
      other text fields are marked `skip_vectorization=True`. Mixing IDs and
      category labels into the vector would dilute the meaning of the search.
    """
    client.collections.create(
        name=COLLECTION_NAME,
        # all-MiniLM-L6-v2 served by the t2v-transformers sidecar container.
        vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
        properties=[
            # Identifiers / labels — stored but kept OUT of the vector.
            Property(name="log_id", data_type=DataType.TEXT,
                     skip_vectorization=True, vectorize_property_name=False),
            Property(name="equipment_id", data_type=DataType.TEXT,
                     skip_vectorization=True, vectorize_property_name=False),
            Property(name="fault_category", data_type=DataType.TEXT,
                     skip_vectorization=True, vectorize_property_name=False),
            # The one field we DO embed — this is what semantic search matches on.
            Property(name="log_text", data_type=DataType.TEXT),
            Property(name="resolution", data_type=DataType.TEXT,
                     skip_vectorization=True, vectorize_property_name=False),
            # Numeric + date fields.
            Property(name="downtime_minutes", data_type=DataType.NUMBER),
            Property(name="timestamp", data_type=DataType.DATE),
        ],
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_rfc3339(ts: str) -> datetime:
    """Weaviate DATE properties need a timezone-aware datetime.

    Our generated timestamps are ISO strings without a timezone
    (e.g. "2024-03-05T14:23:00"). We parse them and assume UTC so Weaviate
    accepts them as valid RFC3339 dates.
    """
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def load_records() -> list:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run training/generate_logs.py first."
        )
    with open(DATA_PATH) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def ingest(client, records):
    """Insert all records in batches of BATCH_SIZE, reporting progress."""
    collection = client.collections.get(COLLECTION_NAME)
    total = len(records)

    # fixed_size batching sends objects to Weaviate in groups, which is far
    # faster than one HTTP round-trip per object. The transformer model embeds
    # each object's log_text as it lands.
    with collection.batch.fixed_size(batch_size=BATCH_SIZE) as batch:
        for i, rec in enumerate(records, start=1):
            batch.add_object(properties={
                "log_id": rec["log_id"],
                "equipment_id": rec["equipment_id"],
                "fault_category": rec["fault_category"],
                "log_text": rec["log_text"],
                "resolution": rec["resolution"],
                "downtime_minutes": rec["downtime_minutes"],
                "timestamp": _to_rfc3339(rec["timestamp"]),
            })
            # Report progress at every batch boundary and at the end.
            if i % BATCH_SIZE == 0 or i == total:
                print(f"  queued {i}/{total} records")

    # The batch context flushes on exit; surface any objects that failed.
    failed = collection.batch.failed_objects
    if failed:
        print(f"  WARNING: {len(failed)} objects failed to insert. "
              f"First error: {failed[0].message}")
    else:
        print(f"  All {total} records ingested successfully.")


# --------------------------------------------------------------------------- #
# Test query
# --------------------------------------------------------------------------- #
def test_query(client):
    """Run one semantic search to prove the vector index works."""
    collection = client.collections.get(COLLECTION_NAME)
    query_text = "bearing noise at high speed on drive shaft"

    # near_text embeds the query with the SAME model used at insert time, then
    # returns the nearest stored vectors. certainty is a 0-1 similarity score
    # (higher = more similar); distance is its inverse (lower = more similar).
    results = collection.query.near_text(
        query=query_text,
        limit=3,
        return_metadata=wvc.query.MetadataQuery(distance=True, certainty=True),
    )

    print(f"\nTop 3 logs similar to: \"{query_text}\"")
    for rank, obj in enumerate(results.objects, start=1):
        props = obj.properties
        certainty = obj.metadata.certainty
        print(f"  {rank}. [{props['fault_category']}] "
              f"similarity={certainty:.3f}")
        print(f"     {props['log_text']}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    client = connect()
    try:
        # Idempotency guard: if the collection already exists AND holds data,
        # skip ingestion so re-running the script never duplicates records.
        if client.collections.exists(COLLECTION_NAME):
            existing = client.collections.get(COLLECTION_NAME)
            count = existing.aggregate.over_all(total_count=True).total_count
            if count and count > 0:
                print(f"Collection '{COLLECTION_NAME}' already has {count} "
                      f"objects — skipping ingestion.")
                test_query(client)
                return
            # Exists but empty: drop the empty shell and recreate cleanly.
            client.collections.delete(COLLECTION_NAME)

        print(f"Creating collection '{COLLECTION_NAME}'...")
        create_collection(client)

        records = load_records()
        print(f"Ingesting {len(records)} records in batches of {BATCH_SIZE}...")
        ingest(client, records)

        test_query(client)
    finally:
        # Always close the connection (frees the gRPC channel).
        client.close()


if __name__ == "__main__":
    main()
