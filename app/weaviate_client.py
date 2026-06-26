"""Shared Weaviate client + in-process embedding helper.

Connection mode (env var WEAVIATE_MODE):
  * "embedded" (default) — runs Weaviate IN-PROCESS via EmbeddedOptions, with no
    separate container. This is what the single-container Hugging Face Spaces
    deployment uses.
  * "cluster" — connects to a managed Weaviate cluster (production path). The
    docker-compose.yml / k8s/ setup provisions that cluster and is intentionally
    different from this embedded path.

Why we vectorise in-process here:
  The Docker/cluster setup vectorises with the `text2vec-transformers` MODULE,
  which is a SEPARATE inference container (the t2v-transformers sidecar).
  Embedded mode is a single process with no sidecar to call, so instead we
  compute embeddings locally with all-MiniLM-L6-v2 (the same model the sidecar
  ran) and store them as self-provided vectors (`vectorizer: none`). Queries
  embed the text the same way and use near_vector. Same model, same vector
  space — just computed in-process instead of in a sidecar.
"""

import os

import weaviate
from weaviate.embedded import EmbeddedOptions

COLLECTION_NAME = "MaintenanceLog"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Module-level singletons. The embedded Weaviate server is tied to its client;
# we keep ONE long-lived client for the whole process rather than starting and
# stopping the embedded server on every call.
_CLIENT = None
_EMBEDDER = None


def get_client():
    """Return a connected Weaviate client (embedded by default), cached."""
    global _CLIENT
    if _CLIENT is not None:
        try:
            if _CLIENT.is_connected():
                return _CLIENT
        except Exception:  # noqa: BLE001 - rebuild on any stale-handle error
            pass

    mode = os.getenv("WEAVIATE_MODE", "embedded")
    if mode == "cluster":
        # Production: connect to the managed cluster from docker-compose / k8s.
        from urllib.parse import urlparse
        url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        parsed = urlparse(url)
        _CLIENT = weaviate.connect_to_local(
            host=parsed.hostname or "localhost",
            port=parsed.port or 8080,
            grpc_port=50051,
        )
    else:
        # Embedded: in-process Weaviate, single container (Hugging Face Spaces).
        _CLIENT = weaviate.WeaviateClient(
            embedded_options=EmbeddedOptions(
                persistence_data_path=os.path.expanduser(
                    os.getenv("WEAVIATE_DATA_PATH", "~/.local/share/weaviate-embedded")
                ),
                # No vectoriser modules needed — we supply vectors ourselves.
                additional_env_vars={
                    "ENABLE_MODULES": "",
                    "DEFAULT_VECTORIZER_MODULE": "none",
                },
            )
        )
        _CLIENT.connect()
    return _CLIENT


def _embedder():
    """Lazily load the local sentence-transformers model, cached."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(EMBED_MODEL)
    return _EMBEDDER


def embed(texts) -> list:
    """Return normalized embedding vectors (list of float lists) for `texts`."""
    return _embedder().encode(list(texts), normalize_embeddings=True).tolist()


def embed_one(text: str) -> list:
    """Embed a single string and return its vector."""
    return embed([text])[0]


def collection_is_populated(client) -> bool:
    """True if the MaintenanceLog collection exists AND holds >= 1 object."""
    if not client.collections.exists(COLLECTION_NAME):
        return False
    total = (
        client.collections.get(COLLECTION_NAME)
        .aggregate.over_all(total_count=True)
        .total_count
    )
    return bool(total and total > 0)
