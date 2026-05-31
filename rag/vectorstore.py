"""
ChromaDB wrapper — create / get collections, upsert chunks, similarity search.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Dict, List, Optional

import chromadb
from langchain_core.documents import Document

from rag.config import CHROMA_DIR
from rag.embeddings import get_embedder


# Chroma's PersistentClient is not safe to instantiate concurrently on the
# same directory — 5 parallel LangGraph master nodes otherwise trip
# "Could not connect to tenant default_tenant" and rust "bindings" errors.
# A single cached client, created behind a lock, serves all threads.
_cached_client: Optional[chromadb.ClientAPI] = None
_client_lock = threading.Lock()


def _get_client() -> chromadb.ClientAPI:
    global _cached_client
    if _cached_client is not None:
        return _cached_client
    with _client_lock:
        if _cached_client is None:
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            _cached_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _cached_client


def _get_embeddings():
    """Unified embedder (local BGE-M3 or OpenAI, see rag/embeddings.py)."""
    return get_embedder()


def _doc_id(doc: Document) -> str:
    """Deterministic ID from full content + source so re-runs are idempotent
    and collisions are astronomically unlikely even across large corpora."""
    sig = f"{doc.metadata.get('source', '')}::{doc.page_content}"
    return hashlib.sha256(sig.encode("utf-8", errors="ignore")).hexdigest()[:20]


# ============================================================
#  Collection management
# ============================================================

def get_or_create_collection(
    collection_name: str,
) -> chromadb.Collection:
    client = _get_client()
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def drop_collection(collection_name: str) -> None:
    client = _get_client()
    try:
        client.delete_collection(collection_name)
    except ValueError:
        pass


def collection_count(collection_name: str) -> int:
    try:
        col = get_or_create_collection(collection_name)
        return col.count()
    except Exception:
        return 0


# ============================================================
#  Upsert
# ============================================================

def upsert_chunks(
    collection_name: str,
    chunks: List[Document],
    batch_size: int = 64,
) -> int:
    """Embed and upsert chunks into ChromaDB. Returns count of upserted docs."""
    if not chunks:
        return 0

    col = get_or_create_collection(collection_name)
    embedder = _get_embeddings()

    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.page_content for c in batch]
        ids = [_doc_id(c) for c in batch]
        metadatas = [_sanitize_metadata(c.metadata) for c in batch]

        embeddings = embedder.embed_documents(texts)

        col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        total += len(batch)
        print(f"    Upserted batch {i // batch_size + 1} ({total}/{len(chunks)})")

    return total


def _sanitize_metadata(meta: dict) -> dict:
    """ChromaDB only accepts str/int/float/bool metadata values."""
    clean: Dict[str, object] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, list):
            clean[k] = ", ".join(str(x) for x in v)
        elif v is not None:
            clean[k] = str(v)
    return clean


# ============================================================
#  Search
# ============================================================

def similarity_search(
    collection_name: str,
    query: str,
    top_k: int = 8,
    where_filter: Optional[dict] = None,
) -> List[Document]:
    """Vector similarity search, returning LangChain Documents."""
    col = get_or_create_collection(collection_name)
    embedder = _get_embeddings()

    query_embedding = embedder.embed_query(query)

    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, col.count() or 1),
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        kwargs["where"] = where_filter

    results = col.query(**kwargs)

    docs: List[Document] = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        meta["_distance"] = dist
        docs.append(Document(page_content=text, metadata=meta))

    return docs
