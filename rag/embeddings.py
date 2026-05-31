"""
Unified embedding interface — switch between OpenAI and local BGE-M3
via the EMBEDDING_PROVIDER env var (or rag.config.EMBEDDING_PROVIDER).

Both backends expose the same two methods:
    embed_documents(texts: list[str]) -> list[list[float]]
    embed_query(text: str) -> list[float]

BGE-M3 runs locally via sentence-transformers. Model file (~2.3GB) is
downloaded on first use to ~/.cache/huggingface/. Subsequent runs load
from cache, no network needed.
"""

from __future__ import annotations

import os
from typing import List

from rag.config import (
    EMBEDDING_PROVIDER,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_DIM,
    LOCAL_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_DIM,
)


# ============================================================
#  Singleton cache — load model once per process
# ============================================================

_cached_local_model = None
_cached_openai = None

# Guarantees only ONE thread actually loads the model; others wait and
# reuse the cached instance. LangGraph fans out 5 masters in parallel,
# which previously triggered 5 concurrent first-loads and caused a
# PyTorch meta-tensor race ("Cannot copy out of meta tensor").
import threading as _threading
_local_model_lock = _threading.Lock()


def _get_local_model():
    """Load BGE-M3 (or configured local model) once and cache it.

    Robust to:
      - Concurrent first-call from multiple threads (LangGraph parallel
        master nodes) — serialized via a module-level lock.
      - PyTorch "meta tensor" load path on torch>=2.5 + transformers
        newer snapshots — we force CPU load first, then move to the
        best available device with to_empty()-safe weight copy.
    """
    global _cached_local_model
    # Fast-path: already loaded, no lock needed
    if _cached_local_model is not None:
        return _cached_local_model

    with _local_model_lock:
        # Re-check inside the lock (another thread may have loaded while
        # we waited)
        if _cached_local_model is not None:
            return _cached_local_model

        # Use HF mirror in China if the direct endpoint is blocked.
        if not os.getenv("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        from sentence_transformers import SentenceTransformer
        import torch

        print(f"  [embeddings] Loading local model '{LOCAL_EMBEDDING_MODEL}' ...")

        # Pick target device. MPS on Apple Silicon, CUDA on GPU, else CPU.
        if torch.backends.mps.is_available():
            target_device = "mps"
        elif torch.cuda.is_available():
            target_device = "cuda"
        else:
            target_device = "cpu"

        try:
            # Preferred path: let sentence-transformers handle device
            # placement. Works for most environments.
            model = SentenceTransformer(
                LOCAL_EMBEDDING_MODEL,
                device=target_device,
            )
        except (NotImplementedError, RuntimeError) as e:
            # Meta-tensor fallback: load on CPU first (no meta path), then
            # move weights. Avoids torch.nn.Module.to() on meta tensors.
            msg = str(e).lower()
            if "meta" not in msg and "to_empty" not in msg:
                raise
            print(
                f"  [embeddings] meta-tensor load path detected, "
                f"falling back to CPU-first load + move to {target_device}"
            )
            model = SentenceTransformer(LOCAL_EMBEDDING_MODEL, device="cpu")
            if target_device != "cpu":
                try:
                    # sentence-transformers wraps a nn.Module that we can
                    # .to() cleanly because it's already materialized on CPU
                    model = model.to(target_device)
                except Exception as move_err:  # noqa: BLE001
                    print(
                        f"  [embeddings] move to {target_device} failed "
                        f"({move_err}); staying on CPU."
                    )

        _cached_local_model = model
        try:
            print(f"  [embeddings] Model loaded on device: {model.device}")
        except Exception:
            pass
        return _cached_local_model


def warmup_local_model() -> None:
    """Preload the embedding model (call once at app startup before any
    parallel RAG calls to avoid concurrent first-load)."""
    _get_local_model()


def _get_openai_embeddings():
    global _cached_openai
    if _cached_openai is not None:
        return _cached_openai

    from langchain_openai import OpenAIEmbeddings

    _cached_openai = OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,
        dimensions=OPENAI_EMBEDDING_DIM,
    )
    return _cached_openai


# ============================================================
#  Public API
# ============================================================

class EmbeddingBackend:
    """Thin wrapper exposing a unified two-method interface."""

    def __init__(self, provider: str):
        self.provider = provider

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "openai":
            return _get_openai_embeddings().embed_documents(texts)
        # default: local
        model = _get_local_model()
        # BGE-M3 recommends normalize_embeddings=True for cosine similarity
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=16,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        if self.provider == "openai":
            return _get_openai_embeddings().embed_query(text)
        model = _get_local_model()
        vector = model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0]
        return vector.tolist()


def get_embedder() -> EmbeddingBackend:
    """Return the configured embedding backend (respects env override)."""
    # Allow runtime override via env, falling back to config default
    provider = os.getenv("EMBEDDING_PROVIDER", EMBEDDING_PROVIDER).lower()
    if provider not in {"openai", "local"}:
        print(f"  [embeddings] Unknown provider '{provider}', defaulting to 'local'")
        provider = "local"
    return EmbeddingBackend(provider)


def current_embedding_dim() -> int:
    """Return vector dimension for the configured backend."""
    provider = os.getenv("EMBEDDING_PROVIDER", EMBEDDING_PROVIDER).lower()
    if provider == "openai":
        return OPENAI_EMBEDDING_DIM
    return LOCAL_EMBEDDING_DIM
