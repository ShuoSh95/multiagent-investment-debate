"""Bootstrap the knowledge base on hosted deployments (e.g. HF Spaces).

The vector store + BM25 index are NOT in the git repo (they contain
copyrighted book text and weigh ~225MB). For public hosting they live in
a PRIVATE HuggingFace dataset; this module downloads and extracts them
when `data/chroma_db` / `data/bm25_index` are missing.

Requires env:
    HF_TOKEN     — token with read access to the dataset
    KB_DATASET   — dataset repo id (default: vae01/investment-debate-kb)

Usage:
    python -m rag.fetch_kb          # CLI (used by the Space Dockerfile)
    from rag.fetch_kb import ensure_kb; ensure_kb()   # runtime safety net
"""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _ROOT / "data"
_KB_ARCHIVE = "kb.tar.gz"


def kb_present() -> bool:
    return (_DATA_DIR / "chroma_db").is_dir() and (_DATA_DIR / "bm25_index").is_dir()


def ensure_kb() -> bool:
    """Download + extract the KB if missing. Returns True when the KB is
    available (already present, or fetched successfully)."""
    if kb_present():
        return True

    token = os.getenv("HF_TOKEN")
    if not token:
        print("[fetch_kb] KB missing and HF_TOKEN not set — RAG will be empty.")
        return False

    # hf-mirror.com is download-only for public repos; private datasets
    # must go through the official endpoint.
    os.environ.pop("HF_ENDPOINT", None)
    from huggingface_hub import hf_hub_download

    repo_id = os.getenv("KB_DATASET", "vae01/investment-debate-kb")
    print(f"[fetch_kb] downloading {_KB_ARCHIVE} from {repo_id} ...")
    archive = hf_hub_download(
        repo_id=repo_id,
        filename=_KB_ARCHIVE,
        repo_type="dataset",
        token=token,
    )

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[fetch_kb] extracting into {_DATA_DIR} ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(_DATA_DIR)

    ok = kb_present()
    print(f"[fetch_kb] done, kb_present={ok}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if ensure_kb() else 1)
