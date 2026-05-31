"""
CLI entry point for building / rebuilding the RAG knowledge base.

Usage:
    python -m rag.build_kb --master buffett
    python -m rag.build_kb --master all
    python -m rag.build_kb --master all --rebuild
    python -m rag.build_kb --master marks --acquire-only
    python -m rag.build_kb --master all --skip-acquire
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rag.config import MASTER_CONFIGS, MASTER_KEYS, RAW_DIR
from rag.loader import load_directory, load_file
from rag.chunker import chunk_documents
from rag.vectorstore import upsert_chunks, drop_collection, collection_count
from rag.retriever import build_bm25_index


# ============================================================
#  Acquire step — download raw data
# ============================================================

def _acquire(master_key: str) -> bool:
    """Run all applicable scrapers for a master."""
    from scrapers.github_sources import acquire_github
    from scrapers.official_sites import acquire_official

    config = MASTER_CONFIGS[master_key]
    display = config["display_name"]
    print(f"\n{'='*56}")
    print(f"  📥 Acquiring data for {display} ({master_key})")
    print(f"{'='*56}")

    ok = True

    if config.get("github_sources"):
        print(f"\n  [Source: GitHub]")
        if not acquire_github(master_key):
            ok = False

    if config.get("official_sources"):
        print(f"\n  [Source: Official Sites]")
        if not acquire_official(master_key):
            ok = False

    raw_dir = RAW_DIR / master_key
    file_count = len(list(raw_dir.glob("*"))) if raw_dir.exists() else 0
    print(f"\n  📂 Files in data/raw/{master_key}/: {file_count}")

    if file_count == 0:
        print(f"  ⚠️  No data files found. Place PDFs/TXT manually in data/raw/{master_key}/")

    return ok


# ============================================================
#  Process step — load, chunk, embed, index
# ============================================================

def _process(master_key: str, rebuild: bool = False) -> bool:
    """Load, chunk, embed and index a master's knowledge base."""
    config = MASTER_CONFIGS[master_key]
    display = config["display_name"]
    collection = config["collection_name"]
    raw_dir = RAW_DIR / master_key

    print(f"\n{'='*56}")
    print(f"  🔧 Processing knowledge base for {display}")
    print(f"{'='*56}")

    if rebuild:
        print(f"  🗑️  Dropping existing collection '{collection}'...")
        drop_collection(collection)

    existing = collection_count(collection)
    if existing > 0 and not rebuild:
        print(f"  ℹ️  Collection '{collection}' already has {existing} chunks.")
        print(f"      Use --rebuild to recreate from scratch.")
        return True

    # Load
    print(f"\n  📄 Loading documents from {raw_dir}...")
    source_type = _infer_default_source_type(master_key)
    docs = load_directory(raw_dir, master_key, source_type)

    if not docs:
        print(f"  ⚠️  No documents loaded. Skipping {master_key}.")
        return False

    total_chars = sum(len(d.page_content) for d in docs)
    print(f"     Loaded {len(docs)} documents ({total_chars:,} characters)")

    # Chunk
    print(f"\n  ✂️  Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"     Produced {len(chunks)} chunks")

    if not chunks:
        print(f"  ⚠️  No chunks produced. Check document content.")
        return False

    # Embed + upsert
    print(f"\n  🧠 Embedding and upserting into ChromaDB '{collection}'...")
    t0 = time.time()
    count = upsert_chunks(collection, chunks)
    elapsed = time.time() - t0
    print(f"     Done: {count} chunks in {elapsed:.1f}s")

    # BM25 index
    print(f"\n  📊 Building BM25 index...")
    build_bm25_index(master_key, chunks)
    print(f"     BM25 index saved")

    final = collection_count(collection)
    print(f"\n  ✅ {display} knowledge base ready: {final} chunks in ChromaDB")

    return True


def _infer_default_source_type(master_key: str) -> str:
    """Default source_type assumed for files whose specific source_type
    is not otherwise tagged (fallback for mixed directories)."""
    type_map = {
        "buffett": "letter",  # all .md are shareholder letters -> Tier 1
        "dalio": "book",  # books + articles (html treated as article via ext check below)
        "marks": "memo",  # all .pdf are Oaktree memos -> Tier 1
        # For Greenblatt/Lynch, we currently only have Wikipedia articles
        # (Tier 2). When the user drops actual books into data/raw/*,
        # those files should be named with e.g. "book_*.pdf" and we'd need
        # per-file tagging -- acceptable limitation for v1.
        "greenblatt": "article",
        "lynch": "article",
    }
    return type_map.get(master_key, "book")


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build RAG knowledge base for investment masters"
    )
    parser.add_argument(
        "--master",
        required=True,
        help=f"Master key ({', '.join(MASTER_KEYS)}) or 'all'",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop and recreate the collection from scratch",
    )
    parser.add_argument(
        "--acquire-only",
        action="store_true",
        help="Only download raw data, don't process/embed",
    )
    parser.add_argument(
        "--skip-acquire",
        action="store_true",
        help="Skip data acquisition, process existing files only",
    )
    args = parser.parse_args()

    masters = MASTER_KEYS if args.master == "all" else [args.master]

    for key in masters:
        if key not in MASTER_CONFIGS:
            print(f"[ERROR] Unknown master: {key}")
            print(f"        Available: {', '.join(MASTER_KEYS)}")
            sys.exit(1)

    print(f"\n{'━'*56}")
    print(f"  🏛️  RAG Knowledge Base Builder")
    print(f"  Masters: {', '.join(masters)}")
    print(f"  Mode: {'acquire-only' if args.acquire_only else 'skip-acquire' if args.skip_acquire else 'full pipeline'}")
    if args.rebuild:
        print(f"  ⚠️  REBUILD mode: existing data will be dropped")
    print(f"{'━'*56}")

    overall_start = time.time()
    results = {}

    for key in masters:
        master_ok = True

        try:
            if not args.skip_acquire:
                if not _acquire(key):
                    master_ok = False

            if not args.acquire_only:
                if not _process(key, rebuild=args.rebuild):
                    master_ok = False
        except Exception as e:
            import traceback
            print(f"\n  [ERROR] Unhandled exception processing {key}: {e}")
            traceback.print_exc()
            print(f"  Continuing to next master...")
            master_ok = False

        results[key] = master_ok

    # Summary
    elapsed = time.time() - overall_start
    print(f"\n{'━'*56}")
    print(f"  📋 Build Summary ({elapsed:.1f}s)")
    print(f"{'━'*56}")
    for key, ok in results.items():
        display = MASTER_CONFIGS[key]["display_name"]
        status = "✅ OK" if ok else "⚠️  Incomplete"
        count = collection_count(MASTER_CONFIGS[key]["collection_name"])
        print(f"  {display}: {status} ({count} chunks)")

    print()


if __name__ == "__main__":
    main()
