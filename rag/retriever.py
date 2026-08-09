"""
Hybrid retriever: BM25 + Vector search with RRF fusion and credibility weighting.

Usage:
    from rag.retriever import retrieve
    results = retrieve("marks", "What does Howard Marks think about market cycles?")
"""

from __future__ import annotations

import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from rank_bm25 import BM25Okapi

from rag.config import (
    BM25_DIR,
    BM25_TOP_K,
    FINAL_TOP_K,
    MASTER_CONFIGS,
    RRF_K,
    SELF_QUERY_MODEL,
    SELF_QUERY_TEMPERATURE,
    TIER_BOOST,
    VECTOR_TOP_K,
)


def _bm25_only() -> bool:
    """Hosted demo skips the local BGE-M3 model (too heavy for ~2.7GB free
    Cloud RAM). EMBEDDING_PROVIDER=bm25|none|off also forces this path."""
    provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    if provider in ("bm25", "none", "off"):
        return True
    try:
        from llm_provider import is_demo_mode
        return is_demo_mode()
    except Exception:
        return False


# ============================================================
#  BM25 index management
# ============================================================

def _bm25_path(master_key: str) -> Path:
    return BM25_DIR / f"{master_key}.pkl"


def build_bm25_index(master_key: str, chunks: List[Document]) -> None:
    """Build and pickle a BM25 index for the given master."""
    BM25_DIR.mkdir(parents=True, exist_ok=True)
    corpus = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(corpus)
    payload = {"bm25": bm25, "docs": chunks}
    with open(_bm25_path(master_key), "wb") as f:
        pickle.dump(payload, f)


def _load_bm25(master_key: str) -> Optional[dict]:
    path = _bm25_path(master_key)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def bm25_search(master_key: str, query: str, top_k: int = BM25_TOP_K) -> List[Document]:
    """Keyword-based BM25 search over the master's corpus."""
    data = _load_bm25(master_key)
    if data is None:
        return []

    bm25: BM25Okapi = data["bm25"]
    docs: List[Document] = data["docs"]
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for idx, score in ranked:
        if score > 0:
            doc = docs[idx]
            doc.metadata["_bm25_score"] = float(score)
            results.append(doc)
    return results


# ============================================================
#  Self-Query rewriting
# ============================================================

_SELF_QUERY_PROMPT = (
    "You are a query rewriter for a RAG system. The knowledge base contains "
    "English-language original texts by the investment master {master_name}.\n\n"
    "Given the user's investment question (possibly in Chinese), rewrite it as a "
    "concise English search query that would match relevant passages in this master's "
    "writings. Focus on the master's key concepts and terminology.\n\n"
    "Output ONLY the rewritten query, nothing else."
)


def self_query_rewrite(master_key: str, user_query: str) -> str:
    """Rewrite user query into the master's conceptual language for better retrieval."""
    from langchain_openai import ChatOpenAI

    config = MASTER_CONFIGS.get(master_key, {})
    master_name = config.get("display_name", master_key)

    llm = ChatOpenAI(
        model=SELF_QUERY_MODEL,
        temperature=SELF_QUERY_TEMPERATURE,
        max_retries=2,
    )
    messages = [
        SystemMessage(content=_SELF_QUERY_PROMPT.format(master_name=master_name)),
        HumanMessage(content=f"User question: {user_query}"),
    ]
    return llm.invoke(messages).content.strip()


# ============================================================
#  Reciprocal Rank Fusion with credibility weighting
# ============================================================

def _rrf_fuse(
    result_lists: List[List[Document]],
    k: int = RRF_K,
) -> List[Document]:
    """Fuse multiple ranked lists using RRF, then apply credibility tier boost."""
    scores: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, Document] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_key = doc.page_content[:200]
            scores[doc_key] += 1.0 / (k + rank + 1)
            doc_map[doc_key] = doc

    for doc_key, doc in doc_map.items():
        tier = doc.metadata.get("credibility_tier", 2)
        boost = TIER_BOOST.get(tier, 1.0)
        scores[doc_key] *= boost

    ranked_keys = sorted(scores, key=scores.get, reverse=True)
    return [doc_map[k] for k in ranked_keys]


# ============================================================
#  Public API
# ============================================================

def retrieve(
    master_key: str,
    user_query: str,
    top_k: int = FINAL_TOP_K,
    use_self_query: bool = True,
) -> List[Document]:
    """
    Full retrieval pipeline:
      1. Self-Query rewrite (optional)
      2. Parallel Vector + BM25 search
      3. RRF fusion with credibility weighting
      4. Return top-k

    In DEMO_MODE / EMBEDDING_PROVIDER=bm25: BM25-only (no local embedding model).
    """
    if _bm25_only():
        return bm25_search(master_key, user_query, top_k=top_k)[:top_k]

    from rag.vectorstore import similarity_search

    collection_name = MASTER_CONFIGS[master_key]["collection_name"]

    search_query = user_query
    if use_self_query:
        try:
            search_query = self_query_rewrite(master_key, user_query)
        except Exception:
            search_query = user_query

    vector_results = similarity_search(
        collection_name, search_query, top_k=VECTOR_TOP_K
    )
    bm25_results = bm25_search(master_key, search_query, top_k=BM25_TOP_K)

    fused = _rrf_fuse([vector_results, bm25_results])
    return fused[:top_k]


def format_retrieved_context(docs: List[Document]) -> str:
    """Format retrieved docs into a prompt-injectable string."""
    if not docs:
        return "（知识库中暂无相关参考内容）"

    lines: list[str] = []
    for i, doc in enumerate(docs, 1):
        tier = doc.metadata.get("credibility_tier", 2)
        source = doc.metadata.get("filename", "unknown")
        year = doc.metadata.get("year", "")
        tier_label = f"Tier {tier}"
        source_label = f"《{source}》"
        if year:
            source_label += f" ({year})"

        snippet = doc.page_content[:500]
        if len(doc.page_content) > 500:
            snippet += "..."

        lines.append(f"[{tier_label}] {source_label}:\n  \"{snippet}\"")

    return "\n".join(lines)
