"""
Semantic chunking with automatic metadata tagging.

Produces chunks with metadata:
  master, source, filename, source_type, year, credibility_tier
"""

from __future__ import annotations

import re
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    SEPARATORS,
    credibility_tier_for,
)


# ============================================================
#  Year extraction heuristics
# ============================================================

_YEAR_IN_FILENAME = re.compile(r"((?:19|20)\d{2})")
_YEAR_IN_TEXT = re.compile(
    r"(?:annual\s+report|letter\s+to|shareholder|chairman)\s*.*?((?:19|20)\d{2})",
    re.IGNORECASE,
)


def _extract_year(doc: Document) -> Optional[str]:
    filename = doc.metadata.get("filename", "")
    m = _YEAR_IN_FILENAME.search(filename)
    if m:
        return m.group(1)

    first_500 = doc.page_content[:500]
    m = _YEAR_IN_TEXT.search(first_500)
    if m:
        return m.group(1)

    return None


# ============================================================
#  Public API
# ============================================================

def chunk_document(doc: Document) -> List[Document]:
    """Split one large Document into smaller chunks with enriched metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        keep_separator=True,
    )

    chunks = splitter.split_documents([doc])

    year = _extract_year(doc)
    source_type = doc.metadata.get("source_type", "book")
    tier = credibility_tier_for(source_type)

    for chunk in chunks:
        chunk.metadata["credibility_tier"] = tier
        if year:
            chunk.metadata["year"] = year

    return chunks


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Chunk a list of Documents, returning a flat list of all chunks."""
    all_chunks: List[Document] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
