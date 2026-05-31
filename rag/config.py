"""
Central configuration for the RAG knowledge-base pipeline.
"""

import os
from pathlib import Path

# ============================================================
#  Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma_db"
BM25_DIR = DATA_DIR / "bm25_index"

# ============================================================
#  Embedding
# ============================================================
# Default is local BGE-M3 -- no API cost, offline-capable, strong cn/en
# cross-lingual. Override per-run via env var EMBEDDING_PROVIDER=openai
# if you want to use OpenAI text-embedding-3-large instead.

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()

# Local (BGE-M3 via sentence-transformers)
LOCAL_EMBEDDING_MODEL = "BAAI/bge-m3"
LOCAL_EMBEDDING_DIM = 1024

# OpenAI
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI_EMBEDDING_DIM = 3072

# Back-compat shims (some legacy code still references these)
EMBEDDING_MODEL = (
    OPENAI_EMBEDDING_MODEL if EMBEDDING_PROVIDER == "openai" else LOCAL_EMBEDDING_MODEL
)
EMBEDDING_DIMENSIONS = (
    OPENAI_EMBEDDING_DIM if EMBEDDING_PROVIDER == "openai" else LOCAL_EMBEDDING_DIM
)

# ============================================================
#  Chunking
# ============================================================

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", ". ", " "]

# ============================================================
#  Credibility Tier System
# ============================================================

TIER_1_TYPES = {"book", "letter", "memo", "whitepaper"}
TIER_2_TYPES = {"speech", "interview", "qa", "article"}
TIER_3_TYPES = {"commentary", "summary", "notes"}

TIER_BOOST = {
    1: 1.5,
    2: 1.0,
    3: 0.5,
}


def credibility_tier_for(source_type: str) -> int:
    source_type = source_type.lower().strip()
    if source_type in TIER_1_TYPES:
        return 1
    if source_type in TIER_2_TYPES:
        return 2
    return 3


# ============================================================
#  Master-specific configuration
# ============================================================

MASTER_CONFIGS = {
    "buffett": {
        "display_name": "沃伦·巴菲特",
        "collection_name": "buffett",
        "github_sources": [
            {
                "repo": "ReeceHarding/buffett-letters",
                "target_subdir": "",
                "source_type": "letter",
            }
        ],
        "official_sources": [],
        "zlib_queries": [],
    },
    "dalio": {
        "display_name": "瑞·达利欧",
        "collection_name": "dalio",
        # Target the pre-extracted TXT/Markdown output directory to avoid
        # downloading huge raw PDFs (the repo's Output/ folder contains clean
        # text versions of all books + a Complete_Collection.md).
        "github_sources": [
            {
                "repo": "Albert-Lsk/ray-dalio-project",
                "target_subdir": "Output",
                "source_type": "book",
            },
            {
                "repo": "Albert-Lsk/ray-dalio-project",
                "target_subdir": "Speeches",
                "source_type": "speech",
            },
            {
                "repo": "Albert-Lsk/ray-dalio-project",
                "target_subdir": "Quotes",
                "source_type": "interview",
            },
        ],
        "official_sources": [
            {
                "url": "https://static.klipfolio.com/ebook/bridgewater-associates-ray-dalio-principles.pdf",
                "filename": "bridgewater_principles_original.pdf",
                "source_type": "whitepaper",
            }
        ],
        "zlib_queries": [],
    },
    "marks": {
        "display_name": "霍华德·马克斯",
        "collection_name": "marks",
        "github_sources": [
            {
                "repo": "anrosent/awesome-memos",
                "target_subdir": "Howard Marks",
                "source_type": "memo",
            }
        ],
        "official_sources": [
            {
                "url": "https://www.oaktreecapital.com/docs/default-source/memos/the-complete-collection.pdf",
                "filename": "marks_complete_memos.pdf",
                "source_type": "memo",
            }
        ],
        "zlib_queries": [],
    },
    "greenblatt": {
        "display_name": "乔尔·格林布拉特",
        "collection_name": "greenblatt",
        "github_sources": [],
        # Fallback Tier-2 sources: Wikipedia encyclopedic articles about the
        # author and his two most-cited books. Used because the books
        # themselves are copyrighted and not available on GitHub.
        "official_sources": [
            {
                "url": "https://en.wikipedia.org/api/rest_v1/page/html/Joel_Greenblatt",
                "filename": "wikipedia_joel_greenblatt.html",
                "source_type": "article",
            },
            {
                "url": "https://en.wikipedia.org/api/rest_v1/page/html/The_Little_Book_That_Beats_the_Market",
                "filename": "wikipedia_little_book_beats_market.html",
                "source_type": "article",
            },
            {
                "url": "https://en.wikipedia.org/api/rest_v1/page/html/Magic_formula_investing",
                "filename": "wikipedia_magic_formula_investing.html",
                "source_type": "article",
            },
            {
                "url": "https://en.wikipedia.org/api/rest_v1/page/html/You_Can_Be_a_Stock_Market_Genius",
                "filename": "wikipedia_stock_market_genius.html",
                "source_type": "article",
            },
        ],
        "zlib_queries": [
            "The Little Book That Beats the Market Joel Greenblatt",
            "You Can Be a Stock Market Genius Joel Greenblatt",
        ],
    },
    "lynch": {
        "display_name": "彼得·林奇",
        "collection_name": "lynch",
        "github_sources": [],
        "official_sources": [
            {
                "url": "https://en.wikipedia.org/api/rest_v1/page/html/Peter_Lynch",
                "filename": "wikipedia_peter_lynch.html",
                "source_type": "article",
            },
            {
                "url": "https://en.wikipedia.org/api/rest_v1/page/html/One_Up_on_Wall_Street",
                "filename": "wikipedia_one_up_wall_street.html",
                "source_type": "article",
            },
            {
                "url": "https://en.wikipedia.org/api/rest_v1/page/html/Beating_the_Street",
                "filename": "wikipedia_beating_the_street.html",
                "source_type": "article",
            },
        ],
        "zlib_queries": [
            "One Up on Wall Street Peter Lynch",
            "Beating the Street Peter Lynch",
        ],
    },
}

MASTER_KEYS = list(MASTER_CONFIGS.keys())

# ============================================================
#  Retrieval
# ============================================================

VECTOR_TOP_K = 8
BM25_TOP_K = 8
FINAL_TOP_K = 5
RRF_K = 60  # standard RRF constant

# ============================================================
#  Self-Query LLM
# ============================================================

SELF_QUERY_MODEL = "gpt-4o-mini"
SELF_QUERY_TEMPERATURE = 0.0
