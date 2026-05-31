"""
Optional Z-Library integration for downloading books (Greenblatt, Lynch).

Requires Z-Library credentials in .env:
    ZLIB_EMAIL=your@email.com
    ZLIB_PASSWORD=your_password

If credentials are not set, this module is silently skipped.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from rag.config import MASTER_CONFIGS, RAW_DIR


def _check_credentials() -> bool:
    return bool(os.getenv("ZLIB_EMAIL") and os.getenv("ZLIB_PASSWORD"))


def _get_client():
    """Lazy-import and initialize zlibrary-sync client."""
    try:
        from zlibrary_sync import Zlibrary
    except ImportError:
        print("  [WARN] zlibrary-sync not installed. Run: pip install zlibrary-sync diskcache")
        return None

    email = os.getenv("ZLIB_EMAIL", "")
    password = os.getenv("ZLIB_PASSWORD", "")

    if not email or not password:
        print("  [WARN] ZLIB_EMAIL / ZLIB_PASSWORD not set in .env")
        return None

    try:
        client = Zlibrary(email=email, password=password)
        return client
    except Exception as e:
        print(f"  [ERROR] Z-Library login failed: {e}")
        return None


def search_and_download(
    query: str,
    dest_dir: Path,
    extensions: Optional[List[str]] = None,
) -> bool:
    """Search Z-Library and download the first matching result."""
    if extensions is None:
        extensions = ["pdf", "epub"]

    client = _get_client()
    if client is None:
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        results = client.search(query=query, extensions=extensions, count=3)
    except Exception as e:
        print(f"  [ERROR] Z-Library search failed for '{query}': {e}")
        return False

    if not results:
        print(f"  [WARN] No results found for '{query}'")
        return False

    book = results[0]
    title = getattr(book, "title", query)
    ext = getattr(book, "extension", "pdf")
    filename = f"{_slugify(title)}.{ext}"
    dest_path = dest_dir / filename

    if dest_path.exists() and dest_path.stat().st_size > 1000:
        print(f"  Already exists: {filename}")
        return True

    try:
        print(f"  Downloading: {title}")
        content = book.download()
        with open(dest_path, "wb") as f:
            f.write(content)
        print(f"  Saved {filename} ({dest_path.stat().st_size / 1024:.0f} KB)")
        return True
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        return False


def _slugify(text: str, max_len: int = 80) -> str:
    import re
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "_", text).strip("_")
    return text[:max_len]


def acquire_zlib(master_key: str) -> bool:
    """Download Z-Library books for a master (if configured and credentials available)."""
    config = MASTER_CONFIGS.get(master_key)
    if not config:
        return False

    queries = config.get("zlib_queries", [])
    if not queries:
        return True

    if not _check_credentials():
        print(f"  [SKIP] Z-Library credentials not configured for {master_key}")
        print(f"         Set ZLIB_EMAIL and ZLIB_PASSWORD in .env, or place PDFs in data/raw/{master_key}/")
        return True

    dest_dir = RAW_DIR / master_key
    success = True
    for query in queries:
        if not search_and_download(query, dest_dir):
            success = False

    return success


def acquire_all_zlib() -> dict:
    """Download Z-Library books for all masters that need them."""
    results = {}
    for key, config in MASTER_CONFIGS.items():
        if config.get("zlib_queries"):
            print(f"\n[Z-Library] Acquiring data for {config['display_name']}...")
            results[key] = acquire_zlib(key)
        else:
            results[key] = True
    return results
