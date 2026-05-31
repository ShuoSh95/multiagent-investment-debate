"""
Download data from official websites (Oaktree memos PDF, Bridgewater Principles, etc.).

Used as primary source for Marks, and fallback/supplement for others.
"""

from __future__ import annotations

import time
from pathlib import Path

import requests

from rag.config import MASTER_CONFIGS, RAW_DIR


def _download_file(url: str, dest: Path, timeout: int = 180) -> bool:
    """Download a file from URL to dest path with retries. Skips if already exists."""
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  Already exists: {dest.name} ({dest.stat().st_size / 1024:.0f} KB)")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url} ...")

    for attempt in range(3):
        try:
            resp = requests.get(
                url,
                timeout=timeout,
                stream=True,
                headers={"User-Agent": "Mozilla/5.0 InvestmentAgent/1.0"},
            )
            if resp.status_code == 404:
                print(f"    [WARN] 404 Not Found")
                return False
            resp.raise_for_status()

            total = 0
            last_report = 0
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    f.write(chunk)
                    total += len(chunk)
                    if total - last_report > 1024 * 1024:
                        print(f"    ... {total / 1024 / 1024:.1f} MB")
                        last_report = total

            tmp.replace(dest)
            print(f"    ✅ Saved {dest.name} ({total / 1024:.0f} KB)")
            return True

        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
            ConnectionResetError,
        ) as e:
            print(f"    [WARN] Attempt {attempt + 1} failed: {type(e).__name__}")
            if attempt < 2:
                wait = 2 ** (attempt + 1)
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
        except requests.RequestException as e:
            print(f"  [ERROR] Download failed: {e}")
            return False

    print(f"  [ERROR] All attempts failed for {url}")
    return False


def acquire_official(master_key: str) -> bool:
    """Download all official-site sources for a master."""
    config = MASTER_CONFIGS.get(master_key)
    if not config:
        print(f"  [WARN] No config for master '{master_key}'")
        return False

    sources = config.get("official_sources", [])
    if not sources:
        print(f"  No official sources configured for {master_key}")
        return True

    dest_dir = RAW_DIR / master_key
    success = True
    for src in sources:
        url = src["url"]
        filename = src["filename"]
        dest = dest_dir / filename
        if not _download_file(url, dest):
            success = False

    return success


def acquire_all_official() -> dict:
    """Download official-site sources for all masters that have them."""
    results = {}
    for key, config in MASTER_CONFIGS.items():
        if config.get("official_sources"):
            print(f"\n[Official] Acquiring data for {config['display_name']}...")
            results[key] = acquire_official(key)
        else:
            results[key] = True
    return results
