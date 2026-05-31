"""
Download curated datasets from GitHub repositories.

Strategy (from experience with unreliable networks):
1. Enumerate files via GitHub API (`api.github.com/repos/.../git/trees/<branch>?recursive=1`)
2. Download each file individually from a list of CDN mirrors, trying in order until one succeeds:
   - raw.githubusercontent.com (primary)
   - cdn.jsdelivr.net (CDN fallback, often works when raw is flaky)
   - ghproxy / gh-proxy mirrors (China-accessible fallbacks)
3. Per-file retries with exponential backoff.
4. Skip files already present on disk (resume-friendly).
5. One failed file does not abort the rest.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Iterable, List, Optional

import requests

from rag.config import MASTER_CONFIGS, RAW_DIR


ALLOWED_EXTS = {".md", ".txt", ".pdf", ".html", ".htm", ".epub", ".csv", ".rst"}

# CDN fallbacks — each takes (owner, repo, branch, path) and returns a URL.
CDN_BUILDERS = [
    lambda o, r, b, p: f"https://raw.githubusercontent.com/{o}/{r}/{b}/{p}",
    lambda o, r, b, p: f"https://cdn.jsdelivr.net/gh/{o}/{r}@{b}/{p}",
    lambda o, r, b, p: f"https://ghproxy.net/https://raw.githubusercontent.com/{o}/{r}/{b}/{p}",
    lambda o, r, b, p: f"https://gh-proxy.com/https://raw.githubusercontent.com/{o}/{r}/{b}/{p}",
]

REQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 InvestmentAgent/1.0",
    "Accept": "*/*",
}


# ============================================================
#  GitHub API: list tree
# ============================================================

def _get_default_branch(owner: str, repo: str) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        resp = requests.get(url, timeout=30, headers=REQ_HEADERS)
        if resp.status_code == 200:
            return resp.json().get("default_branch") or "main"
    except requests.RequestException:
        pass
    return "main"


def _list_repo_tree(owner: str, repo: str, branch: str) -> Optional[List[dict]]:
    """Return flat list of file blobs {path, size} or None on failure."""
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=45, headers=REQ_HEADERS)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("truncated"):
                    print(f"    [WARN] Tree listing for {repo} was truncated by API")
                blobs = [
                    {"path": e["path"], "size": e.get("size", 0)}
                    for e in data.get("tree", [])
                    if e.get("type") == "blob"
                ]
                return blobs
            elif resp.status_code == 403:
                print(f"    [WARN] API rate limited: {resp.headers.get('X-RateLimit-Remaining')}")
                return None
            else:
                print(f"    [WARN] API HTTP {resp.status_code}")
        except requests.RequestException as e:
            print(f"    [WARN] API attempt {attempt + 1} failed: {type(e).__name__}")
        if attempt < 2:
            time.sleep(2 ** (attempt + 1))
    return None


# ============================================================
#  File download with multi-CDN fallback
# ============================================================

def _download_file(owner: str, repo: str, branch: str, path: str, dest: Path,
                   expected_size: int = 0) -> bool:
    """Try each CDN until one delivers the full file. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 0:
        if expected_size == 0 or abs(dest.stat().st_size - expected_size) < 1024:
            return True  # already downloaded

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    for cdn_idx, build_url in enumerate(CDN_BUILDERS):
        url = build_url(owner, repo, branch, requests.utils.quote(path, safe="/"))
        for attempt in range(2):
            try:
                with requests.get(url, timeout=90, stream=True, headers=REQ_HEADERS) as resp:
                    if resp.status_code == 404:
                        break  # try next CDN
                    if resp.status_code != 200:
                        break
                    total = 0
                    with open(tmp, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=65536):
                            if not chunk:
                                continue
                            f.write(chunk)
                            total += len(chunk)
                    # verify size roughly
                    if expected_size > 0 and total < expected_size * 0.5:
                        tmp.unlink(missing_ok=True)
                        break
                    tmp.replace(dest)
                    return True
            except (requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ReadTimeout,
                    ConnectionResetError) as e:
                if attempt == 0:
                    time.sleep(2)
                continue
            except Exception as e:
                print(f"        [ERROR] {type(e).__name__}: {e}")
                break
        # next CDN
    tmp.unlink(missing_ok=True)
    return False


# ============================================================
#  Path filtering
# ============================================================

def _filter_files(blobs: List[dict], target_subdir: str) -> List[dict]:
    """Keep only files whose extension is allowed and path matches subdir."""
    prefix = target_subdir.strip("/")
    kept = []
    for b in blobs:
        path = b["path"]
        ext = Path(path).suffix.lower()
        if ext not in ALLOWED_EXTS:
            continue
        if prefix and not (path == prefix or path.startswith(prefix + "/")):
            continue
        kept.append(b)
    return kept


def _safe_filename(path: str) -> str:
    """Flatten a subpath into a single safe filename."""
    # Preserve extension, replace separators
    name = path.replace("/", "__").replace("\\", "__")
    name = re.sub(r"[^\w\-\.\(\),]", "_", name)
    return name[:200]


# ============================================================
#  Public API
# ============================================================

def acquire_github(master_key: str) -> bool:
    """Download all configured GitHub sources for a master."""
    config = MASTER_CONFIGS.get(master_key)
    if not config:
        print(f"  [WARN] No config for master '{master_key}'")
        return False

    sources = config.get("github_sources", [])
    if not sources:
        print(f"  No GitHub sources configured for {master_key}")
        return True

    dest_dir = RAW_DIR / master_key
    dest_dir.mkdir(parents=True, exist_ok=True)

    overall_ok = True

    for src in sources:
        repo_full = src["repo"]
        target_subdir = src.get("target_subdir", "")
        owner, repo = repo_full.split("/", 1)

        print(f"\n  Repo: {repo_full}")
        if target_subdir:
            print(f"    Subdir filter: '{target_subdir}'")

        branch = _get_default_branch(owner, repo)
        print(f"    Default branch: {branch}")

        blobs = _list_repo_tree(owner, repo, branch)
        if not blobs:
            print(f"    [ERROR] Could not list repo tree via API")
            overall_ok = False
            continue

        kept = _filter_files(blobs, target_subdir)
        total_size = sum(b["size"] for b in kept)
        print(f"    Found {len(kept)} matching files ({total_size / 1024:.0f} KB total)")

        if not kept:
            overall_ok = False
            continue

        ok_count = 0
        fail_count = 0
        for i, b in enumerate(kept):
            path = b["path"]
            fname = _safe_filename(path)
            dest = dest_dir / fname

            if dest.exists() and dest.stat().st_size > 0:
                if b["size"] == 0 or abs(dest.stat().st_size - b["size"]) < 1024:
                    ok_count += 1
                    continue

            print(f"    [{i + 1}/{len(kept)}] {path} ({b['size'] / 1024:.0f} KB)...", flush=True)
            if _download_file(owner, repo, branch, path, dest, b["size"]):
                ok_count += 1
            else:
                fail_count += 1
                print(f"         ❌ failed")

        print(f"    ✅ {ok_count}/{len(kept)} files OK ({fail_count} failed)")
        if fail_count > 0:
            overall_ok = False

    return overall_ok


def acquire_all_github() -> dict:
    """Download GitHub sources for all masters that have them configured."""
    results = {}
    for key, config in MASTER_CONFIGS.items():
        if config.get("github_sources"):
            print(f"\n[GitHub] Acquiring data for {config['display_name']}...")
            results[key] = acquire_github(key)
        else:
            results[key] = True
    return results
