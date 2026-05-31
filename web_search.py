"""
Real-time web search layer for the Investment Agent.

Priority chain:
  1. Gemini 2.5 native Google-Search grounding — FREE, requires
     GOOGLE_API_KEY. Preferred when the user is on Gemini.
  2. Tavily API — cross-provider fallback, requires TAVILY_API_KEY.
     Free tier: 1000 calls/month.
  3. Graceful no-op — returns empty string and a warning so the
     debate can still run on RAG + market_data only.

Exposes:
    perform_web_search(query, focus="") -> (markdown_text, source_used)
"""

from __future__ import annotations

import os
import threading
from typing import Optional, Tuple

# ------------------------------------------------------------
#  Quota tracking (in-process, resets on program restart)
# ------------------------------------------------------------

_call_counter = {
    "gemini_grounding": 0,
    "tavily": 0,
    "failed": 0,
}
_counter_lock = threading.Lock()

# Soft quotas — warn user when exceeded.
# Gemini free tier: 1500 req/day total; keep grounding under 200/day for safety
# (each debate triggers ~1 call, so 200 = 200 debates/day)
GEMINI_DAILY_SOFT_LIMIT = 200
TAVILY_MONTHLY_SOFT_LIMIT = 1000


def get_call_stats() -> dict:
    with _counter_lock:
        return dict(_call_counter)


# ------------------------------------------------------------
#  Gemini native grounding (preferred, free)
# ------------------------------------------------------------

def _search_via_gemini(query: str, focus: str) -> str:
    """Use Gemini 2.5's built-in Google Search tool to fetch grounded facts.

    Retries transient 503/429 errors and falls through a small list of
    model candidates so the request succeeds even when one snapshot is
    overloaded."""
    import time as _time

    from google import genai
    from google.genai import types

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.startswith("your_"):
        raise RuntimeError("GOOGLE_API_KEY missing for Gemini grounding")

    client = genai.Client(api_key=api_key)
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(
        tools=[grounding_tool],
        temperature=0.2,
    )

    focus_line = f"\n重点关注: {focus}" if focus else ""
    prompt = (
        "你是一位投资研究助理。请针对下列问题，使用谷歌搜索获取"
        "最新的、可信的公开信息，并用简体中文归纳成一份要点式简报。\n"
        f"【问题】{query}{focus_line}\n\n"
        "简报请包含（若能查到）：\n"
        "  1. 最新价格 / 估值水平（PE、PB、市值等）\n"
        "  2. 最近一期财报核心数据 & 业务进展\n"
        "  3. 近期宏观或行业相关事件\n"
        "  4. 市场情绪 / 主流机构观点\n"
        "输出控制在 500 字以内，每条要点前加序号，末尾列出 3-5 个信息来源 URL。"
    )

    primary = os.getenv("WEB_SEARCH_MODEL", "gemini-2.5-flash")
    candidates = [primary]
    for alt in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"):
        if alt not in candidates:
            candidates.append(alt)

    last_err: Optional[Exception] = None
    response = None
    for model_name in candidates:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                last_err = e
                # Only retry on transient server-side errors
                if any(code in msg for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED")):
                    wait = 2 * (attempt + 1)
                    print(
                        f"  [web_search] {model_name} attempt {attempt+1} "
                        f"hit transient error, retrying in {wait}s..."
                    )
                    _time.sleep(wait)
                    continue
                # Non-transient → give up on this model
                break
        if response is not None:
            break

    if response is None:
        raise RuntimeError(
            f"Gemini grounding failed across all models: {last_err}"
        )

    text = response.text or ""

    # Append grounding sources if provided by the API
    try:
        grounding = response.candidates[0].grounding_metadata
        chunks = getattr(grounding, "grounding_chunks", None) or []
        if chunks:
            lines = []
            for ch in chunks[:5]:
                web = getattr(ch, "web", None)
                if web and getattr(web, "uri", None):
                    title = getattr(web, "title", "") or web.uri
                    lines.append(f"  - {title}: {web.uri}")
            if lines and "【信息来源】" not in text:
                text = text.rstrip() + "\n\n【信息来源】\n" + "\n".join(lines)
    except (AttributeError, IndexError):
        pass

    return text.strip()


# ------------------------------------------------------------
#  Tavily fallback (cross-provider)
# ------------------------------------------------------------

def _search_via_tavily(query: str, focus: str) -> str:
    import requests

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY missing")

    full_q = f"{query} {focus}" if focus else query
    resp = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": full_q,
            "search_depth": "advanced",
            "include_answer": True,
            "max_results": 5,
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()

    lines = []
    if data.get("answer"):
        lines.append(f"【核心摘要】{data['answer']}")
    lines.append("\n【检索结果】")
    for i, r in enumerate(data.get("results", [])[:5], 1):
        title = r.get("title", "")
        content = (r.get("content") or "").strip()[:280]
        url = r.get("url", "")
        lines.append(f"{i}. {title}\n   {content}\n   来源: {url}")
    return "\n".join(lines)


# ------------------------------------------------------------
#  Public entry — tries providers in priority order
# ------------------------------------------------------------

def perform_web_search(query: str, focus: str = "") -> Tuple[str, str]:
    """Fetch web-search results for the given query.

    Returns:
        (markdown_text, source_used)  where source_used is one of
        'gemini_grounding' / 'tavily' / 'none'.
    """
    # Priority 1: Gemini grounding (free if user is on Gemini provider)
    if os.getenv("GOOGLE_API_KEY") and not os.getenv("GOOGLE_API_KEY", "").startswith("your_"):
        with _counter_lock:
            used = _call_counter["gemini_grounding"]
        if used >= GEMINI_DAILY_SOFT_LIMIT:
            print(
                f"  [web_search] ⚠️  Gemini grounding calls today: {used} "
                f"(soft limit {GEMINI_DAILY_SOFT_LIMIT}). "
                "Switching to Tavily if available."
            )
        else:
            try:
                text = _search_via_gemini(query, focus)
                with _counter_lock:
                    _call_counter["gemini_grounding"] += 1
                return text, "gemini_grounding"
            except Exception as e:
                print(f"  [web_search] Gemini grounding failed: {e}")
                with _counter_lock:
                    _call_counter["failed"] += 1

    # Priority 2: Tavily
    if os.getenv("TAVILY_API_KEY"):
        with _counter_lock:
            used = _call_counter["tavily"]
        if used >= TAVILY_MONTHLY_SOFT_LIMIT:
            print(
                f"  [web_search] ⚠️  Tavily calls: {used} "
                f"(soft limit {TAVILY_MONTHLY_SOFT_LIMIT})."
            )
        try:
            text = _search_via_tavily(query, focus)
            with _counter_lock:
                _call_counter["tavily"] += 1
            return text, "tavily"
        except Exception as e:
            print(f"  [web_search] Tavily failed: {e}")
            with _counter_lock:
                _call_counter["failed"] += 1

    # Priority 3: graceful degradation
    print(
        "  [web_search] ⚠️  No web search provider available. "
        "Set GOOGLE_API_KEY (Gemini grounding, free) or TAVILY_API_KEY."
    )
    return "", "none"
