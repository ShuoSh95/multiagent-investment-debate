"""Rate limiting for the public demo deployment.

Three layers (all only enforced when DEMO_MODE=1):
  1. Global daily cap    — total debates per calendar day (counted from
                           the debates table; resets on Space restart,
                           which is acceptable for a soft limit).
  2. Per-session cap     — debates started by one browser session.
  3. Concurrency slot    — only ONE live debate at a time. The free
                           Gemini tier allows ~10 requests/min; two
                           concurrent debates would trip each other.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Tuple

from web import history_db

_slot_lock = threading.Lock()
_slot = {"active": False, "since": 0.0}

# A debate that has "held the slot" longer than this is considered dead
# (browser closed mid-run, thread killed) and the slot is reclaimed.
_STALE_AFTER_S = 30 * 60


def daily_limit() -> int:
    return int(os.getenv("DEMO_DAILY_DEBATE_LIMIT", "30"))


def session_limit() -> int:
    return int(os.getenv("DEMO_SESSION_DEBATE_LIMIT", "2"))


def check_quota(session_started: int) -> Tuple[bool, str]:
    """Pre-flight check before starting a debate. Returns (ok, reason)."""
    if session_started >= session_limit():
        return False, (
            f"您本次访问已发起 {session_started} 场辩论（上限 {session_limit()} 场）。"
            "欢迎围观侧边栏的精选辩论回放，或明天再来！"
        )
    used = history_db.count_debates_today()
    if used >= daily_limit():
        return False, (
            f"今日公共辩论额度（{daily_limit()} 场）已用完。"
            "您仍可在侧边栏围观精选辩论回放，或明天再来！"
        )
    return True, ""


def try_acquire_slot() -> bool:
    """Claim the single live-debate slot. Returns False if busy."""
    with _slot_lock:
        if _slot["active"] and (time.time() - _slot["since"]) < _STALE_AFTER_S:
            return False
        _slot["active"] = True
        _slot["since"] = time.time()
        return True


def release_slot() -> None:
    with _slot_lock:
        _slot["active"] = False
        _slot["since"] = 0.0
