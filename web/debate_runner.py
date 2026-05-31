"""Thin wrapper around main.app.stream() that yields structured events
for the Streamlit UI. The CLI renderer in main.py is preserved intact.

Event shapes (all dicts):
    {"type": "market_data",  "text": str}
    {"type": "round_start",  "round": int, "order": List[str]}
    {"type": "master_turn",  "round": int, "master": str, "emoji": str,
                             "content": str, "tendency": str | None,
                             "vote": str | None, "skipped": bool}
    {"type": "cross_question","round": int, "entries": List[str]}
    {"type": "round_end",    "round": int, "votes": Dict[str, str],
                             "reason": str}  # reason in {"in_progress", "max_rounds", "consensus", "early_stop"}
    {"type": "final_report", "text": str}
    {"type": "error",        "message": str}

Consumers drive this via a plain Python `for ev in run_debate(query): ...`
loop (or thread + queue if they want non-blocking UI).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Generator, List

# Make the project root importable even when launched via `streamlit run`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import main as M  # noqa: E402


def _extract_master_event(round_no: int, master: str, update: dict) -> Dict[str, Any]:
    emoji = M.MASTER_PERSONAS[master]["emoji"]
    new_entries = update.get("debate_history", [])
    new_votes = update.get("votes", {})
    new_tendencies = update.get("tendencies", {}).get(round_no, {})
    # On-demand search the master triggered this round (Issue #1)
    search_log = update.get("search_log", []) or []

    if not new_entries:
        return {
            "type": "master_turn",
            "round": round_no,
            "master": master,
            "emoji": emoji,
            "content": "",
            "tendency": new_tendencies.get(master),
            "vote": None,
            "skipped": True,
            "search": search_log,
        }

    raw = new_entries[0]
    content = raw.split("】: ", 1)[-1] if "】: " in raw else raw
    return {
        "type": "master_turn",
        "round": round_no,
        "master": master,
        "emoji": emoji,
        "content": content.strip(),
        "tendency": new_tendencies.get(master),
        "vote": new_votes.get(master),
        "skipped": False,
        "search": search_log,
    }


def run_debate(query: str) -> Generator[Dict[str, Any], None, None]:
    """Stream structured events from one end-to-end debate."""

    inputs: M.AgentState = {
        "query": query,
        "market_data": "",
        "debate_history": [],
        "round_count": 0,
        "votes": {},
        "tendencies": {},
        "early_stop": False,
        "round_order": {},
        "final_report": "",
        "search_log": [],
    }

    current_round = 0
    round_order_plan: Dict[int, List[str]] = {}
    # Buffer parallel master events so the UI can render them in the
    # stratified-shuffle order instead of LangGraph arrival order.
    buffer: List[Dict[str, Any]] = []
    header_emitted = False
    accumulated_votes: Dict[str, str] = {}

    def _flush_buffer() -> List[Dict[str, Any]]:
        if not buffer:
            return []
        plan = round_order_plan.get(current_round, M.MASTER_NAMES)
        by_master = {ev["master"]: ev for ev in buffer}
        ordered = [by_master[m] for m in plan if m in by_master]
        buffer.clear()
        return ordered

    try:
        for event in M.app.stream(inputs, stream_mode="updates"):
            for node_name, update in event.items():

                if node_name == "DataCollection":
                    current_round = 1
                    round_order_plan = update.get("round_order", {}) or {}
                    md = update.get("market_data") or ""
                    yield {"type": "market_data", "text": md}
                    yield {
                        "type": "round_start",
                        "round": current_round,
                        "order": round_order_plan.get(current_round, M.MASTER_NAMES),
                    }
                    header_emitted = True

                elif node_name in M.MASTER_NAMES:
                    if not header_emitted:
                        # Defensive: emit a round_start if we somehow missed it
                        yield {
                            "type": "round_start",
                            "round": current_round,
                            "order": round_order_plan.get(
                                current_round, M.MASTER_NAMES
                            ),
                        }
                        header_emitted = True
                    buffer.append(
                        _extract_master_event(current_round, node_name, update)
                    )
                    # Track votes so we can emit them at round_end
                    accumulated_votes.update(update.get("votes", {}) or {})

                elif node_name == "CrossQuestion":
                    # All parallel masters finished this round — flush them
                    for ev in _flush_buffer():
                        yield ev
                    entries = update.get("debate_history", [])
                    if entries:
                        yield {
                            "type": "cross_question",
                            "round": current_round,
                            "entries": entries,
                        }

                elif node_name == "UpdateRound":
                    # Round officially closed. Emit round_end.
                    new_round = update.get("round_count", current_round)
                    early_stop = update.get("early_stop", False)
                    reason = (
                        "max_rounds"
                        if new_round >= M.MAX_ROUNDS
                        else ("consensus_or_early_stop" if early_stop else "in_progress")
                    )
                    yield {
                        "type": "round_end",
                        "round": current_round,
                        "votes": dict(accumulated_votes),
                        "reason": reason,
                    }
                    current_round = new_round + 1 if not early_stop else new_round
                    header_emitted = False

                elif node_name == "Researcher":
                    # Flush any leftover masters before the final report
                    for ev in _flush_buffer():
                        yield ev
                    yield {
                        "type": "final_report",
                        "text": update.get("final_report", ""),
                    }
    except Exception as e:  # noqa: BLE001
        yield {"type": "error", "message": str(e)}
