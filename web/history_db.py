"""Local SQLite store for past debates (D4-2).

Schema:
    debates(
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at   TEXT NOT NULL,           -- ISO-8601
        query        TEXT NOT NULL,
        final_report TEXT NOT NULL,
        transcript   TEXT NOT NULL,           -- JSON-encoded rounds + market data
        model        TEXT                     -- LLM model used (diagnostic)
    )

Followup chats are stored as an append-only JSON list on the same row
(`followup`) to keep everything self-contained per debate.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "history.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# SQLite connections are NOT shareable across threads by default.
# Streamlit serves each session in its own thread, so we use a lock
# and `check_same_thread=False` on a module-level connection.
_conn_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _init_schema(_conn)
    return _conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS debates (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at   TEXT NOT NULL,
            query        TEXT NOT NULL,
            final_report TEXT NOT NULL,
            transcript   TEXT NOT NULL,
            followup     TEXT NOT NULL DEFAULT '[]',
            model        TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_debates_created_at
            ON debates(created_at DESC);
        """
    )
    conn.commit()


def save_debate(
    query: str,
    final_report: str,
    transcript: Dict[str, Any],
    model: Optional[str] = None,
) -> int:
    """Persist a completed debate; return its row id."""
    with _conn_lock:
        cur = _get_conn().execute(
            """
            INSERT INTO debates(created_at, query, final_report, transcript, model)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                query,
                final_report,
                json.dumps(transcript, ensure_ascii=False),
                model or "",
            ),
        )
        _get_conn().commit()
        return int(cur.lastrowid)


def list_debates(limit: int = 50) -> List[Dict[str, Any]]:
    with _conn_lock:
        rows = _get_conn().execute(
            "SELECT id, created_at, query FROM debates "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def load_debate(debate_id: int) -> Optional[Dict[str, Any]]:
    with _conn_lock:
        row = _get_conn().execute(
            "SELECT * FROM debates WHERE id = ?", (debate_id,)
        ).fetchone()
    if row is None:
        return None
    data = dict(row)
    data["transcript"] = json.loads(data["transcript"])
    data["followup"] = json.loads(data["followup"] or "[]")
    return data


def append_followup(debate_id: int, role: str, content: str) -> None:
    """Append one message to this debate's followup chat log."""
    with _conn_lock:
        row = _get_conn().execute(
            "SELECT followup FROM debates WHERE id = ?", (debate_id,)
        ).fetchone()
        if row is None:
            return
        msgs: List[Dict[str, str]] = json.loads(row["followup"] or "[]")
        msgs.append({"role": role, "content": content})
        _get_conn().execute(
            "UPDATE debates SET followup = ? WHERE id = ?",
            (json.dumps(msgs, ensure_ascii=False), debate_id),
        )
        _get_conn().commit()


def delete_debate(debate_id: int) -> None:
    with _conn_lock:
        _get_conn().execute("DELETE FROM debates WHERE id = ?", (debate_id,))
        _get_conn().commit()
