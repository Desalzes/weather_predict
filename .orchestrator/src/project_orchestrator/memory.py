from __future__ import annotations

from hashlib import sha1
import sqlite3

from .db import utc_now


DIRECTIVE_MARKERS = ("always", "never", "do not", "don't", "must", "remember")


def extract_directive_candidates(text: str) -> list[str]:
    normalized = " ".join(text.strip().split())
    lowered = normalized.lower()
    if not normalized:
        return []
    if any(marker in lowered for marker in DIRECTIVE_MARKERS):
        return [normalized]
    return []


def upsert_directive(conn: sqlite3.Connection, content: str, source: str = "user_prompt") -> str:
    directive_id = sha1(content.encode("utf-8")).hexdigest()[:16]
    conn.execute(
        """
        INSERT OR REPLACE INTO directives (id, source, content, priority, active, created_at)
        VALUES (?, ?, ?, ?, 1, ?)
        """,
        (directive_id, source, content, "normal", utc_now()),
    )
    conn.commit()
    return directive_id
