"""
SQLite-backed store for retrieval events.
"""
import sqlite3
import json
import time
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class RetrievalEvent:
    query: str
    label: str
    chunks: list[dict]
    session_id: Optional[str] = None
    user: Optional[str] = None
    metadata: Optional[dict] = None
    timestamp: Optional[float] = None
    id: Optional[int] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class SQLiteStore:
    def __init__(self, db_path: str = ".rag_debug/sessions.db"):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self._migrate()

    def _migrate(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   REAL NOT NULL,
                session_id  TEXT,
                user        TEXT,
                label       TEXT,
                query       TEXT NOT NULL,
                chunks      TEXT NOT NULL,
                metadata    TEXT
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session
            ON retrieval_events (session_id)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON retrieval_events (timestamp)
        """)
        self.conn.commit()

    def save(self, event: RetrievalEvent) -> int:
        cur = self.conn.execute(
            """INSERT INTO retrieval_events
               (timestamp, session_id, user, label, query, chunks, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                event.timestamp,
                event.session_id,
                event.user,
                event.label or "",
                event.query,
                json.dumps(event.chunks),
                json.dumps(event.metadata) if event.metadata else None,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def recent(self, limit: int = 50) -> list[RetrievalEvent]:
        rows = self.conn.execute(
            "SELECT * FROM retrieval_events ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def by_session(self, session_id: str) -> list[RetrievalEvent]:
        rows = self.conn.execute(
            "SELECT * FROM retrieval_events WHERE session_id = ? ORDER BY timestamp",
            (session_id,)
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def count_similar_gaps(self, missing_topics: list[str], window_days: int = 7) -> int:
        """Count how often similar gap topics appeared in recent sessions."""
        since = time.time() - (window_days * 86400)
        rows = self.conn.execute(
            "SELECT metadata FROM retrieval_events WHERE timestamp > ? AND metadata IS NOT NULL",
            (since,)
        ).fetchall()
        count = 0
        for (meta_json,) in rows:
            try:
                meta = json.loads(meta_json)
                past_gaps = meta.get("missing_topics", [])
                if any(t.lower() in " ".join(past_gaps).lower() for t in missing_topics):
                    count += 1
            except (json.JSONDecodeError, AttributeError):
                continue
        return count

    @staticmethod
    def _row_to_event(row) -> RetrievalEvent:
        return RetrievalEvent(
            id=row[0],
            timestamp=row[1],
            session_id=row[2],
            user=row[3],
            label=row[4],
            query=row[5],
            chunks=json.loads(row[6]),
            metadata=json.loads(row[7]) if row[7] else None,
        )

    def close(self):
        self.conn.close()
