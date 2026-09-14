"""
session() — context manager that groups retrieval events under a session ID.

Usage:
    with rd.session(id="conv-123", user="user-42") as s:
        retriever.get_relevant_documents("first query")
        retriever.get_relevant_documents("follow-up query")

    print(s.summary())

Thread-safe: uses threading.local() so concurrent sessions don't bleed into
each other. Each thread has its own active session stack.
"""
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Generator

from .store import SQLiteStore, RetrievalEvent


# ── thread-local session stack ────────────────────────────────────────────────

_local = threading.local()


def _get_stack() -> list:
    if not hasattr(_local, "stack"):
        _local.stack = []
    return _local.stack


def get_active_session_id() -> Optional[str]:
    """Return the innermost active session ID, or None."""
    stack = _get_stack()
    return stack[-1] if stack else None


def _push(session_id: str) -> None:
    _get_stack().append(session_id)


def _pop() -> None:
    stack = _get_stack()
    if stack:
        stack.pop()


# ── SessionSummary ────────────────────────────────────────────────────────────

@dataclass
class SessionSummary:
    session_id: str
    user: Optional[str]
    tags: list[str]
    started_at: float
    ended_at: float
    total_events: int
    queries: list[str]
    avg_top_score: float          # mean of best chunk scores across all events
    lowest_top_score: float       # worst single retrieval in session
    gap_count: int                # events where best score < gap_threshold
    gap_threshold: float

    @property
    def duration_ms(self) -> int:
        return round((self.ended_at - self.started_at) * 1000)

    def __str__(self) -> str:
        lines = [
            f"Session {self.session_id}",
            f"  duration:    {self.duration_ms}ms",
            f"  events:      {self.total_events}",
            f"  avg score:   {self.avg_top_score:.3f}",
            f"  worst score: {self.lowest_top_score:.3f}",
            f"  gaps:        {self.gap_count} / {self.total_events}",
        ]
        if self.user:
            lines.insert(1, f"  user:        {self.user}")
        for i, q in enumerate(self.queries, 1):
            lines.append(f"  [{i}] {q[:80]}{'...' if len(q) > 80 else ''}")
        return "\n".join(lines)


# ── Session ───────────────────────────────────────────────────────────────────

class Session:
    def __init__(
        self,
        store: SQLiteStore,
        id: Optional[str] = None,
        user: Optional[str] = None,
        tags: Optional[list[str]] = None,
        gap_threshold: float = 0.65,
    ):
        self.id = id or f"sess-{uuid.uuid4().hex[:12]}"
        self.user = user
        self.tags = tags or []
        self.gap_threshold = gap_threshold
        self._store = store
        self._started_at: Optional[float] = None
        self._ended_at: Optional[float] = None
        self._summary: Optional[SessionSummary] = None

    def __enter__(self) -> "Session":
        self._started_at = time.time()
        _push(self.id)
        return self

    def __exit__(self, *_) -> None:
        self._ended_at = time.time()
        _pop()
        self._summary = self._build_summary()

    def summary(self) -> SessionSummary:
        if self._summary is None:
            raise RuntimeError("summary() is only available after the session context exits.")
        return self._summary

    def _build_summary(self) -> SessionSummary:
        events = self._store.by_session(self.id)
        top_scores = []
        queries = []

        for event in events:
            queries.append(event.query)
            scores = [
                c.get("score") for c in event.chunks
                if c.get("score") is not None
            ]
            if scores:
                top_scores.append(max(scores))

        avg = round(sum(top_scores) / len(top_scores), 4) if top_scores else 0.0
        worst = round(min(top_scores), 4) if top_scores else 0.0
        gaps = sum(1 for s in top_scores if s < self.gap_threshold)

        return SessionSummary(
            session_id=self.id,
            user=self.user,
            tags=self.tags,
            started_at=self._started_at,
            ended_at=self._ended_at,
            total_events=len(events),
            queries=queries,
            avg_top_score=avg,
            lowest_top_score=worst,
            gap_count=gaps,
            gap_threshold=self.gap_threshold,
        )
