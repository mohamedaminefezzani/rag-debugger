"""
Top-level rd.init() / rd.wrap_retriever() / rd.log_retrieval() API.
"""
from typing import Any, Optional
from .store import SQLiteStore, RetrievalEvent
from .wrap import wrap_retriever as _wrap
from .gap_detector import ChunkResult
from .session import Session

# Global state — set by rd.init()
_store: Optional[SQLiteStore] = None
_project: Optional[str] = None


def init(
    project: str,
    db_path: str = ".rag_debug/sessions.db",
    **kwargs,
) -> None:
    """Initialize rag-debugger. Call once at app startup."""
    global _store, _project
    _project = project
    _store = SQLiteStore(db_path=db_path)


def _get_store() -> SQLiteStore:
    if _store is None:
        raise RuntimeError(
            "rag_debugger not initialized. Call rd.init(project='my-app') first."
        )
    return _store


def wrap_retriever(
    retriever: Any,
    label: Optional[str] = None,
    session_id: Optional[str] = None,
    method: Optional[str] = None,
) -> Any:
    """Wrap any retriever so its calls are logged automatically."""
    return _wrap(retriever, _get_store(), label=label, session_id=session_id, method=method)


def log_retrieval(
    query: str,
    chunks: list[ChunkResult],
    label: Optional[str] = None,
    metadata: Optional[dict] = None,
    session_id: Optional[str] = None,
) -> RetrievalEvent:
    """Manually log a retrieval event."""
    event = RetrievalEvent(
        query=query,
        label=label or "manual",
        chunks=[
            {
                "content": c.content,
                "score": c.score,
                "chunk_id": c.chunk_id,
                "source": c.source,
                "selected": c.selected,
                "metadata": c.metadata,
            }
            for c in chunks
        ],
        session_id=session_id,
        metadata=metadata,
    )
    _get_store().save(event)
    return event


def session(
    id: Optional[str] = None,
    user: Optional[str] = None,
    tags: Optional[list] = None,
    gap_threshold: float = 0.65,
) -> Session:
    """
    Context manager that groups retrieval events under a single session.

    Usage:
        with rd.session(id="conv-123", user="user-42") as s:
            retriever.get_relevant_documents("query 1")
            retriever.get_relevant_documents("query 2")
        print(s.summary())
    """
    return Session(
        store=_get_store(),
        id=id,
        user=user,
        tags=tags,
        gap_threshold=gap_threshold,
    )


def dashboard(
    host: str = "127.0.0.1",
    port: int = 7842,
    open_browser: bool = True,
) -> None:
    """Launch the local dashboard at http://localhost:7842"""
    from .dashboard import launch
    launch(_get_store(), _project or "rag-debugger", host=host, port=port, open_browser=open_browser)
