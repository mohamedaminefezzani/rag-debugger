"""
@rd.trace decorator — instrument any retrieval function with one line.

Usage:
    @rd.trace
    def retrieve(query): ...

    @rd.trace(label="my-retriever", score_key="relevance", content_key="text")
    async def retrieve(query): ...
"""
import asyncio
import functools
import time
from typing import Optional, Callable, Any

from .store import SQLiteStore, RetrievalEvent
from .session import get_active_session_id
from .wrap import _normalize_langchain, _normalize_llamaindex, _normalize_custom


def _normalize_result(result: Any) -> list[dict]:
    """Auto-detect return type and normalize to list[dict]."""
    if not result:
        return []

    first = result[0] if isinstance(result, list) and result else result

    # LlamaIndex NodeWithScore — check before LangChain since MagicMock bleeds
    if hasattr(first, "score") and hasattr(first, "node"):
        return _normalize_llamaindex(result)

    # LangChain Document — has page_content attribute
    if hasattr(first, "page_content"):
        return _normalize_langchain(result)

    # everything else
    return _normalize_custom(result)


def _apply_key_overrides(chunks: list[dict], score_key: str, content_key: str) -> list[dict]:
    """Remap custom score/content keys to standard ones."""
    if score_key == "score" and content_key == "content":
        return chunks  # nothing to remap
    out = []
    for c in chunks:
        remapped = dict(c)
        if score_key != "score" and score_key in c:
            remapped["score"] = c[score_key]
        if content_key != "content" and content_key in c:
            remapped["content"] = c[content_key]
        out.append(remapped)
    return out


def _make_trace(
    fn: Callable,
    store: SQLiteStore,
    label: Optional[str],
    score_key: str,
    content_key: str,
    session_id: Optional[str],
):
    """Wrap a sync function with tracing."""
    @functools.wraps(fn)
    def sync_wrapper(*args, **kwargs):
        # extract query — first positional arg or 'query' kwarg
        query = kwargs.get("query", args[0] if args else "")
        t0 = time.time()
        result = fn(*args, **kwargs)
        elapsed = round((time.time() - t0) * 1000)
        chunks = _apply_key_overrides(_normalize_result(result), score_key, content_key)
        store.save(RetrievalEvent(
            query=str(query),
            label=label or fn.__name__,
            chunks=chunks,
            session_id=session_id or get_active_session_id(),
            metadata={"elapsed_ms": elapsed},
        ))
        return result
    return sync_wrapper


def _make_async_trace(
    fn: Callable,
    store: SQLiteStore,
    label: Optional[str],
    score_key: str,
    content_key: str,
    session_id: Optional[str],
):
    """Wrap an async function with tracing."""
    @functools.wraps(fn)
    async def async_wrapper(*args, **kwargs):
        query = kwargs.get("query", args[0] if args else "")
        t0 = time.time()
        result = await fn(*args, **kwargs)
        elapsed = round((time.time() - t0) * 1000)
        chunks = _apply_key_overrides(_normalize_result(result), score_key, content_key)
        store.save(RetrievalEvent(
            query=str(query),
            label=label or fn.__name__,
            chunks=chunks,
            session_id=session_id or get_active_session_id(),
            metadata={"elapsed_ms": elapsed},
        ))
        return result
    return async_wrapper


def make_trace_decorator(store: SQLiteStore):
    """
    Returns the @rd.trace decorator bound to a SQLiteStore.
    Supports both @rd.trace and @rd.trace(label='x', ...) usage.
    """
    def trace(
        fn: Optional[Callable] = None,
        *,
        label: Optional[str] = None,
        score_key: str = "score",
        content_key: str = "content",
        session_id: Optional[str] = None,
    ):
        def decorator(f: Callable) -> Callable:
            if asyncio.iscoroutinefunction(f):
                return _make_async_trace(f, store, label, score_key, content_key, session_id)
            return _make_trace(f, store, label, score_key, content_key, session_id)

        # @rd.trace — called with the function directly
        if fn is not None:
            return decorator(fn)

        # @rd.trace(...) — called with arguments, return decorator
        return decorator

    return trace
