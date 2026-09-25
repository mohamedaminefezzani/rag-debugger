"""
wrap_retriever() — zero-friction instrumentation for any retrieval object.

Detection order:
  1. LangChain BaseRetriever  → patch get_relevant_documents()
  2. LlamaIndex BaseRetriever → patch retrieve()
  3. Custom object            → patch first found: retrieve / query / get_relevant_documents
  4. None matched             → raise clear TypeError
"""
import functools
import time
from typing import Any, Optional
from .store import SQLiteStore, RetrievalEvent
from .session import get_active_session_id


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_langchain(obj: Any) -> bool:
    try:
        from langchain_core.retrievers import BaseRetriever
        return isinstance(obj, BaseRetriever)
    except ImportError:
        try:
            from langchain.schema import BaseRetriever
            return isinstance(obj, BaseRetriever)
        except ImportError:
            return False


def _is_llamaindex(obj: Any) -> bool:
    try:
        from llama_index.core.retrievers import BaseRetriever
        return isinstance(obj, BaseRetriever)
    except ImportError:
        try:
            from llama_index.retrievers import BaseRetriever
            return isinstance(obj, BaseRetriever)
        except ImportError:
            return False


def _normalize_langchain(results: list) -> list[dict]:
    """Convert LangChain Document list to chunk dicts."""
    out = []
    for doc in results:
        chunk = {"content": getattr(doc, "page_content", str(doc))}
        if hasattr(doc, "metadata"):
            chunk["metadata"] = doc.metadata
            chunk["source"] = doc.metadata.get("source", "")
            # LangChain doesn't always attach scores — use metadata if present
            chunk["score"] = doc.metadata.get("score",
                             doc.metadata.get("relevance_score",
                             doc.metadata.get("similarity_score", None)))
        out.append(chunk)
    return out


def _normalize_llamaindex(results: list) -> list[dict]:
    """Convert LlamaIndex NodeWithScore list to chunk dicts."""
    out = []
    for node in results:
        score = getattr(node, "score", None)
        n = getattr(node, "node", node)
        chunk = {
            "content": getattr(n, "text", getattr(n, "get_content", lambda: str(n))()),
            "score": score,
            "chunk_id": getattr(n, "node_id", None),
            "metadata": getattr(n, "metadata", {}),
        }
        out.append(chunk)
    return out


def _normalize_custom(results: Any) -> list[dict]:
    """Best-effort normalization for unknown return types."""
    if not results:
        return []
    if isinstance(results, list):
        out = []
        for item in results:
            if isinstance(item, dict):
                # already dict — just ensure "content" key exists
                content = item.get("content") or item.get("text") or item.get("page_content") or str(item)
                out.append({**item, "content": content})
            elif isinstance(item, str):
                out.append({"content": item})
            else:
                # try common attributes
                content = (getattr(item, "content", None) or
                           getattr(item, "text", None) or
                           getattr(item, "page_content", None) or str(item))
                score = getattr(item, "score", None) or getattr(item, "similarity", None)
                out.append({"content": content, "score": score})
        return out
    return [{"content": str(results)}]


# ── core wrapper ──────────────────────────────────────────────────────────────

def wrap_retriever(
    retriever: Any,
    store: SQLiteStore,
    label: Optional[str] = None,
    session_id: Optional[str] = None,
    method: Optional[str] = None,
) -> Any:
    """
    Wrap a retriever object so every retrieval call is logged to SQLite.
    Returns the same object with the retrieval method monkey-patched.

    Args:
        retriever:  Any retriever object.
        store:      SQLiteStore instance to log events to.
        label:      Human-readable name shown in the dashboard.
        session_id: Group this retriever's calls under a session.
        method:     Force a specific method name to patch (skips auto-detection).
    """
    if _is_langchain(retriever):
        return _wrap_langchain(retriever, store, label, session_id)

    if _is_llamaindex(retriever):
        return _wrap_llamaindex(retriever, store, label, session_id)

    # custom object — find method to patch
    candidate_methods = [method] if method else ["retrieve", "query", "get_relevant_documents", "search"]
    for m in candidate_methods:
        if callable(getattr(retriever, m, None)):
            return _wrap_custom(retriever, m, store, label, session_id)

    raise TypeError(
        f"wrap_retriever() could not find a retrieval method on {type(retriever).__name__}. "
        f"Expected one of: retrieve(), query(), get_relevant_documents(), search(). "
        f"Pass method='your_method_name' to specify it explicitly."
    )


def _make_logger(normalize_fn, store, label, session_id, method_name="retriever"):
    """Returns a decorator that logs retrieval calls."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # args[0] may be 'self' (unbound method) or the query string (bound method)
            # detect by checking if first arg is a string
            if args and isinstance(args[0], str):
                query = args[0]
            elif len(args) > 1 and isinstance(args[1], str):
                query = args[1]
            else:
                query = kwargs.get("query", "")
            t0 = time.time()
            raw = fn(*args, **kwargs)
            elapsed = round(time.time() - t0, 4)
            chunks = normalize_fn(raw)
            store.save(RetrievalEvent(
                query=str(query),
                label=label or getattr(fn, "__name__", method_name),
                chunks=chunks,
                session_id=session_id or get_active_session_id(),
                metadata={"elapsed_ms": round(elapsed * 1000)},
            ))
            return raw
        return wrapper
    return decorator


def _wrap_langchain(retriever, store, label, session_id):
    decorator = _make_logger(_normalize_langchain, store, label, session_id)
    # patch both sync and async
    retriever.get_relevant_documents = decorator(retriever.get_relevant_documents)
    if hasattr(retriever, "aget_relevant_documents"):
        @functools.wraps(retriever.aget_relevant_documents)
        async def async_wrapper(query, *args, **kwargs):
            raw = await retriever.aget_relevant_documents.__wrapped__(query, *args, **kwargs)
            chunks = _normalize_langchain(raw)
            store.save(RetrievalEvent(
                query=str(query),
                label=label or "langchain",
                chunks=chunks,
                session_id=session_id,
            ))
            return raw
        retriever.aget_relevant_documents = async_wrapper
    return retriever


def _wrap_llamaindex(retriever, store, label, session_id):
    decorator = _make_logger(_normalize_llamaindex, store, label, session_id)
    retriever.retrieve = decorator(retriever.retrieve)
    return retriever


def _wrap_custom(retriever, method_name, store, label, session_id):
    decorator = _make_logger(_normalize_custom, store, label, session_id, method_name)
    setattr(retriever, method_name, decorator(getattr(retriever, method_name)))
    return retriever