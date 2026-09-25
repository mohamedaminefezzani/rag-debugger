"""
Tests for @rd.trace decorator — sync, async, bare, with args, key overrides.
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from rag_debugger.store import SQLiteStore
from rag_debugger.trace import make_trace_decorator, _normalize_result, _apply_key_overrides
from rag_debugger.session import Session, _get_stack


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_session_stack():
    _get_stack().clear()
    yield
    _get_stack().clear()


@pytest.fixture
def store(tmp_path):
    return SQLiteStore(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def trace(store):
    return make_trace_decorator(store)


# ── _normalize_result ─────────────────────────────────────────────────────────

class TestNormalizeResult:
    def test_list_of_dicts(self):
        result = [{"content": "hello", "score": 0.9}]
        assert _normalize_result(result)[0]["content"] == "hello"

    def test_list_of_strings(self):
        result = ["chunk one", "chunk two"]
        out = _normalize_result(result)
        assert out[0]["content"] == "chunk one"

    def test_langchain_document(self):
        doc = MagicMock(spec=["page_content", "metadata"])
        doc.page_content = "langchain chunk"
        doc.metadata = {}
        out = _normalize_result([doc])
        assert out[0]["content"] == "langchain chunk"

    def test_llamaindex_node(self):
        node = MagicMock()
        node.score = 0.88
        node.node.text = "llama chunk"
        node.node.node_id = "n1"
        node.node.metadata = {}
        out = _normalize_result([node])
        assert out[0]["content"] == "llama chunk"
        assert out[0]["score"] == 0.88

    def test_empty_returns_empty(self):
        assert _normalize_result([]) == []

    def test_none_returns_empty(self):
        assert _normalize_result(None) == []


# ── _apply_key_overrides ──────────────────────────────────────────────────────

class TestApplyKeyOverrides:
    def test_no_override_passthrough(self):
        chunks = [{"content": "text", "score": 0.8}]
        result = _apply_key_overrides(chunks, "score", "content")
        assert result == chunks

    def test_custom_score_key(self):
        chunks = [{"content": "text", "relevance": 0.9}]
        result = _apply_key_overrides(chunks, score_key="relevance", content_key="content")
        assert result[0]["score"] == 0.9

    def test_custom_content_key(self):
        chunks = [{"text": "hello", "score": 0.7}]
        result = _apply_key_overrides(chunks, score_key="score", content_key="text")
        assert result[0]["content"] == "hello"

    def test_both_keys_overridden(self):
        chunks = [{"body": "chunk text", "relevance": 0.85}]
        result = _apply_key_overrides(chunks, score_key="relevance", content_key="body")
        assert result[0]["content"] == "chunk text"
        assert result[0]["score"] == 0.85


# ── @trace bare (no parens) ───────────────────────────────────────────────────

class TestTraceBare:
    def test_bare_decorator_logs_event(self, trace, store):
        @trace
        def retrieve(query):
            return [{"content": "doc", "score": 0.8}]

        retrieve("test query")
        events = store.recent()
        assert len(events) == 1
        assert events[0].query == "test query"

    def test_bare_uses_function_name_as_label(self, trace, store):
        @trace
        def my_retriever(query):
            return [{"content": "doc", "score": 0.8}]

        my_retriever("q")
        assert store.recent()[0].label == "my_retriever"

    def test_bare_return_value_preserved(self, trace, store):
        original = [{"content": "doc", "score": 0.8}]

        @trace
        def retrieve(query):
            return original

        result = retrieve("q")
        assert result == original


# ── @trace(label=x) with args ────────────────────────────────────────────────

class TestTraceWithArgs:
    def test_label_arg(self, trace, store):
        @trace(label="custom-label")
        def retrieve(query):
            return [{"content": "doc", "score": 0.8}]

        retrieve("q")
        assert store.recent()[0].label == "custom-label"

    def test_score_key_override(self, trace, store):
        @trace(score_key="relevance")
        def retrieve(query):
            return [{"content": "doc", "relevance": 0.91}]

        retrieve("q")
        assert store.recent()[0].chunks[0]["score"] == 0.91

    def test_content_key_override(self, trace, store):
        @trace(content_key="text")
        def retrieve(query):
            return [{"text": "hello", "score": 0.7}]

        retrieve("q")
        assert store.recent()[0].chunks[0]["content"] == "hello"

    def test_both_key_overrides(self, trace, store):
        @trace(score_key="relevance", content_key="body")
        def retrieve(query):
            return [{"body": "chunk", "relevance": 0.85}]

        retrieve("q")
        event = store.recent()[0]
        assert event.chunks[0]["content"] == "chunk"
        assert event.chunks[0]["score"] == 0.85

    def test_explicit_session_id(self, trace, store):
        @trace(session_id="explicit-sess")
        def retrieve(query):
            return [{"content": "doc", "score": 0.8}]

        retrieve("q")
        assert store.recent()[0].session_id == "explicit-sess"


# ── async support ─────────────────────────────────────────────────────────────

class TestTraceAsync:
    def test_async_bare_decorator(self, trace, store):
        @trace
        async def retrieve(query):
            return [{"content": "async doc", "score": 0.75}]

        asyncio.run(retrieve("async query"))
        events = store.recent()
        assert len(events) == 1
        assert events[0].query == "async query"
        assert events[0].chunks[0]["content"] == "async doc"

    def test_async_with_label(self, trace, store):
        @trace(label="async-retriever")
        async def retrieve(query):
            return [{"content": "doc", "score": 0.8}]

        asyncio.run(retrieve("q"))
        assert store.recent()[0].label == "async-retriever"

    def test_async_return_value_preserved(self, trace, store):
        original = [{"content": "doc", "score": 0.8}]

        @trace
        async def retrieve(query):
            return original

        result = asyncio.run(retrieve("q"))
        assert result == original


# ── session integration ───────────────────────────────────────────────────────

class TestTraceSessionIntegration:
    def test_picks_up_active_session(self, trace, store):
        @trace
        def retrieve(query):
            return [{"content": "doc", "score": 0.8}]

        with Session(store, id="sess-trace-001"):
            retrieve("query inside session")

        events = store.by_session("sess-trace-001")
        assert len(events) == 1
        assert events[0].query == "query inside session"

    def test_no_session_id_when_outside(self, trace, store):
        @trace
        def retrieve(query):
            return [{"content": "doc", "score": 0.8}]

        retrieve("outside")
        assert store.recent()[0].session_id is None

    def test_explicit_session_overrides_context(self, trace, store):
        @trace(session_id="explicit")
        def retrieve(query):
            return [{"content": "doc", "score": 0.8}]

        with Session(store, id="context-sess"):
            retrieve("q")

        assert store.recent()[0].session_id == "explicit"
        assert store.by_session("context-sess") == []


# ── elapsed_ms in metadata ────────────────────────────────────────────────────

class TestTraceMetadata:
    def test_elapsed_ms_logged(self, trace, store):
        @trace
        def retrieve(query):
            return [{"content": "doc", "score": 0.8}]

        retrieve("q")
        assert "elapsed_ms" in store.recent()[0].metadata

    def test_multiple_calls_all_logged(self, trace, store):
        @trace
        def retrieve(query):
            return [{"content": "doc", "score": 0.8}]

        retrieve("q1")
        retrieve("q2")
        retrieve("q3")
        assert len(store.recent()) == 3

    def test_query_from_kwarg(self, trace, store):
        @trace
        def retrieve(query):
            return [{"content": "doc", "score": 0.8}]

        retrieve(query="kwarg query")
        assert store.recent()[0].query == "kwarg query"
