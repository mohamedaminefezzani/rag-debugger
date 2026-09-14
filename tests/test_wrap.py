"""
Tests for wrap_retriever() — all mocked, no real LangChain/LlamaIndex needed.
"""
import pytest
import tempfile, os
from unittest.mock import MagicMock, patch
from rag_debugger.store import SQLiteStore
from rag_debugger.wrap import wrap_retriever, _normalize_langchain, _normalize_llamaindex, _normalize_custom


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    return SQLiteStore(db_path=str(tmp_path / "test.db"))


def make_langchain_retriever(docs):
    """Mock a LangChain-style retriever."""
    retriever = MagicMock()
    retriever.get_relevant_documents = MagicMock(return_value=docs)
    # make isinstance check pass
    with patch("rag_debugger.wrap._is_langchain", return_value=True), \
         patch("rag_debugger.wrap._is_llamaindex", return_value=False):
        return retriever


def make_llamaindex_retriever(nodes):
    retriever = MagicMock()
    retriever.retrieve = MagicMock(return_value=nodes)
    with patch("rag_debugger.wrap._is_langchain", return_value=False), \
         patch("rag_debugger.wrap._is_llamaindex", return_value=True):
        return retriever


def make_custom_retriever(method="retrieve", return_value=None):
    retriever = MagicMock(spec=[method])
    getattr(retriever, method).return_value = return_value or [
        {"content": "custom result", "score": 0.8}
    ]
    return retriever


# ── Normalization unit tests ──────────────────────────────────────────────────

class TestNormalizeLangchain:
    def test_extracts_page_content(self):
        doc = MagicMock()
        doc.page_content = "Hello world"
        doc.metadata = {}
        result = _normalize_langchain([doc])
        assert result[0]["content"] == "Hello world"

    def test_extracts_score_from_metadata(self):
        doc = MagicMock()
        doc.page_content = "text"
        doc.metadata = {"score": 0.91}
        result = _normalize_langchain([doc])
        assert result[0]["score"] == 0.91

    def test_empty_list(self):
        assert _normalize_langchain([]) == []


class TestNormalizeLlamaindex:
    def test_extracts_text_and_score(self):
        node = MagicMock()
        node.score = 0.85
        node.node.text = "llama chunk"
        node.node.node_id = "node-1"
        node.node.metadata = {}
        result = _normalize_llamaindex([node])
        assert result[0]["score"] == 0.85
        assert result[0]["chunk_id"] == "node-1"

    def test_empty_list(self):
        assert _normalize_llamaindex([]) == []


class TestNormalizeCustom:
    def test_dict_with_content_key(self):
        result = _normalize_custom([{"content": "hello", "score": 0.5}])
        assert result[0]["content"] == "hello"

    def test_dict_with_text_key(self):
        result = _normalize_custom([{"text": "hello"}])
        assert result[0]["content"] == "hello"

    def test_plain_string(self):
        result = _normalize_custom(["plain string"])
        assert result[0]["content"] == "plain string"

    def test_empty(self):
        assert _normalize_custom([]) == []

    def test_none(self):
        assert _normalize_custom(None) == []


# ── wrap_retriever integration tests ─────────────────────────────────────────

class TestWrapRetriever:
    def test_langchain_logs_to_store(self, store):
        doc = MagicMock()
        doc.page_content = "cancellation policy"
        doc.metadata = {}

        retriever = MagicMock()
        retriever.get_relevant_documents = MagicMock(return_value=[doc])

        with patch("rag_debugger.wrap._is_langchain", return_value=True), \
             patch("rag_debugger.wrap._is_llamaindex", return_value=False):
            wrapped = wrap_retriever(retriever, store, label="test-lc")

        wrapped.get_relevant_documents("how to cancel")
        events = store.recent()
        assert len(events) == 1
        assert events[0].query == "how to cancel"
        assert events[0].label == "test-lc"
        assert events[0].chunks[0]["content"] == "cancellation policy"

    def test_llamaindex_logs_to_store(self, store):
        node = MagicMock()
        node.score = 0.77
        node.node.text = "refund policy"
        node.node.node_id = "n1"
        node.node.metadata = {}

        retriever = MagicMock()
        retriever.retrieve = MagicMock(return_value=[node])

        with patch("rag_debugger.wrap._is_langchain", return_value=False), \
             patch("rag_debugger.wrap._is_llamaindex", return_value=True):
            wrapped = wrap_retriever(retriever, store, label="test-li")

        wrapped.retrieve("refund question")
        events = store.recent()
        assert len(events) == 1
        assert events[0].chunks[0]["content"] == "refund policy"

    def test_custom_retrieve_method(self, store):
        retriever = MagicMock(spec=["retrieve"])
        retriever.retrieve.return_value = [{"content": "custom doc", "score": 0.6}]

        with patch("rag_debugger.wrap._is_langchain", return_value=False), \
             patch("rag_debugger.wrap._is_llamaindex", return_value=False):
            wrapped = wrap_retriever(retriever, store)

        wrapped.retrieve("test query")
        events = store.recent()
        assert len(events) == 1
        assert events[0].chunks[0]["content"] == "custom doc"

    def test_custom_query_method(self, store):
        retriever = MagicMock(spec=["query"])
        retriever.query.return_value = [{"content": "query result", "score": 0.7}]

        with patch("rag_debugger.wrap._is_langchain", return_value=False), \
             patch("rag_debugger.wrap._is_llamaindex", return_value=False):
            wrapped = wrap_retriever(retriever, store)

        wrapped.query("test")
        events = store.recent()
        assert len(events) == 1

    def test_explicit_method_override(self, store):
        retriever = MagicMock(spec=["fetch_docs"])
        retriever.fetch_docs.return_value = [{"content": "fetched", "score": 0.5}]

        with patch("rag_debugger.wrap._is_langchain", return_value=False), \
             patch("rag_debugger.wrap._is_llamaindex", return_value=False):
            wrapped = wrap_retriever(retriever, store, method="fetch_docs")

        wrapped.fetch_docs("test")
        events = store.recent()
        assert len(events) == 1

    def test_original_return_value_preserved(self, store):
        """wrap_retriever must not change what the retriever returns."""
        original_docs = [{"content": "doc1"}, {"content": "doc2"}]
        retriever = MagicMock(spec=["retrieve"])
        retriever.retrieve.return_value = original_docs

        with patch("rag_debugger.wrap._is_langchain", return_value=False), \
             patch("rag_debugger.wrap._is_llamaindex", return_value=False):
            wrapped = wrap_retriever(retriever, store)

        result = wrapped.retrieve("query")
        assert result == original_docs

    def test_no_matching_method_raises(self, store):
        retriever = MagicMock(spec=["some_other_method"])
        with patch("rag_debugger.wrap._is_langchain", return_value=False), \
             patch("rag_debugger.wrap._is_llamaindex", return_value=False):
            with pytest.raises(TypeError, match="wrap_retriever\\(\\)"):
                wrap_retriever(retriever, store)

    def test_session_id_attached(self, store):
        retriever = MagicMock(spec=["retrieve"])
        retriever.retrieve.return_value = [{"content": "doc"}]

        with patch("rag_debugger.wrap._is_langchain", return_value=False), \
             patch("rag_debugger.wrap._is_llamaindex", return_value=False):
            wrapped = wrap_retriever(retriever, store, session_id="sess-123")

        wrapped.retrieve("query")
        events = store.recent()
        assert events[0].session_id == "sess-123"

    def test_multiple_calls_all_logged(self, store):
        retriever = MagicMock(spec=["retrieve"])
        retriever.retrieve.return_value = [{"content": "doc"}]

        with patch("rag_debugger.wrap._is_langchain", return_value=False), \
             patch("rag_debugger.wrap._is_llamaindex", return_value=False):
            wrapped = wrap_retriever(retriever, store)

        wrapped.retrieve("query 1")
        wrapped.retrieve("query 2")
        wrapped.retrieve("query 3")
        assert len(store.recent()) == 3

    def test_elapsed_ms_in_metadata(self, store):
        retriever = MagicMock(spec=["retrieve"])
        retriever.retrieve.return_value = [{"content": "doc"}]

        with patch("rag_debugger.wrap._is_langchain", return_value=False), \
             patch("rag_debugger.wrap._is_llamaindex", return_value=False):
            wrapped = wrap_retriever(retriever, store)

        wrapped.retrieve("query")
        event = store.recent()[0]
        assert "elapsed_ms" in event.metadata


# ── SQLiteStore unit tests ────────────────────────────────────────────────────

class TestSQLiteStore:
    def test_save_and_retrieve(self, store):
        from rag_debugger.store import RetrievalEvent
        event = RetrievalEvent(
            query="test query",
            label="test",
            chunks=[{"content": "chunk", "score": 0.8}],
        )
        store.save(event)
        events = store.recent()
        assert len(events) == 1
        assert events[0].query == "test query"
        assert events[0].chunks[0]["content"] == "chunk"

    def test_by_session(self, store):
        from rag_debugger.store import RetrievalEvent
        store.save(RetrievalEvent(query="q1", label="l", chunks=[], session_id="s1"))
        store.save(RetrievalEvent(query="q2", label="l", chunks=[], session_id="s2"))
        store.save(RetrievalEvent(query="q3", label="l", chunks=[], session_id="s1"))

        s1_events = store.by_session("s1")
        assert len(s1_events) == 2
        assert all(e.session_id == "s1" for e in s1_events)

    def test_recent_limit(self, store):
        from rag_debugger.store import RetrievalEvent
        for i in range(10):
            store.save(RetrievalEvent(query=f"q{i}", label="l", chunks=[]))
        assert len(store.recent(limit=5)) == 5

    def test_db_persists_across_instances(self, tmp_path):
        from rag_debugger.store import RetrievalEvent
        db = str(tmp_path / "persist.db")
        s1 = SQLiteStore(db_path=db)
        s1.save(RetrievalEvent(query="persisted", label="l", chunks=[]))
        s1.close()

        s2 = SQLiteStore(db_path=db)
        events = s2.recent()
        assert events[0].query == "persisted"
        s2.close()
