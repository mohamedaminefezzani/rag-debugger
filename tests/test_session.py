"""
Tests for session() context manager.
"""
import pytest
import threading
import time
from unittest.mock import patch
from rag_debugger.store import SQLiteStore, RetrievalEvent
from rag_debugger.session import Session, get_active_session_id, _get_stack
from rag_debugger.wrap import wrap_retriever


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_session_stack():
    """Ensure clean session stack before every test."""
    _get_stack().clear()
    yield
    _get_stack().clear()


@pytest.fixture
def store(tmp_path):
    return SQLiteStore(db_path=str(tmp_path / "test.db"))


def _make_retriever(store, docs=None):
    from unittest.mock import MagicMock
    retriever = MagicMock(spec=["retrieve"])
    retriever.retrieve.return_value = docs or [{"content": "doc", "score": 0.8}]
    with patch("rag_debugger.wrap._is_langchain", return_value=False), \
         patch("rag_debugger.wrap._is_llamaindex", return_value=False):
        return wrap_retriever(retriever, store)


# ── get_active_session_id ─────────────────────────────────────────────────────

class TestActiveSessionId:
    def test_none_when_no_session(self):
        assert get_active_session_id() is None

    def test_returns_id_inside_context(self, store):
        with Session(store, id="abc") as s:
            assert get_active_session_id() == "abc"

    def test_none_after_context_exits(self, store):
        with Session(store, id="abc"):
            pass
        assert get_active_session_id() is None

    def test_nested_sessions_innermost_wins(self, store):
        with Session(store, id="outer"):
            assert get_active_session_id() == "outer"
            with Session(store, id="inner"):
                assert get_active_session_id() == "inner"
            assert get_active_session_id() == "outer"
        assert get_active_session_id() is None


# ── Session context manager ───────────────────────────────────────────────────

class TestSessionContextManager:
    def test_auto_generates_id_if_not_provided(self, store):
        with Session(store) as s:
            assert s.id.startswith("sess-")
            assert len(s.id) > 5

    def test_uses_provided_id(self, store):
        with Session(store, id="conv-123") as s:
            assert s.id == "conv-123"

    def test_summary_unavailable_before_exit(self, store):
        s = Session(store, id="test")
        s.__enter__()
        with pytest.raises(RuntimeError, match="after the session context exits"):
            s.summary()
        s.__exit__(None, None, None)

    def test_summary_available_after_exit(self, store):
        with Session(store, id="test") as s:
            pass
        summary = s.summary()
        assert summary is not None
        assert summary.session_id == "test"

    def test_exception_inside_context_still_pops(self, store):
        try:
            with Session(store, id="err-session"):
                assert get_active_session_id() == "err-session"
                raise ValueError("oops")
        except ValueError:
            pass
        assert get_active_session_id() is None


# ── Retrieval events grouped under session ────────────────────────────────────

class TestSessionGrouping:
    def test_wrapped_retriever_picks_up_session(self, store):
        retriever = _make_retriever(store)
        with Session(store, id="grp-001") as s:
            retriever.retrieve("query inside session")

        events = store.by_session("grp-001")
        assert len(events) == 1
        assert events[0].query == "query inside session"

    def test_events_outside_session_have_no_session_id(self, store):
        retriever = _make_retriever(store)
        retriever.retrieve("outside query")
        events = store.recent()
        assert events[0].session_id is None

    def test_multiple_queries_grouped(self, store):
        retriever = _make_retriever(store)
        with Session(store, id="multi") as s:
            retriever.retrieve("query 1")
            retriever.retrieve("query 2")
            retriever.retrieve("query 3")

        events = store.by_session("multi")
        assert len(events) == 3
        assert [e.query for e in events] == ["query 1", "query 2", "query 3"]

    def test_two_sessions_dont_bleed(self, store):
        r1 = _make_retriever(store)
        r2 = _make_retriever(store)

        with Session(store, id="sess-A") as sA:
            r1.retrieve("A query")

        with Session(store, id="sess-B") as sB:
            r2.retrieve("B query")

        assert len(store.by_session("sess-A")) == 1
        assert len(store.by_session("sess-B")) == 1
        assert store.by_session("sess-A")[0].query == "A query"
        assert store.by_session("sess-B")[0].query == "B query"

    def test_explicit_session_id_takes_priority(self, store):
        """wrap_retriever(session_id=X) should override the active session."""
        from unittest.mock import MagicMock
        retriever = MagicMock(spec=["retrieve"])
        retriever.retrieve.return_value = [{"content": "doc", "score": 0.7}]

        with patch("rag_debugger.wrap._is_langchain", return_value=False), \
             patch("rag_debugger.wrap._is_llamaindex", return_value=False):
            wrapped = wrap_retriever(retriever, store, session_id="explicit-id")

        with Session(store, id="context-id"):
            wrapped.retrieve("query")

        # explicit id wins — event should be under explicit-id
        assert len(store.by_session("explicit-id")) == 1
        assert len(store.by_session("context-id")) == 0


# ── SessionSummary ────────────────────────────────────────────────────────────

class TestSessionSummary:
    def test_total_events_count(self, store):
        retriever = _make_retriever(store, docs=[{"content": "d", "score": 0.8}])
        with Session(store, id="sum-test") as s:
            retriever.retrieve("q1")
            retriever.retrieve("q2")
        assert s.summary().total_events == 2

    def test_queries_list(self, store):
        retriever = _make_retriever(store)
        with Session(store, id="q-test") as s:
            retriever.retrieve("first")
            retriever.retrieve("second")
        assert s.summary().queries == ["first", "second"]

    def test_avg_top_score(self, store):
        r1 = _make_retriever(store, docs=[{"content": "d", "score": 0.8}])
        r2 = _make_retriever(store, docs=[{"content": "d", "score": 0.6}])
        with Session(store, id="score-test") as s:
            r1.retrieve("q1")
            r2.retrieve("q2")
        summary = s.summary()
        assert summary.avg_top_score == pytest.approx(0.7, abs=0.01)

    def test_lowest_top_score(self, store):
        r1 = _make_retriever(store, docs=[{"content": "d", "score": 0.9}])
        r2 = _make_retriever(store, docs=[{"content": "d", "score": 0.4}])
        with Session(store, id="low-test") as s:
            r1.retrieve("q1")
            r2.retrieve("q2")
        assert s.summary().lowest_top_score == pytest.approx(0.4, abs=0.01)

    def test_gap_count(self, store):
        r_good = _make_retriever(store, docs=[{"content": "d", "score": 0.9}])
        r_bad  = _make_retriever(store, docs=[{"content": "d", "score": 0.3}])
        with Session(store, id="gap-test", gap_threshold=0.65) as s:
            r_good.retrieve("covered")
            r_bad.retrieve("gap query")
        assert s.summary().gap_count == 1

    def test_empty_session_summary(self, store):
        with Session(store, id="empty") as s:
            pass
        summary = s.summary()
        assert summary.total_events == 0
        assert summary.avg_top_score == 0.0
        assert summary.gap_count == 0

    def test_str_representation(self, store):
        retriever = _make_retriever(store)
        with Session(store, id="str-test", user="user-99") as s:
            retriever.retrieve("some query")
        text = str(s.summary())
        assert "str-test" in text
        assert "user-99" in text
        assert "events" in text

    def test_duration_ms_positive(self, store):
        with Session(store, id="dur-test") as s:
            time.sleep(0.01)
        assert s.summary().duration_ms >= 10


# ── Thread safety ─────────────────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_sessions_isolated(self, store):
        """Two threads with different sessions must not bleed into each other."""
        results = {}

        def run(session_id, query, delay):
            retriever = _make_retriever(store)
            with Session(store, id=session_id):
                time.sleep(delay)
                retriever.retrieve(query)
            results[session_id] = store.by_session(session_id)

        t1 = threading.Thread(target=run, args=("thread-A", "query A", 0.02))
        t2 = threading.Thread(target=run, args=("thread-B", "query B", 0.01))
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert len(results["thread-A"]) == 1
        assert len(results["thread-B"]) == 1
        assert results["thread-A"][0].query == "query A"
        assert results["thread-B"][0].query == "query B"
