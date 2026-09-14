"""
Tests for dashboard API endpoint — no browser, no uvicorn needed.
"""
import pytest
import json
from unittest.mock import patch, MagicMock
from rag_debugger.store import SQLiteStore, RetrievalEvent
from rag_debugger.dashboard import _build_app

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


@pytest.fixture
def store(tmp_path):
    s = SQLiteStore(db_path=str(tmp_path / "test.db"))
    # seed with realistic data
    s.save(RetrievalEvent(
        query="How do I cancel?",
        label="support-kb",
        session_id="sess-001",
        user="alice",
        chunks=[
            {"content": "Go to Settings > Billing > Cancel", "score": 0.88},
            {"content": "Changes take effect end of cycle", "score": 0.71},
        ],
        metadata={"elapsed_ms": 42},
    ))
    s.save(RetrievalEvent(
        query="How do I get a refund?",
        label="support-kb",
        session_id="sess-001",
        user="alice",
        chunks=[
            {"content": "Go to Settings > Billing > Cancel", "score": 0.41},
        ],
        metadata={"elapsed_ms": 38},
    ))
    s.save(RetrievalEvent(
        query="What is the price?",
        label="support-kb",
        session_id="sess-002",
        chunks=[
            {"content": "Pro plan costs $29/month", "score": 0.91},
        ],
        metadata={"elapsed_ms": 35},
    ))
    return s


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestDashboardAPI:
    @pytest.fixture
    def client(self, store):
        app = _build_app(store, project="test-project")
        return TestClient(app)

    def test_index_returns_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "rag-debugger" in r.text
        assert "<html" in r.text

    def test_data_endpoint_returns_json(self, client):
        r = client.get("/api/data")
        assert r.status_code == 200
        data = r.json()
        assert "stats" in data
        assert "sessions" in data
        assert "project" in data

    def test_project_name_in_response(self, client):
        r = client.get("/api/data")
        assert r.json()["project"] == "test-project"

    def test_stats_total_events(self, client):
        r = client.get("/api/data")
        assert r.json()["stats"]["total_events"] == 3

    def test_stats_total_sessions(self, client):
        r = client.get("/api/data")
        assert r.json()["stats"]["total_sessions"] == 2

    def test_gap_events_counted(self, client):
        r = client.get("/api/data")
        # refund query has top score 0.41 < 0.65 → 1 gap
        assert r.json()["stats"]["gap_events"] == 1

    def test_avg_top_score_reasonable(self, client):
        r = client.get("/api/data")
        avg = r.json()["stats"]["avg_top_score"]
        assert 0.0 < avg <= 1.0

    def test_sessions_list_structure(self, client):
        r = client.get("/api/data")
        sessions = r.json()["sessions"]
        assert len(sessions) == 2
        for s in sessions:
            assert "id" in s
            assert "event_count" in s
            assert "gap_count" in s
            assert "avg_top_score" in s
            assert "lowest_top_score" in s
            assert "events" in s

    def test_session_event_count(self, client):
        r = client.get("/api/data")
        sessions = {s["id"]: s for s in r.json()["sessions"]}
        assert sessions["sess-001"]["event_count"] == 2
        assert sessions["sess-002"]["event_count"] == 1

    def test_session_gap_count(self, client):
        r = client.get("/api/data")
        sessions = {s["id"]: s for s in r.json()["sessions"]}
        assert sessions["sess-001"]["gap_count"] == 1  # refund query
        assert sessions["sess-002"]["gap_count"] == 0

    def test_events_contain_chunks(self, client):
        r = client.get("/api/data")
        sessions = {s["id"]: s for s in r.json()["sessions"]}
        events = sessions["sess-001"]["events"]
        assert all("chunks" in e for e in events)
        assert all("query" in e for e in events)

    def test_empty_store_returns_zeros(self, tmp_path):
        empty_store = SQLiteStore(db_path=str(tmp_path / "empty.db"))
        app = _build_app(empty_store, project="empty")
        client = TestClient(app)
        r = client.get("/api/data")
        stats = r.json()["stats"]
        assert stats["total_events"] == 0
        assert stats["total_sessions"] == 0
        assert stats["gap_events"] == 0
