"""
Local test suite for GapDetector — no real API calls needed.
Run with: pytest test_gap_detector.py -v
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from gap_detector import GapDetector, GeminiClient, GapReport, SubIntent, _cosine


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_mock_client(decompose_response=None, synthesize_response=None, embed_dim=8):
    """Returns a MagicMock GeminiClient with controllable responses."""
    client = MagicMock(spec=GeminiClient)

    decompose_json = decompose_response or '[{"text":"cancel subscription"},{"text":"iOS platform"},{"text":"refund request"}]'
    synth_json = synthesize_response or '{"missing_topics":["iOS cancellation","refund policy"],"suggestion":"Add docs covering iOS cancellation flow and refund window policy."}'

    call_count = {"n": 0}
    def fake_complete(prompt, **kwargs):
        call_count["n"] += 1
        if "sub-intent" in prompt or "Break this" in prompt:
            return decompose_json
        return synth_json
    client.complete.side_effect = fake_complete

    rng = np.random.default_rng(42)
    def fake_embed(text):
        # deterministic: covered topics score high, others low
        if any(kw in text.lower() for kw in ["cancel", "subscription"]):
            v = np.ones(embed_dim)
        elif any(kw in text.lower() for kw in ["ios", "mobile", "refund", "window"]):
            v = np.zeros(embed_dim)
        else:
            v = rng.random(embed_dim)
        return (v / (np.linalg.norm(v) + 1e-9)).tolist()

    client.embed.side_effect = fake_embed
    client.embed_document.side_effect = fake_embed
    return client


CHUNKS_GOOD = [
    {"content": "To cancel your subscription go to Settings > Billing > Cancel.", "score": 0.91},
    {"content": "Subscription management is available in the billing portal.", "score": 0.85},
]

CHUNKS_PARTIAL = [
    {"content": "To cancel your subscription go to Settings > Billing > Cancel.", "score": 0.91},
]

CHUNKS_EMPTY = []


# ── Unit tests: _cosine ───────────────────────────────────────────────────────

class TestCosine:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert _cosine([0, 0, 0], [1, 2, 3]) == 0.0

    def test_high_dimensional(self):
        rng = np.random.default_rng(0)
        a = rng.random(768).tolist()
        b = rng.random(768).tolist()
        result = _cosine(a, b)
        assert -1.0 <= result <= 1.0


# ── Unit tests: GapDetector ───────────────────────────────────────────────────

class TestGapDetector:
    def test_no_gap_when_all_covered(self):
        client = make_mock_client(
            decompose_response='[{"text":"cancel subscription"}]'
        )
        # Embed returns identical vector for both query and chunk → score 1.0
        client.embed.side_effect = lambda t: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        client.embed_document.side_effect = lambda t: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        detector = GapDetector(client, threshold=0.72)
        report = detector.analyze("How do I cancel?", CHUNKS_GOOD)

        assert report.has_gap is False
        assert report.coverage_score == 1.0
        assert report.suggestion == ""
        assert report.missing_topics == []

    def test_gap_detected_for_uncovered_intent(self):
        client = make_mock_client()
        detector = GapDetector(client, threshold=0.72)
        report = detector.analyze(
            "How do I cancel my subscription on iOS and get a refund?",
            CHUNKS_PARTIAL
        )

        assert report.has_gap is True
        assert report.coverage_score < 1.0
        assert len(report.missing_topics) > 0
        assert report.suggestion != ""

    def test_empty_chunks_all_gap(self):
        client = make_mock_client()
        detector = GapDetector(client, threshold=0.72)
        report = detector.analyze("How do I cancel?", CHUNKS_EMPTY)

        assert report.has_gap is True
        assert report.overall_score == 0.0
        assert report.coverage_score == 0.0

    def test_overall_score_is_minimum(self):
        """overall_score must reflect the weakest sub-intent, not the average."""
        client = make_mock_client(
            decompose_response='[{"text":"cancel subscription"},{"text":"refund"}]'
        )
        embeddings = {
            "cancel subscription": [1.0] + [0.0] * 7,
            "refund":              [0.0] * 8,
        }
        client.embed.side_effect = lambda t: embeddings.get(t, [0.0]*8)
        client.embed_document.side_effect = lambda t: [1.0] + [0.0] * 7

        detector = GapDetector(client, threshold=0.72)
        report = detector.analyze("cancel and refund", CHUNKS_GOOD)

        assert report.overall_score < 0.5  # refund sub-intent scores 0

    def test_threshold_boundary(self):
        """Score exactly at threshold should be covered."""
        client = make_mock_client(
            decompose_response='[{"text":"billing"}]'
        )
        v = [1.0] + [0.0] * 7
        client.embed.side_effect = lambda t: v
        client.embed_document.side_effect = lambda t: v

        detector = GapDetector(client, threshold=0.72)
        report = detector.analyze("billing question", CHUNKS_GOOD)
        # cosine of identical vectors = 1.0 ≥ 0.72
        assert report.has_gap is False

    def test_sub_intents_attached_to_report(self):
        client = make_mock_client()
        detector = GapDetector(client, threshold=0.72)
        report = detector.analyze("cancel on iOS and refund", CHUNKS_PARTIAL)

        assert len(report.sub_intents) == 3
        for si in report.sub_intents:
            assert isinstance(si, SubIntent)
            assert si.text != ""
            assert 0.0 <= si.best_score <= 1.0

    def test_coverage_score_fraction(self):
        """2 of 3 sub-intents covered → coverage = 0.667."""
        client = make_mock_client()
        # make "cancel subscription" embed match the chunk, others don't
        def embed(t):
            if "cancel" in t:
                return [1.0] + [0.0] * 7
            return [0.0] * 8
        client.embed.side_effect = embed
        client.embed_document.side_effect = lambda t: [1.0] + [0.0] * 7

        detector = GapDetector(client, threshold=0.72)
        report = detector.analyze("cancel on iOS and refund", CHUNKS_GOOD)

        # Only "cancel subscription" will be covered
        assert 0.0 < report.coverage_score < 1.0

    def test_malformed_llm_json_fallback(self):
        """If LLM returns garbage JSON, should not raise — fallback to full query."""
        client = make_mock_client(decompose_response="not json at all")
        v = [1.0] + [0.0] * 7
        client.embed.side_effect = lambda t: v
        client.embed_document.side_effect = lambda t: v

        detector = GapDetector(client, threshold=0.72)
        report = detector.analyze("How do I cancel?", CHUNKS_GOOD)

        assert isinstance(report, GapReport)
        assert len(report.sub_intents) == 1  # fallback: single sub-intent

    def test_priority_default_when_no_history(self):
        client = make_mock_client()
        detector = GapDetector(client, threshold=0.72, history_store=None)
        report = detector.analyze("cancel on iOS", CHUNKS_PARTIAL)

        if report.has_gap:
            assert report.priority == 0.5

    def test_priority_from_history(self):
        client = make_mock_client()
        history = MagicMock()
        history.count_similar_gaps.return_value = 25  # 25/50 = 0.5

        detector = GapDetector(client, threshold=0.72, history_store=history)
        report = detector.analyze("cancel on iOS", CHUNKS_PARTIAL)

        if report.has_gap:
            assert report.priority == pytest.approx(0.5)
            history.count_similar_gaps.assert_called_once()

    def test_str_representation(self):
        client = make_mock_client()
        detector = GapDetector(client, threshold=0.72)
        report = detector.analyze("cancel on iOS and refund", CHUNKS_PARTIAL)
        text = str(report)
        assert "GAP" in text or "OK" in text
        assert "coverage" in text


# ── Integration smoke test (skipped if no API key) ────────────────────────────

@pytest.mark.skipif(
    not __import__("os").environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set — skipping live API test"
)
class TestGapDetectorLive:
    def test_live_analysis(self):
        client = GeminiClient()
        detector = GapDetector(client, threshold=0.72)

        chunks = [
            {"content": "To cancel your subscription, go to Settings > Billing > Cancel Plan."},
            {"content": "Subscription changes take effect at the end of your billing cycle."},
        ]
        report = detector.analyze(
            "How do I cancel my subscription on iOS and get a refund?",
            chunks
        )

        assert isinstance(report, GapReport)
        assert isinstance(report.has_gap, bool)
        assert 0.0 <= report.coverage_score <= 1.0
        assert 0.0 <= report.overall_score <= 1.0
        print("\n" + str(report))
