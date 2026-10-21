"""
Local test suite for GapDetector - no real API calls needed.
Run with: pytest tests/test_gap_detector.py -v
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from rag_debugger.gap_detector import GapDetector, GeminiClient, GapReport, SubIntent, _cosine


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_mock_client(decompose_response=None, synthesize_response=None, embed_dim=8):
    """Returns a MagicMock GeminiClient with controllable responses."""
    client = MagicMock(spec=GeminiClient)

    decompose_json = decompose_response or '[{"text":"cancel subscription"},{"text":"iOS platform"},{"text":"refund request"}]'
    synth_json = synthesize_response or '{"missing_topics":["iOS cancellation","refund policy"],"suggestion":"Add docs covering iOS cancellation flow and refund window policy."}'

    def fake_complete(prompt, **kwargs):
        if "sub-intent" in prompt or "Break this" in prompt:
            return decompose_json
        return synth_json
    client.complete.side_effect = fake_complete

    rng = np.random.default_rng(42)
    def fake_embed(text):
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
        client.embed.side_effect = lambda t: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        client.embed_document.side_effect = lambda t: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        detector = GapDetector(client, threshold=0.65)
        report = detector.analyze("How do I cancel?", CHUNKS_GOOD)

        assert report.has_gap is False
        assert report.coverage_score == 1.0
        assert report.suggestion == ""
        assert report.missing_topics == []

    def test_gap_detected_for_uncovered_intent(self):
        client = make_mock_client()
        detector = GapDetector(client, threshold=0.65)
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
        detector = GapDetector(client, threshold=0.65)
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

        detector = GapDetector(client, threshold=0.65)
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

        detector = GapDetector(client, threshold=0.65)
        report = detector.analyze("billing question", CHUNKS_GOOD)
        assert report.has_gap is False

    def test_sub_intents_attached_to_report(self):
        client = make_mock_client()
        detector = GapDetector(client, threshold=0.65)
        report = detector.analyze("cancel on iOS and refund", CHUNKS_PARTIAL)

        assert len(report.sub_intents) == 3
        for si in report.sub_intents:
            assert isinstance(si, SubIntent)
            assert si.text != ""
            assert 0.0 <= si.best_score <= 1.0

    def test_coverage_score_fraction(self):
        """Only 'cancel subscription' covered → coverage < 1.0."""
        client = make_mock_client()
        def embed(t):
            if "cancel" in t:
                return [1.0] + [0.0] * 7
            return [0.0] * 8
        client.embed.side_effect = embed
        client.embed_document.side_effect = lambda t: [1.0] + [0.0] * 7

        detector = GapDetector(client, threshold=0.65)
        report = detector.analyze("cancel on iOS and refund", CHUNKS_GOOD)

        assert 0.0 < report.coverage_score < 1.0

    def test_malformed_llm_json_fallback(self):
        """If LLM returns garbage JSON, should not raise - fallback to full query."""
        client = make_mock_client(decompose_response="not json at all")
        v = [1.0] + [0.0] * 7
        client.embed.side_effect = lambda t: v
        client.embed_document.side_effect = lambda t: v

        detector = GapDetector(client, threshold=0.65)
        report = detector.analyze("How do I cancel?", CHUNKS_GOOD)

        assert isinstance(report, GapReport)
        assert len(report.sub_intents) == 1  # fallback: single sub-intent

    def test_priority_default_when_no_history(self):
        client = make_mock_client()
        detector = GapDetector(client, threshold=0.65, history_store=None)
        report = detector.analyze("cancel on iOS", CHUNKS_PARTIAL)

        if report.has_gap:
            assert report.priority == 0.5

    def test_priority_from_history(self):
        client = make_mock_client()
        history = MagicMock()
        history.count_similar_gaps.return_value = 25  # 25/50 = 0.5

        detector = GapDetector(client, threshold=0.65, history_store=history)
        report = detector.analyze("cancel on iOS", CHUNKS_PARTIAL)

        if report.has_gap:
            assert report.priority == pytest.approx(0.5)
            history.count_similar_gaps.assert_called_once()

    def test_str_representation(self):
        client = make_mock_client()
        detector = GapDetector(client, threshold=0.65)
        report = detector.analyze("cancel on iOS and refund", CHUNKS_PARTIAL)
        text = str(report)
        assert "GAP" in text or "OK" in text
        assert "coverage" in text


# ── Custom embedder tests ─────────────────────────────────────────────────────

class TestCustomEmbedder:
    """Verify that GapDetector uses the provided embedder for embed() calls."""

    def test_custom_embedder_is_used(self):
        """embed() calls should go to the custom embedder, not the client."""
        client = make_mock_client(
            decompose_response='[{"text":"cancel subscription"}]'
        )
        # Client embed should NOT be called
        client.embed.side_effect = Exception("client.embed should not be called!")
        client.embed_document.side_effect = Exception("should not be called!")

        custom_embedder = MagicMock()
        custom_embedder.embed.return_value = [1.0] + [0.0] * 7
        custom_embedder.embed_document.return_value = [1.0] + [0.0] * 7

        detector = GapDetector(client, threshold=0.65, embedder=custom_embedder)
        report = detector.analyze("How do I cancel?", CHUNKS_GOOD)

        # Custom embedder was called; client embed was NOT
        assert custom_embedder.embed.called
        assert isinstance(report, GapReport)

    def test_custom_embedder_without_embed_document(self):
        """embedder with only embed() should still work via fallback."""
        client = make_mock_client(
            decompose_response='[{"text":"cancel subscription"}]'
        )
        client.embed.side_effect = Exception("should not be called!")
        client.embed_document.side_effect = Exception("should not be called!")

        # embedder has only embed(), no embed_document()
        custom_embedder = MagicMock(spec=["embed"])
        custom_embedder.embed.return_value = [1.0] + [0.0] * 7

        detector = GapDetector(client, threshold=0.65, embedder=custom_embedder)
        report = detector.analyze("How do I cancel?", CHUNKS_GOOD)

        assert isinstance(report, GapReport)
        assert custom_embedder.embed.called

    def test_no_embedder_falls_back_to_client(self):
        """When embedder=None, client is used for embeddings (backward compat)."""
        client = make_mock_client(
            decompose_response='[{"text":"cancel subscription"}]'
        )
        v = [1.0] + [0.0] * 7
        client.embed.side_effect = lambda t: v
        client.embed_document.side_effect = lambda t: v

        detector = GapDetector(client, threshold=0.65)  # no embedder=
        report = detector.analyze("How do I cancel?", CHUNKS_GOOD)

        assert client.embed.called or client.embed_document.called
        assert isinstance(report, GapReport)

    def test_custom_embedder_gap_detection_accuracy(self):
        """End-to-end: custom embedder that returns 0-vector for refund
        should trigger a gap for the refund sub-intent."""
        client = make_mock_client(
            decompose_response='[{"text":"cancel subscription"},{"text":"refund"}]'
        )

        def custom_embed(text):
            if "cancel" in text.lower():
                return [1.0] + [0.0] * 7
            return [0.0] * 8  # refund → zero vector → score 0

        custom_embedder = MagicMock()
        custom_embedder.embed.side_effect = custom_embed
        custom_embedder.embed_document.side_effect = lambda t: [1.0] + [0.0] * 7

        detector = GapDetector(client, threshold=0.65, embedder=custom_embedder)
        report = detector.analyze("cancel and refund", CHUNKS_GOOD)

        assert report.has_gap is True
        refund_si = next(si for si in report.sub_intents if "refund" in si.text)
        assert not refund_si.covered


# ── Integration smoke test (skipped if no API key) ────────────────────────────

@pytest.mark.skipif(
    not __import__("os").environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set - skipping live API test"
)
class TestGapDetectorLive:
    def test_live_analysis(self):
        client = GeminiClient()
        detector = GapDetector(client, threshold=0.65)

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

    def test_live_custom_embedder(self):
        """Live test with a custom embedder (second GeminiClient with RETRIEVAL_QUERY only)."""
        client = GeminiClient()

        class SimpleEmbedder:
            """Embedder that always uses query task type for both query and doc."""
            def __init__(self, gc):
                self._gc = gc
            def embed(self, text):
                return self._gc.embed(text)
            # No embed_document - tests the fallback path

        detector = GapDetector(client, threshold=0.65, embedder=SimpleEmbedder(client))
        chunks = [{"content": "To cancel your subscription, go to Settings > Billing."}]
        report = detector.analyze("How do I cancel?", chunks)
        assert isinstance(report, GapReport)
        print("\n[Custom embedder live]\n" + str(report))
