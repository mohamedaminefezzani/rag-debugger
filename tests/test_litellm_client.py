"""
Tests for LiteLLMClient - all mocked, no real API calls.
Run with: pytest tests/test_litellm_client.py -v
"""
import pytest
from unittest.mock import MagicMock, patch
from rag_debugger.gap_detector import LiteLLMClient, GapDetector, GapReport


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_litellm_mock(complete_text="YES", embed_vec=None):
    """Patch litellm at import time inside LiteLLMClient."""
    mock = MagicMock()

    # completion() response shape
    choice = MagicMock()
    choice.message.content = complete_text
    mock.completion.return_value = MagicMock(choices=[choice])

    # embedding() response shape
    vec = embed_vec or [1.0] + [0.0] * 7
    mock.embedding.return_value = MagicMock(data=[{"embedding": vec}])

    return mock


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestLiteLLMClient:
    def test_complete_returns_string(self):
        with patch.dict("sys.modules", {"litellm": make_litellm_mock("hello world")}):
            client = LiteLLMClient("gpt-4o-mini")
            result = client.complete("say hello")
        assert result == "hello world"

    def test_complete_passes_model_and_max_tokens(self):
        mock_litellm = make_litellm_mock()
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            client = LiteLLMClient("gpt-4o-mini")
            client.complete("test prompt", max_tokens=256)
        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["max_tokens"] == 256
        assert call_kwargs["temperature"] == 0.0

    def test_embed_returns_vector(self):
        vec = [0.1, 0.2, 0.3, 0.4]
        with patch.dict("sys.modules", {"litellm": make_litellm_mock(embed_vec=vec)}):
            client = LiteLLMClient("gpt-4o-mini", embed_model="text-embedding-3-small")
            result = client.embed("hello")
        assert result == vec

    def test_embed_document_same_as_embed(self):
        vec = [0.5] * 4
        with patch.dict("sys.modules", {"litellm": make_litellm_mock(embed_vec=vec)}):
            client = LiteLLMClient("gpt-4o-mini", embed_model="text-embedding-3-small")
            assert client.embed("x") == client.embed_document("x")

    def test_embed_raises_when_no_embed_model(self):
        with patch.dict("sys.modules", {"litellm": make_litellm_mock()}):
            client = LiteLLMClient("gpt-4o-mini", embed_model=None)
            with pytest.raises(ValueError, match="embed_model is None"):
                client.embed("test")

    def test_api_key_forwarded(self):
        mock_litellm = make_litellm_mock()
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            client = LiteLLMClient("gpt-4o-mini", api_key="sk-test-123")
            client.complete("hello")
        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs["api_key"] == "sk-test-123"

    def test_api_base_forwarded(self):
        mock_litellm = make_litellm_mock()
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            client = LiteLLMClient("ollama/llama3.2",
                                   embed_model=None,
                                   api_base="http://localhost:11434")
            client.complete("hello")
        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs["api_base"] == "http://localhost:11434"

    def test_import_error_when_litellm_missing(self):
        with patch.dict("sys.modules", {"litellm": None}):
            with pytest.raises(ImportError, match="pip install"):
                LiteLLMClient("gpt-4o-mini")


# ── Integration: LiteLLMClient as GapDetector client ─────────────────────────

class TestLiteLLMWithGapDetector:
    def _make_client(self, decompose_json=None, synth_json=None, embed_vec=None):
        decompose = decompose_json or '[{"text":"cancel subscription"},{"text":"refund"}]'
        synth = synth_json or '{"missing_topics":["refund policy"],"suggestion":"Add refund docs."}'
        vec = embed_vec or [1.0] + [0.0] * 7

        def fake_complete(model, messages, **kwargs):
            prompt = messages[0]["content"]
            if "sub-intent" in prompt or "Break this" in prompt:
                txt = decompose
            else:
                txt = synth
            choice = MagicMock()
            choice.message.content = txt
            return MagicMock(choices=[choice])

        mock_litellm = MagicMock()
        mock_litellm.completion.side_effect = fake_complete
        mock_litellm.embedding.return_value = MagicMock(data=[{"embedding": vec}])
        return mock_litellm

    def test_gap_detected_via_litellm(self):
        mock_litellm = self._make_client()
        chunks = [{"content": "To cancel go to Settings > Billing.", "score": 0.9}]

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            client = LiteLLMClient("gpt-4o-mini",
                                   embed_model="text-embedding-3-small")
            # give refund sub-intent a zero vector so it scores 0
            call_count = {"n": 0}
            def embed_side(model, input, **kw):
                call_count["n"] += 1
                text = input[0]
                if "cancel" in text.lower():
                    v = [1.0] + [0.0] * 7
                else:
                    v = [0.0] * 8
                return MagicMock(data=[{"embedding": v}])
            mock_litellm.embedding.side_effect = embed_side

            detector = GapDetector(client, threshold=0.65)
            report = detector.analyze("cancel and refund", chunks)

        assert isinstance(report, GapReport)
        assert report.has_gap is True

    def test_litellm_as_llm_only_with_custom_embedder(self):
        """LiteLLMClient with embed_model=None + separate custom embedder."""
        mock_litellm = self._make_client()
        chunks = [{"content": "Cancel at Settings > Billing.", "score": 0.9}]

        custom_embedder = MagicMock()
        custom_embedder.embed.return_value = [1.0] + [0.0] * 7
        custom_embedder.embed_document.return_value = [1.0] + [0.0] * 7

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            client = LiteLLMClient("gpt-4o-mini", embed_model=None)
            detector = GapDetector(client, threshold=0.65, embedder=custom_embedder)
            report = detector.analyze("How do I cancel?", chunks)

        assert isinstance(report, GapReport)
        assert custom_embedder.embed.called
        # litellm.embedding should NOT have been called
        mock_litellm.embedding.assert_not_called()
