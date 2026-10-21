"""
gap_detector.py - core gap detection logic for rag-debugger-amine
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Protocol


# ── Protocols ─────────────────────────────────────────────────────────────────

class LLMClientProtocol(Protocol):
    """Minimum interface any LLM client must implement to work with GapDetector.

    Implement this to bring your own provider (OpenAI, Anthropic, Mistral, …).

    Example
    -------
    class MyClient:
        def complete(self, prompt: str, max_tokens: int = 1024) -> str:
            ...  # call your API, return the text string
    """
    def complete(self, prompt: str, max_tokens: int = 1024) -> str: ...


class EmbedderProtocol(Protocol):
    """Any object that can embed text.  embed() and embed_document() must
    return a list[float].  Implementing embed() alone is enough - the default
    embed_document() falls back to embed() when not present."""

    def embed(self, text: str) -> list[float]: ...


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ChunkResult:
    content: str
    score: float
    chunk_id: str = ""
    reranked: bool = False       # True when score was confirmed/denied by LLM


@dataclass
class SubIntent:
    text: str
    best_score: float = 0.0
    covered: bool = False
    best_chunk: Optional[ChunkResult] = None


@dataclass
class GapReport:
    has_gap: bool
    coverage_score: float          # fraction of sub-intents covered
    overall_score: float           # min sub-intent score
    sub_intents: List[SubIntent] = field(default_factory=list)
    missing_topics: List[str] = field(default_factory=list)
    suggestion: str = ""
    priority: float = 0.5

    def __str__(self) -> str:
        tag = "[GAP DETECTED]" if self.has_gap else "[OK]"
        lines = [
            f"{tag} coverage={self.coverage_score:.0%}  "
            f"worst_score={self.overall_score:.2f}  priority={self.priority:.2f}"
        ]
        if self.missing_topics:
            lines.append(f"  Missing: {', '.join(self.missing_topics)}")
        if self.suggestion:
            lines.append(f"  Fix: {self.suggestion}")
        for si in self.sub_intents:
            mark = "✓" if si.covered else "✗"
            lines.append(f"    {mark} [{si.best_score:.2f}] {si.text}")
        return "\n".join(lines)


# ── Cosine similarity ─────────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ── GeminiClient ──────────────────────────────────────────────────────────────

class GeminiClient:
    """Thin wrapper around google-generativeai for LLM completions and
    embeddings.  Implements EmbedderProtocol so it can be passed as the
    default embedder to GapDetector."""

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "gemini-3.5-flash-lite",
                 embed_model: str = "models/gemini-embedding-001"):
        import google.generativeai as genai  # type: ignore
        import os
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not set")
        genai.configure(api_key=key)
        self._genai = genai
        self._model = genai.GenerativeModel(model)
        self._embed_model = embed_model

    def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        resp = self._model.generate_content(
            prompt,
            generation_config=self._genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.0,
            ),
        )
        return resp.text.strip()

    def embed(self, text: str) -> list[float]:
        result = self._genai.embed_content(
            model=self._embed_model,
            content=text,
            task_type="RETRIEVAL_QUERY",
        )
        return result["embedding"]

    def embed_document(self, text: str) -> list[float]:
        result = self._genai.embed_content(
            model=self._embed_model,
            content=text,
            task_type="RETRIEVAL_DOCUMENT",
        )
        return result["embedding"]


# ── LiteLLMClient ────────────────────────────────────────────────────────────

class LiteLLMClient:
    """LLM client backed by LiteLLM - supports 100+ providers behind one interface.

    Requires:  pip install rag-debugger-amine[litellm]
               (installs litellm + any provider SDK you need)

    Parameters
    ----------
    model : str
        Any LiteLLM model string, e.g.:
          "gpt-4o-mini"            → OpenAI  (OPENAI_API_KEY)
          "claude-3-5-haiku-20241022" → Anthropic (ANTHROPIC_API_KEY)
          "mistral/mistral-small"  → Mistral (MISTRAL_API_KEY)
          "ollama/llama3.2"        → local Ollama (no key needed)
          "gemini/gemini-2.5-flash"→ Gemini via LiteLLM (GEMINI_API_KEY)
    embed_model : str
        Any LiteLLM embedding model string, e.g.:
          "text-embedding-3-small" → OpenAI
          "cohere/embed-english-v3.0" → Cohere
          "ollama/nomic-embed-text"   → local Ollama
          None → embeddings disabled (pass a separate embedder= to GapDetector)

    Examples
    --------
    # OpenAI for both LLM and embeddings
    client = LiteLLMClient("gpt-4o-mini", embed_model="text-embedding-3-small")
    detector = GapDetector(client)

    # Anthropic LLM + local Ollama embeddings
    llm = LiteLLMClient("claude-3-5-haiku-20241022", embed_model=None)
    emb = LiteLLMClient("ollama/llama3.2", embed_model="ollama/nomic-embed-text")
    detector = GapDetector(llm, embedder=emb)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        embed_model: Optional[str] = "text-embedding-3-small",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        try:
            import litellm  # type: ignore
        except ImportError:
            raise ImportError(
                "LiteLLMClient requires litellm.\n"
                "Install it with:  pip install rag-debugger-amine[litellm]"
            )
        self._litellm = litellm
        self._model = model
        self._embed_model = embed_model
        self._api_key = api_key
        self._api_base = api_base

    def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        kwargs: dict = dict(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base
        resp = self._litellm.completion(**kwargs)
        return resp.choices[0].message.content.strip()

    def embed(self, text: str) -> list[float]:
        if not self._embed_model:
            raise ValueError(
                "embed_model is None - pass embed_model= to LiteLLMClient "
                "or use a separate embedder= on GapDetector."
            )
        kwargs: dict = dict(model=self._embed_model, input=[text])
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base
        resp = self._litellm.embedding(**kwargs)
        return resp.data[0]["embedding"]

    def embed_document(self, text: str) -> list[float]:
        """Same as embed() - LiteLLM handles task_type internally per provider."""
        return self.embed(text)


# ── GapDetector ───────────────────────────────────────────────────────────────

class GapDetector:
    """Detects knowledge-base gaps for a given query + retrieved chunks.

    Parameters
    ----------
    client : LLMClientProtocol
        Any object with a ``complete(prompt, max_tokens) -> str`` method.
        Built-in options: GeminiClient, LiteLLMClient.
        Used for LLM reasoning steps (decompose / rerank / synthesize).
    threshold : float
        Cosine similarity threshold below which a sub-intent is uncovered.
    history_store : optional
        Must implement ``count_similar_gaps(missing_topics, window_days) -> int``.
    embedder : optional
        Any object with ``.embed(text) -> list[float]`` and optionally
        ``.embed_document(text) -> list[float]``.  When not provided,
        ``client`` is used for embeddings (backward-compatible default).

    Examples
    --------
    # Gemini (default, free tier):
    detector = GapDetector(GeminiClient())

    # OpenAI via LiteLLM:
    client = LiteLLMClient("gpt-4o-mini", embed_model="text-embedding-3-small")
    detector = GapDetector(client)

    # Anthropic LLM + local HF embedder:
    llm = LiteLLMClient("claude-3-5-haiku-20241022", embed_model=None)
    detector = GapDetector(llm, embedder=my_hf_embedder)

    # Fully custom - bring any object that has .complete():
    detector = GapDetector(my_custom_client)
    """

    def __init__(
        self,
        client: GeminiClient,
        threshold: float = 0.65,
        history_store=None,
        embedder: Optional[Any] = None,
    ):
        self.client = client
        self.threshold = threshold
        self.history = history_store
        # If a custom embedder is provided, use it; otherwise fall back to client.
        self._embedder = embedder if embedder is not None else client

    # ── public ────────────────────────────────────────────────────────────────

    def analyze(self, query: str, chunks: list[dict]) -> GapReport:
        """Run gap analysis on *query* against *chunks*.

        chunks items: ``{"content": str, "score": float, ...}``
        """
        sub_intents = self._decompose(query)

        if not chunks:
            for si in sub_intents:
                si.best_score = 0.0
                si.covered = False
            return GapReport(
                has_gap=True,
                coverage_score=0.0,
                overall_score=0.0,
                sub_intents=sub_intents,
                missing_topics=[si.text for si in sub_intents],
                suggestion="No chunks retrieved - add documents covering the query.",
                priority=self._priority([si.text for si in sub_intents]),
            )

        chunk_results = self._to_chunk_results(chunks)
        chunk_vecs = [self._embed_doc(cr.content) for cr in chunk_results]

        for si in sub_intents:
            si_vec = self._embed_query(si.text)
            scores = [_cosine(si_vec, cv) for cv in chunk_vecs]
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            si.best_score = scores[best_idx]
            si.best_chunk = chunk_results[best_idx]

            # Rerank borderline scores with LLM yes/no
            if self.threshold * 0.9 <= si.best_score <= self.threshold * 1.1:
                si.best_score, si.best_chunk.reranked = self._rerank(
                    si.text, si.best_chunk.content, si.best_score
                )

            si.covered = si.best_score >= self.threshold

        covered = [si for si in sub_intents if si.covered]
        uncovered = [si for si in sub_intents if not si.covered]
        coverage = len(covered) / len(sub_intents) if sub_intents else 1.0
        overall = min(si.best_score for si in sub_intents) if sub_intents else 0.0
        has_gap = len(uncovered) > 0

        missing_topics: list[str] = []
        suggestion = ""
        if has_gap:
            missing_topics, suggestion = self._synthesize(
                query, [si.text for si in uncovered]
            )

        return GapReport(
            has_gap=has_gap,
            coverage_score=coverage,
            overall_score=overall,
            sub_intents=sub_intents,
            missing_topics=missing_topics,
            suggestion=suggestion,
            priority=self._priority(missing_topics) if has_gap else 0.0,
        )

    # ── private ───────────────────────────────────────────────────────────────

    def _embed_query(self, text: str) -> list[float]:
        """Embed a query using the configured embedder."""
        return self._embedder.embed(text)

    def _embed_doc(self, text: str) -> list[float]:
        """Embed a document using the configured embedder.
        Falls back to embed() when embed_document() is not available."""
        embed_doc = getattr(self._embedder, "embed_document", None)
        if callable(embed_doc):
            return embed_doc(text)
        return self._embedder.embed(text)

    def _decompose(self, query: str) -> list[SubIntent]:
        prompt = (
            "Break this user query into atomic sub-intents (1–5 short phrases). "
            "Return ONLY a JSON array of objects with a 'text' key, no extra text.\n\n"
            f"Query: {query}"
        )
        raw = self.client.complete(prompt, max_tokens=1024)
        try:
            # Strip markdown fences if present
            cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
            items = json.loads(cleaned)
            intents = [SubIntent(text=item["text"]) for item in items if "text" in item]
            return intents if intents else [SubIntent(text=query)]
        except (json.JSONDecodeError, KeyError):
            return [SubIntent(text=query)]

    def _rerank(self, intent: str, chunk: str, score: float) -> tuple[float, bool]:
        prompt = (
            f"Does this chunk adequately answer the question?\n\n"
            f"Question: {intent}\n\nChunk: {chunk}\n\n"
            "Reply with exactly YES or NO."
        )
        try:
            answer = self.client.complete(prompt, max_tokens=8).strip().upper()
        except (ValueError, Exception):
            # Safety filter or other API error - skip rerank, keep raw score.
            # finish_reason=2 (SAFETY) is the most common trigger here.
            return score, False
        if answer.startswith("YES"):
            return max(score, self.threshold), True
        return min(score, self.threshold - 0.01), True

    def _synthesize(self, query: str, uncovered: list[str]) -> tuple[list[str], str]:
        prompt = (
            "You are a knowledge base curator. A RAG system could not answer part of a user query.\n\n"
            f"User query: {query}\n"
            "Uncovered sub-intents (what the KB is missing):\n"
            + "\n".join(f"- {t}" for t in uncovered)
            + "\n\n"
            "Your job:\n"
            "1. Rewrite each uncovered sub-intent as a precise, searchable topic name "
            "(e.g. 'job application process' not 'how to apply for the job').\n"
            "2. Write a concrete suggestion describing what document or section to add - "
            "include the content type (FAQ, guide, policy), key points it should cover, "
            "and any user-facing actions it should explain.\n\n"
            "Return ONLY valid JSON with two keys:\n"
            '  "missing_topics": list of rewritten topic names (strings)\n'
            '  "suggestion": one paragraph describing what to write and why\n'
            "No extra text outside the JSON."
        )
        raw = self.client.complete(prompt, max_tokens=512)
        try:
            cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
            data = json.loads(cleaned)
            if not isinstance(data, dict):
                raise ValueError("expected dict")
            return data.get("missing_topics", uncovered), data.get("suggestion", "")
        except (json.JSONDecodeError, KeyError, ValueError):
            return uncovered, f"Add docs covering: {', '.join(uncovered)}."

    def _priority(self, missing_topics: list[str]) -> float:
        if not self.history or not missing_topics:
            return 0.5
        count = self.history.count_similar_gaps(missing_topics, window_days=30)
        return min(count / 50.0, 1.0)

    @staticmethod
    def _to_chunk_results(chunks: list[dict]) -> list[ChunkResult]:
        results = []
        for c in chunks:
            content = c.get("content") or c.get("text") or str(c)
            score = float(c.get("score", 0.0))
            chunk_id = c.get("chunk_id") or c.get("id") or ""
            results.append(ChunkResult(content=content, score=score, chunk_id=chunk_id))
        return results
