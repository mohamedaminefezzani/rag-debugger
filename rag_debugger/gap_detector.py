from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np
import json
import os
import re

@dataclass
class SubIntent:
    text: str
    embedding: list[float] = field(default_factory=list)
    best_score: float = 0.0
    best_chunk: Optional[str] = None
    covered: bool = False

@dataclass
class GapReport:
    has_gap: bool
    overall_score: float
    coverage_score: float
    missing_topics: list[str]
    suggestion: str
    priority: float
    sub_intents: list[SubIntent]

    def __str__(self):
        status = "GAP DETECTED" if self.has_gap else "OK"
        lines = [
            f"[{status}] coverage={self.coverage_score:.0%}  worst_score={self.overall_score:.2f}  priority={self.priority:.2f}",
        ]
        if self.missing_topics:
            lines.append(f"  Missing: {', '.join(self.missing_topics)}")
        if self.suggestion:
            lines.append(f"  Fix: {self.suggestion}")
        for si in self.sub_intents:
            mark = "✓" if si.covered else "✗"
            lines.append(f"    {mark} [{si.best_score:.2f}] {si.text}")
        return "\n".join(lines)


class GeminiClient:
    """Thin wrapper around the Gemini generative + embedding APIs."""

    def __init__(self, api_key: Optional[str] = None):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Run: pip install google-generativeai")

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("Set GEMINI_API_KEY env var or pass api_key=")

        genai.configure(api_key=key)
        self._genai = genai
        self._model = genai.GenerativeModel("gemini-2.5-flash")
        self._embed_model = "gemini-embedding-001"

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        resp = self._model.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tokens, "temperature": 0.1},
        )
        return resp.text.strip()

    def embed(self, text: str) -> list[float]:
        result = self._genai.embed_content(
            model=self._embed_model,
            content=text,
            task_type="retrieval_query",
        )
        return result["embedding"]

    def embed_document(self, text: str) -> list[float]:
        result = self._genai.embed_content(
            model=self._embed_model,
            content=text,
            task_type="retrieval_document",
        )
        return result["embedding"]


class GapDetector:
    def __init__(
        self,
        client: GeminiClient,
        threshold: float = 0.65,
        history_store=None,
    ):
        self.client = client
        self.threshold = threshold
        self.history = history_store

    def analyze(self, query: str, chunks: list[dict]) -> GapReport:
        # Stage 1 — decompose query into sub-intents
        sub_intents = self._decompose(query)

        # Stage 2 — score each sub-intent against retrieved chunks
        chunk_embeddings = [self.client.embed_document(c["content"]) for c in chunks]

        for si in sub_intents:
            si.embedding = self.client.embed(si.text)
            scores = [_cosine(si.embedding, ce) for ce in chunk_embeddings]
            if scores:
                best_idx = int(np.argmax(scores))
                si.best_score = round(scores[best_idx], 4)
                si.best_chunk = chunks[best_idx]["content"]
            si.covered = si.best_score >= self.threshold

        # Stage 3 — synthesize suggestion for uncovered sub-intents
        uncovered = [si for si in sub_intents if not si.covered]
        has_gap = len(uncovered) > 0
        overall = min(si.best_score for si in sub_intents) if sub_intents else 0.0
        coverage = sum(si.covered for si in sub_intents) / max(len(sub_intents), 1)

        suggestion, missing_topics = "", []
        if has_gap:
            suggestion, missing_topics = self._synthesize(query, uncovered)

        priority = self._priority(missing_topics) if has_gap else 0.0

        return GapReport(
            has_gap=has_gap,
            overall_score=round(overall, 3),
            coverage_score=round(coverage, 3),
            missing_topics=missing_topics,
            suggestion=suggestion,
            priority=priority,
            sub_intents=sub_intents,
        )

    # ── private ──────────────────────────────────────────────────────────────

    def _decompose(self, query: str) -> list[SubIntent]:
        prompt = f"""Break this query into atomic sub-intents.
    Return ONLY a JSON array of objects with a "text" key. No markdown, no preamble.
    Example: [{{"text": "how to cancel"}}]
    Query: {query}"""
        raw = self.client.complete(prompt, max_tokens=1024)
        raw = raw.strip()
        for fence in ["```json", "```"]:
            raw = raw.replace(fence, "")
        raw = raw.strip()
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        raw = match.group(0) if match else raw
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            items = [{"text": query}]
        return [SubIntent(text=item["text"]) for item in items if "text" in item]

    def _synthesize(self, query: str, uncovered: list[SubIntent]) -> tuple[str, list[str]]:
        gaps = "\n".join(
            f"- {si.text} (best score: {si.best_score:.2f})" for si in uncovered
        )
        prompt = f"""A user asked: "{query}"
These sub-topics had no good match in the knowledge base:
{gaps}

Return ONLY JSON with this shape. No markdown, no preamble:
{{"missing_topics": ["topic1", "topic2"], "suggestion": "one plain-English sentence about what docs to add"}}"""
        raw = self.client.complete(prompt, max_tokens=1024)
        raw = raw.strip()
        for fence in ["```json", "```"]:
            raw = raw.replace(fence, "")
        raw = raw.strip()
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        raw = match.group(0) if match else raw
        try:
            result = json.loads(raw)
            return result.get("suggestion", ""), result.get("missing_topics", [])
        except json.JSONDecodeError:
            return "Could not generate suggestion.", []

    def _priority(self, missing_topics: list[str]) -> float:
        if not self.history:
            return 0.5
        similar = self.history.count_similar_gaps(missing_topics, window_days=7)
        return round(min(similar / 50.0, 1.0), 3)


def _cosine(a: list[float], b: list[float]) -> float:
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0
