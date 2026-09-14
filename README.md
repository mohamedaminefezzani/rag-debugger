# rag-debugger

**Intercept, inspect, and fix your RAG retrieval pipeline.**

Most RAG bugs aren't in your code — they're in your retrieval. Wrong chunks get
selected, knowledge gaps go undetected, and you find out when users complain.
`rag-debugger` gives you visibility into exactly what your vector DB returned,
why it won, and what's missing from your knowledge base.

## Install

```bash
pip install rag-debugger-amine
```

For the local dashboard:

```bash
pip install rag-debugger-amine[dashboard]
```

## Quickstart

```python
import rag_debugger as rd

rd.init(project="my-rag-app")
retriever = rd.wrap_retriever(your_retriever)
```

## Gap detection

Find what your knowledge base is missing before your users do:

```python
from rag_debugger import GeminiClient, GapDetector

client = GeminiClient()  # set GEMINI_API_KEY env var
detector = GapDetector(client)  # default threshold is 0.65

chunks = your_retriever.get_relevant_documents(query)
report = detector.analyze(query, [{"content": c.page_content} for c in chunks])

print(report)
# [GAP DETECTED] coverage=50%  worst_score=0.64  priority=0.50
#   Missing: refund policy, iOS-specific cancellation
#   Fix: Add docs covering refund eligibility and iOS cancellation flow.
#     ✓ [0.71] how to cancel
#     ✗ [0.64] how to get a refund
```

## Why sub-intent decomposition

A query like *"cancel my iOS subscription and get a refund"* is really four
questions. Standard RAG scores the whole query — if cancellation chunks score
high, the query looks covered. `rag-debugger` decomposes it into atomic
sub-intents and scores each one independently, so a missing refund policy
is always caught even when the cancellation docs are excellent.

Borderline scores (0.60–0.75) are passed through a reranker — a lightweight
LLM call that asks "does this chunk actually answer this question?" — so
semantically similar but irrelevant chunks don't pass as covered.

## Session grouping

Group multi-turn conversations under a single session to get a summary of
retrieval quality across the whole interaction:

```python
with rd.session(id="conv-123", user="user-42") as s:
    retriever.get_relevant_documents("first query")
    retriever.get_relevant_documents("follow-up query")

summary = s.summary()
print(summary)
# Session conv-123
#   duration:    430ms
#   events:      2
#   avg score:   0.741
#   worst score: 0.677
#   gaps:        0 / 2
```

Sessions are thread-safe — concurrent requests in a web app won't bleed into
each other.

## Local dashboard

Visualize retrieval events, chunk scores, and gap flags in a local web UI:

```python
rd.dashboard()  # opens http://localhost:7842
```

The dashboard shows:
- Per-session summary — avg score, worst score, gap count
- Per-event chunk score bars with content preview
- Gap flags with missing topics and fix suggestions
- Auto-refreshes every 10 seconds

## Integrations

Works with LangChain, LlamaIndex, and any custom pipeline:

```python
# LangChain
retriever = rd.wrap_retriever(vectorstore.as_retriever(), label="docs")

# LlamaIndex
retriever = rd.wrap_retriever(index.as_retriever())

# Custom object
retriever = rd.wrap_retriever(my_retriever, method="fetch_docs")
```

## LLM & embedding backend

`rag-debugger` uses Google Gemini by default (free tier via
[Google AI Studio](https://aistudio.google.com)):

```python
from rag_debugger import GeminiClient
client = GeminiClient(api_key="...")  # or set GEMINI_API_KEY env var
```

Models used:
- LLM: `gemini-3.5-flash-lite`
- Embeddings: `gemini-embedding-001`

## License

MIT