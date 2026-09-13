# rag-debugger

**Intercept, inspect, and fix your RAG retrieval pipeline.**

Most RAG bugs aren't in your code — they're in your retrieval. Wrong chunks get
selected, knowledge gaps go undetected, and you find out when users complain.
`rag-debugger` gives you visibility into exactly what your vector DB returned,
why it won, and what's missing from your knowledge base.

## Install

```bash
pip install rag-debugger
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
detector = GapDetector(client) # default threshold is 0.65, change here if needed

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

## LLM & embedding backend

`rag-debugger` uses Google Gemini by default (free tier via
[Google AI Studio](https://aistudio.google.com)):

```python
from rag_debugger import GeminiClient
client = GeminiClient(api_key="...")   # or set GEMINI_API_KEY as an environment variable
```

Used models:
- LLM: `gemini-2.5-flash`
- Embeddings: `gemini-embedding-001`

## Integrations

Works with LangChain, LlamaIndex, and any custom pipeline. See the
[SDK reference](https://github.com/mohamedaminefezzani/rag-debugger) for details.

## License

MIT
