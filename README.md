# rag-debugger-amine

**Find and fix gaps in your RAG knowledge base.**

Standard RAG evals tell you if retrieval worked. They don't tell you *why* it
failed or *what to add* to fix it. `rag-debugger` intercepts your retrieval
calls, decomposes each query into atomic sub-intents, scores each one
independently against your chunks, and flags exactly which topics have no
coverage, then suggests in plain English what docs to add.

```
[GAP DETECTED] coverage=50%  worst_score=0.64  priority=0.50
  Missing: refund policy, iOS-specific cancellation
  Fix: Add docs covering refund eligibility and iOS cancellation flow.
    ✓ [0.71] how to cancel
    ✗ [0.64] how to get a refund
```

---

## Install

Pick the setup that matches your needs:

### I want the simplest free setup
Uses Gemini 3.5 Flash Lite (LLM) + gemini-embedding-001 (embeddings). Both free tier via Google AI Studio.

```bash
pip install rag-debugger-amine
export GEMINI_API_KEY="your-key"
```

### I want to use OpenAI / Anthropic / Mistral / any cloud provider
Uses LiteLLM to proxy any provider behind one interface.

```bash
pip install rag-debugger-amine[litellm]
export OPENAI_API_KEY="your-key"   # or ANTHROPIC_API_KEY, MISTRAL_API_KEY, etc.
```

### I want local embeddings (no embedding quota, private, fast)
Uses a cloud LLM for reasoning + Google's Embedding Gemma 2 locally for cosine scoring.

```bash
pip install rag-debugger-amine[litellm]
pip install sentence-transformers
# no extra API key needed for embeddings
```

### I want everything fully local (no internet, no API keys)
Uses Ollama for both LLM and embeddings.

```bash
pip install rag-debugger-amine[litellm]
# install Ollama from https://ollama.com, then:
ollama pull llama3.2
ollama pull nomic-embed-text
```

### I also want the dashboard
Add `[dashboard]` to any of the above:

```bash
pip install rag-debugger-amine[dashboard]
pip install rag-debugger-amine[litellm,dashboard]
pip install rag-debugger-amine[all]   # everything at once
```

---

## Quickstart

```python
import rag_debugger as rd
from rag_debugger import GeminiClient, GapDetector

rd.init(project="my-bot", db_path=".rag_debug/events.db")

client   = GeminiClient()   # free tier - set GEMINI_API_KEY
detector = GapDetector(client)

chunks = your_retriever.retrieve(query)
report = detector.analyze(query, chunks)
print(report)
```

Works with LangChain, LlamaIndex, or any custom retriever.

---

<<<<<<< HEAD
## LLM Client
=======
Borderline scores (0.60–0.75) are passed through a reranker, a lightweight
LLM call that asks "does this chunk actually answer this question?" — so
semantically similar but irrelevant chunks don't pass as covered.
>>>>>>> dac26a00aa9768a23229a3ea751c6c03c76f3d06

`GapDetector` accepts any LLM client. Three built-in options:

<<<<<<< HEAD
### GeminiClient - free tier, zero config
=======
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

Sessions are thread-safe, concurrent requests in a web app won't bleed into
each other.

## Trace decorator

Instrument any retrieval function with a single line — no need to wrap a retriever object:

```python
import rag_debugger as rd

rd.init(project="my-rag-app")

@rd.trace(label="my-retriever")
def retrieve(query: str) -> list[dict]:
    return vector_db.search(query)

# Also works bare, async, and with custom key names
@rd.trace
async def retrieve(query: str) -> list[dict]:
    return await vector_db.asearch(query)

@rd.trace(score_key="relevance", content_key="text")
def retrieve(query: str) -> list[dict]:
    return pinecone.query(query)
```

## Local dashboard

Visualize retrieval events, chunk scores, and gap flags in a local web UI:

```python
rd.dashboard()  # opens http://localhost:7842
```

The dashboard shows:
- Per-session summary, avg score, worst score, gap count
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
>>>>>>> dac26a00aa9768a23229a3ea751c6c03c76f3d06

```python
from rag_debugger import GeminiClient

client = GeminiClient(
    api_key="your-key",                        # or GEMINI_API_KEY env var
    model="gemini-3.5-flash-lite",             # default
    embed_model="models/gemini-embedding-001", # default
)

detector = GapDetector(client)
```

### LiteLLMClient - 100+ providers, one class

```bash
pip install rag-debugger-amine[litellm]
```

<<<<<<< HEAD
```python
from rag_debugger import LiteLLMClient

client = LiteLLMClient(
    model="...",        # LiteLLM model string (required)
    embed_model="...",  # embedding model - pass None to disable
    api_key="...",      # optional, falls back to env var
    api_base="...",     # optional, for local models (Ollama, vLLM)
)
```

| Provider | `model=` | `embed_model=` | Env var |
|---|---|---|---|
| OpenAI | `"gpt-4o-mini"` | `"text-embedding-3-small"` | `OPENAI_API_KEY` |
| Anthropic | `"claude-3-5-haiku-20241022"` | *(none)* | `ANTHROPIC_API_KEY` |
| Gemini | `"gemini/gemini-3.5-flash-lite"` | `"gemini/text-embedding-004"` | `GEMINI_API_KEY` |
| Mistral | `"mistral/mistral-small"` | `"mistral/mistral-embed"` | `MISTRAL_API_KEY` |
| Cohere | `"command-r"` | `"cohere/embed-english-v3.0"` | `COHERE_API_KEY` |
| Ollama (local) | `"ollama/llama3.2"` | `"ollama/nomic-embed-text"` | *(none)* |

```python
# OpenAI for everything
client = LiteLLMClient("gpt-4o-mini", embed_model="text-embedding-3-small")
detector = GapDetector(client)

# Anthropic LLM + OpenAI embeddings
llm = LiteLLMClient("claude-3-5-haiku-20241022", embed_model=None)
emb = LiteLLMClient("gpt-4o-mini", embed_model="text-embedding-3-small")
detector = GapDetector(llm, embedder=emb)

# Fully local - no API key, no cost
client = LiteLLMClient(
    model="ollama/llama3.2",
    embed_model="ollama/nomic-embed-text",
    api_base="http://localhost:11434",
)
detector = GapDetector(client)
```

### Custom client - bring any object

Implement `.complete()` and pass it directly - no base class needed:

```python
class MyClient:
    def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        return my_api.chat(prompt).text

detector = GapDetector(MyClient())
```

Add `.embed()` and optionally `.embed_document()` if you want the same object
to handle embeddings too.

---

## Local embeddings with Embedding Gemma 2

Google's [Embedding Gemma 2](https://huggingface.co/google/embedding-gemma-2) is the
recommended local embedder - strong quality, runs on CPU, no API quota.

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer
from rag_debugger import LiteLLMClient, GapDetector

class GemmaEmbedder:
    """Local embedder using Google's Embedding Gemma 2."""

    MODEL = "google/embedding-gemma-2"

    def __init__(self, model_name: str = MODEL, device: str = "cpu"):
        print(f"Loading embedder: {model_name}")
        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, text: str) -> list[float]:
        """Embed a query."""
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_document(self, text: str) -> list[float]:
        """Embed a KB document - same as embed() since Gemma 2 is symmetric."""
        return self.embed(text)


# Cloud LLM for reasoning, Gemma 2 locally for cosine scoring
llm      = LiteLLMClient("gpt-4o-mini", embed_model=None)
embedder = GemmaEmbedder()

detector = GapDetector(llm, threshold=0.62, embedder=embedder)
```

Threshold guide by embedding model:

| Model | Recommended threshold |
|---|---|
| `gemini-embedding-001` | 0.65 |
| `google/embedding-gemma-2` | 0.62 |
| `text-embedding-3-small` | 0.70 |
| `nomic-embed-text` | 0.60 |

---

## Auto-logging with wrap_retriever

Log every retrieval call to SQLite with one line:

```python
# LangChain
retriever = rd.wrap_retriever(lc_retriever, label="langchain")

# LlamaIndex
retriever = rd.wrap_retriever(li_retriever, label="llamaindex")

# Custom object - auto-detects retrieve() / query() / get_relevant_documents()
retriever = rd.wrap_retriever(my_retriever, label="custom")

# Explicit method name
retriever = rd.wrap_retriever(my_retriever, method="fetch_docs", label="custom")
```

---

## Session grouping

Group multi-turn conversations for dashboard analysis:

```python
with rd.session(id="conv-001", user="alice") as sess:
    for query in conversation:
        chunks = retriever.retrieve(query)
        report = detector.analyze(query, chunks)

summary = sess.summary()
print(summary.gap_count)          # gaps detected this session
print(summary.avg_top_score)      # average top chunk score
print(summary.lowest_top_score)   # worst retrieval in the session
print(summary.duration_ms)        # total session time
```

---

## @rd.trace decorator

Instrument any retrieval function without changing its return value:

```python
import rag_debugger as rd

rd.init(project="my-bot")

@rd.trace
def retrieve(query: str) -> list[dict]:
    ...

@rd.trace(label="reranker")
def rerank(query: str, docs: list) -> list[dict]:
    ...

@rd.trace(label="async-retriever")
async def async_retrieve(query: str) -> list[dict]:
    ...
```

---

## Dashboard

```python
rd.dashboard()   # opens http://localhost:8000 in your browser
```

Shows per-session gap count, chunk scores, missing topics, fix suggestions, and session timeline.

---

## GapReport fields

```python
report = detector.analyze(query, chunks)

report.has_gap          # bool
report.coverage_score   # fraction of sub-intents covered (0.0–1.0)
report.overall_score    # worst sub-intent score (0.0–1.0)
report.missing_topics   # list[str] - topics with no coverage
report.suggestion       # plain-English fix suggestion
report.priority         # 0.0–1.0, based on gap frequency in history
report.sub_intents      # list[SubIntent] - per sub-intent breakdown

for si in report.sub_intents:
    print(si.text, si.best_score, si.covered)
```

---

## GapDetector options

```python
detector = GapDetector(
    client,              # any LLM client with .complete()
    threshold=0.65,      # cosine similarity cutoff (default 0.65)
    history_store=None,  # SQLiteStore for priority scoring
    embedder=None,       # custom embedder, falls back to client
)
```

---

## Decision guide

```
Free tier / simplest setup?
  → GeminiClient()

Already paying for OpenAI / Anthropic / Mistral?
  → LiteLLMClient(model="...", embed_model="...")

Want local embeddings (no quota, private)?
  → LiteLLMClient(model="...", embed_model=None) + embedder=GemmaEmbedder()

Fully air-gapped / no internet?
  → LiteLLMClient("ollama/llama3.2", api_base="http://localhost:11434")

Custom API / wrapper?
  → implement .complete() and pass it directly
```

---

## Changelog

**v0.4.0**
- `LiteLLMClient` - 100+ providers via LiteLLM, optional `[litellm]` extra
- `LLMClientProtocol` / `EmbedderProtocol` - formal protocols for custom clients
- `embedder=` parameter on `GapDetector` - decouple LLM and embedding providers
- Embedding Gemma 2 recommended as default local embedder
- Safety filter crash fix in reranker (`finish_reason=2` no longer raises)

**v0.3.0**
- `@rd.trace` decorator - instrument any retrieval function, sync and async
- `rd.dashboard()` - FastAPI dashboard with gap flags and session view

**v0.2.0**
- `rd.wrap_retriever()` - auto-logging for LangChain, LlamaIndex, custom
- `rd.session()` - multi-turn conversation grouping
- SQLite event store

**v0.1.0**
- `GapDetector` - sub-intent decomposition, cosine scoring, reranker, synthesizer
- `GeminiClient` - free tier via Google AI Studio
=======
MIT
>>>>>>> dac26a00aa9768a23229a3ea751c6c03c76f3d06
