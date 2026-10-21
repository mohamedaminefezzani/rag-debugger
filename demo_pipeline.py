"""
Multi-turn RAG pipeline demo — LiteLLMClient (OpenAI) + local HF embedder.

Shows:
  - LiteLLMClient for LLM reasoning (decompose / rerank / synthesize)
  - Custom HF embedder for all cosine scoring (no OpenAI embedding quota used)
  - wrap_retriever() auto-logging
  - session() grouping
  - gap analysis stored back to SQLite
  - rd.dashboard() to inspect results

Install:
    pip install fastapi uvicorn litellm sentence-transformers numpy
    export OPENAI_API_KEY="your-key"   # or any provider LiteLLM supports

Run:
    python demo_pipeline.py
"""
import os, json
import rag_debugger as rd
from rag_debugger import GapDetector, LiteLLMClient
from rag_debugger.core import _get_store

# ── Custom HF embedder ────────────────────────────────────────────────────────

class GemmaEmbedder:
    """
    Local embedder using sentence-transformers.

    Swap MODEL for anything on HuggingFace
    """
    MODEL = "google/embeddinggemma-2"

    def __init__(self, model_name: str = MODEL, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedder: {model_name}  (device={device})")
        self._model = SentenceTransformer(model_name, device=device)
        print("Embedder ready.\n")

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_document(self, text: str) -> list[float]:
        # BGE and E5 are asymmetric — prefix docs with "passage: "
        # For symmetric models (Gemma-2) just return self.embed(text)
        vec = self._model.encode("passage: " + text, normalize_embeddings=True)
        return vec.tolist()


# ── Init ──────────────────────────────────────────────────────────────────────

rd.init(project="demo-support-bot", db_path=".rag_debug/demo.db")

# ── Knowledge base ────────────────────────────────────────────────────────────

KNOWLEDGE_BASE = [
    {"content": "To cancel your subscription go to Settings > Billing > Cancel Plan. Changes take effect at end of billing cycle.", "score": None, "source": "billing-faq.md"},
    {"content": "You can update your payment method under Settings > Billing > Payment Methods.", "score": None, "source": "billing-faq.md"},
    {"content": "Our Pro plan costs $29/month and includes unlimited API calls and priority support.", "score": None, "source": "pricing.md"},
    {"content": "To reset your password, click Forgot Password on the login screen and follow the email instructions.", "score": None, "source": "account-faq.md"},
    {"content": "Two-factor authentication can be enabled under Settings > Security > Enable 2FA.", "score": None, "source": "account-faq.md"},
]

# ── Retriever ─────────────────────────────────────────────────────────────────

class EmbeddingRetriever:
    def __init__(self, embedder, knowledge_base):
        self.embedder = embedder
        self.kb = knowledge_base
        print("Pre-embedding knowledge base…")
        self.doc_embeddings = [
            embedder.embed_document(doc["content"]) for doc in knowledge_base
        ]
        print(f"KB ready ({len(knowledge_base)} docs)\n")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        from rag_debugger.gap_detector import _cosine
        q_emb = self.embedder.embed(query)
        scored = [
            {**doc, "score": round(_cosine(q_emb, d_emb), 3)}
            for doc, d_emb in zip(self.kb, self.doc_embeddings)
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


# ── Clients ───────────────────────────────────────────────────────────────────

# LiteLLMClient: LLM-only (decompose / rerank / synthesize)
# embed_model=None because embeddings are handled by GemmaEmbedder locally
llm = LiteLLMClient(
    model="gemini/gemini-3.5-flash-lite",        # swap for any LiteLLM model string:
                                 #   "claude-3-5-haiku-20241022"  → Anthropic
                                 #   "gemini/gemini-3.5-flash-lite"    → Gemini via LiteLLM
                                 #   "ollama/llama3.2"            → local, no key, just add 'api_base' arg
    embed_model=None,
)

"""llm = LiteLLMClient(
    model="ollama/qwen3.5:4b",
    embed_model=None,
    api_base="http://localhost:11434",
)"""

embedder  = GemmaEmbedder()
retriever = rd.wrap_retriever(
    EmbeddingRetriever(embedder, KNOWLEDGE_BASE),
    label="support-kb"
)
detector  = GapDetector(llm, threshold=0.65, embedder=embedder)

# ── Turns ─────────────────────────────────────────────────────────────────────

TURNS = [
    "How do I cancel my subscription?",           # covered
    "What does the Pro plan cost?",               # covered
    "How do I set up 2FA on my iPhone?",          # partial
    "I cancelled but I want a refund for this month.",  # gap
    "How do I reset my password?",                # covered
    "How to apply for the job?",                  # gap
]

print("\n" + "="*60)
print("  rag-debugger demo — LiteLLMClient + HF embedder")
print("="*60 + "\n")

with rd.session(id="conv-demo-local2", user="test-user") as sess:
    for i, query in enumerate(TURNS, 1):
        chunks = retriever.retrieve(query)
        report = detector.analyze(query, chunks)

        print(f"Turn {i}: {query}")
        print(str(report))
        print()

        # write gap metadata back to the logged retrieval event
        store  = _get_store()
        latest = store.recent(limit=1)[0]
        store.conn.execute(
            "UPDATE retrieval_events SET metadata=? WHERE id=?",
            (json.dumps({
                **latest.metadata,
                "has_gap":        report.has_gap,
                "missing_topics": report.missing_topics,
                "suggestion":     report.suggestion,
                "coverage":       report.coverage_score,
            }), latest.id)
        )
        store.conn.commit()

summary = sess.summary()
print("="*60)
print(str(summary))
print("="*60)

print("\nLaunching dashboard… (Ctrl+C to stop)\n")
rd.dashboard()