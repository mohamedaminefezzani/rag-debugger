"""
Multi-turn RAG pipeline demo.

Tests:
  - wrap_retriever() auto-logging
  - session() grouping across turns
  - gap_report() on a weak query
  - rd.dashboard() to inspect results

Run:
    pip install fastapi uvicorn google-generativeai numpy
    export GEMINI_API_KEY="your-key"
    python demo_pipeline.py
"""
import os, json
import rag_debugger as rd
from rag_debugger import GeminiClient, GapDetector
from rag_debugger.core import _get_store

# ── Init ──────────────────────────────────────────────────────────────────────

rd.init(project="demo-support-bot", db_path=".rag_debug/demo.db")

# ── Fake knowledge base ───────────────────────────────────────────────────────
# Simulates a small support doc KB with deliberate gaps

KNOWLEDGE_BASE = [
    {"content": "To cancel your subscription go to Settings > Billing > Cancel Plan. Changes take effect at end of billing cycle.", "score": None, "source": "billing-faq.md"},
    {"content": "You can update your payment method under Settings > Billing > Payment Methods.", "score": None, "source": "billing-faq.md"},
    {"content": "Our Pro plan costs $29/month and includes unlimited API calls and priority support.", "score": None, "source": "pricing.md"},
    {"content": "To reset your password, click Forgot Password on the login screen and follow the email instructions.", "score": None, "source": "account-faq.md"},
    {"content": "Two-factor authentication can be enabled under Settings > Security > Enable 2FA.", "score": None, "source": "account-faq.md"},
]

# ── Fake retriever ────────────────────────────────────────────────────────────
# In real usage this would be a LangChain/LlamaIndex retriever hitting a vector DB

class EmbeddingRetriever:
    def __init__(self, client, knowledge_base):
        self.client = client
        self.kb = knowledge_base
        # pre-embed all docs once
        self.doc_embeddings = [
            client.embed_document(doc["content"]) for doc in knowledge_base
        ]

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        from rag_debugger.gap_detector import _cosine
        q_emb = self.client.embed(query)
        scored = []
        for doc, doc_emb in zip(self.kb, self.doc_embeddings):
            score = round(_cosine(q_emb, doc_emb), 3)
            scored.append({**doc, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

# ── Gemini client + retriever + gap detector ──────────────────────────────────────────────

gemini = GeminiClient()
retriever = rd.wrap_retriever(EmbeddingRetriever(gemini, KNOWLEDGE_BASE), label="support-kb")
detector = GapDetector(gemini, threshold=0.65)

# ── Multi-turn conversation ───────────────────────────────────────────────────

TURNS = [
    # Turn 1: well-covered query
    "How do I cancel my subscription?",
    # Turn 2: also covered
    "What does the Pro plan cost?",
    # Turn 3: partial coverage — 2FA exists but iOS-specific doesn't
    "How do I set up 2FA on my iPhone?",
    # Turn 4: clear gap — no refund docs
    "I cancelled but I want a refund for this month.",
    # Turn 5: covered
    "How do I reset my password?",
]

print("\n" + "="*60)
print("  rag-debugger demo — multi-turn support bot")
print("="*60 + "\n")

with rd.session(id="conv-demo-005", user="test-user") as sess:
    for i, query in enumerate(TURNS, 1):
        chunks = retriever.retrieve(query)
        report = detector.analyze(query, chunks)

        # store gap analysis in the logged event metadata
        store = _get_store()
        latest = store.recent(limit=1)[0]
        # update metadata with gap info
        store.conn.execute(
            "UPDATE retrieval_events SET metadata=? WHERE id=?",
            (json.dumps({
                **latest.metadata,
                "has_gap": report.has_gap,
                "missing_topics": report.missing_topics,
                "suggestion": report.suggestion,
                "coverage": report.coverage_score,
            }), latest.id)
    )
    store.conn.commit()

# ── Session summary ───────────────────────────────────────────────────────────

summary = sess.summary()
print("="*60)
print(str(summary))
print("="*60)

# ── Launch dashboard ──────────────────────────────────────────────────────────

print("\nLaunching dashboard… (Ctrl+C to stop)\n")
rd.dashboard()
