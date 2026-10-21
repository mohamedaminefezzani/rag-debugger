from .gap_detector import (
    GapDetector,
    GapReport,
    GeminiClient,
    LiteLLMClient,
    LLMClientProtocol,
    EmbedderProtocol,
    SubIntent,
    ChunkResult,
)
from .store import SQLiteStore, RetrievalEvent
from .core import init, wrap_retriever, log_retrieval, session, dashboard, trace

__version__ = "0.4.0"