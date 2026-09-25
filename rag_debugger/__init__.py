from .gap_detector import GapDetector, GapReport, GeminiClient, SubIntent, ChunkResult
from .core import init, wrap_retriever, log_retrieval, session, dashboard, trace
from .store import SQLiteStore, RetrievalEvent

__version__ = "0.2.0"
__all__ = [
    # core API
    "init",
    "wrap_retriever",
    "log_retrieval",
    "session",
    "dashboard",
    "trace",
    # gap detection
    "GapDetector",
    "GapReport",
    "GeminiClient",
    "SubIntent",
    "ChunkResult",
    # storage
    "SQLiteStore",
    "RetrievalEvent",
]
