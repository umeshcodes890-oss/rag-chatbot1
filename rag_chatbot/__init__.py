"""RAG pipeline package for the LLM-powered chatbot.

Each submodule owns exactly one stage of the pipeline and can be unit-tested in
isolation. Stage order is: ingestion -> chunking -> embeddings -> vector_store
-> retrieval -> chatbot. The :mod:`rag_chatbot.pipeline` module is the only one
that wires the stages together.
"""

from __future__ import annotations

__all__ = [
    "chatbot",
    "chunking",
    "embeddings",
    "ingestion",
    "llm",
    "pipeline",
    "retrieval",
    "vector_store",
]
