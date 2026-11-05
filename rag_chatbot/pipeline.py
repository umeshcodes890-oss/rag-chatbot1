"""Pipeline orchestration: wire the stages together in dependency order.

The only module that knows the whole flow. UI code and tests call these functions
rather than sequencing ingestion, chunking, embeddings, and storage themselves —
which is what keeps :mod:`app` free of pipeline logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llama_index.core import VectorStoreIndex
    from config import Settings
    from rag_chatbot.chatbot import Chatbot


def ingest_documents(settings: "Settings") -> "VectorStoreIndex":
    """Run the full indexing flow over ``settings.data_dir``.

    Intended sequence:
    ``ingestion.discover_files`` -> ``ingestion.load_documents`` ->
    ``chunking.split_documents`` -> ``embeddings.build_embed_model`` ->
    ``vector_store.build_vector_store`` -> ``vector_store.build_index``.

    Parameters
    ----------
    settings : Settings
        Supplies the data directory, chunk geometry, and storage paths.

    Returns
    -------
    VectorStoreIndex
        Index over the freshly stored chunks.

    Raises
    ------
    NotImplementedError
        Orchestration is not implemented yet.
    """
    raise NotImplementedError("Document ingestion orchestration is not implemented yet.")


def build_chatbot(settings: "Settings") -> "Chatbot":
    """Attach a conversation-ready assistant to the existing index.

    Intended sequence: verify ``vector_store.index_exists``, reopen the stored
    collection, and wrap it with :class:`rag_chatbot.chatbot.Chatbot`.

    Parameters
    ----------
    settings : Settings
        Supplies storage paths and model selection.

    Returns
    -------
    Chatbot
        Assistant bound to the persisted index.

    Raises
    ------
    FileNotFoundError
        If no index has been ingested yet.
    NotImplementedError
        Orchestration is not implemented yet.
    """
    raise NotImplementedError("Chatbot assembly is not implemented yet.")
