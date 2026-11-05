"""Storage stage: persist chunks in ChromaDB and expose a queryable index.

Owns the on-disk lifecycle of the vector store. Retrieval and chat modules
consume the index built here and never talk to ChromaDB directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores.types import BasePydanticVectorStore

if TYPE_CHECKING:
    from config import Settings
    from llama_index.core.schema import TextNode
    from rag_chatbot.embeddings import BaseEmbedding


def index_exists(settings: Settings) -> bool:
    """Report whether a previously built Chroma collection is on disk.

    Returns
    -------
    bool
        ``True`` when the persistence directory holds a usable collection, which
        the UI uses to offer "chat" before "ingest".

    Raises
    ------
    NotImplementedError
        Stage not implemented yet.
    """
    raise NotImplementedError("Index existence check is not implemented yet.")


def build_vector_store(settings: Settings) -> BasePydanticVectorStore:
    """Open the persistent Chroma collection as a LlamaIndex vector store.

    Intended implementation: create a ``chromadb.PersistentClient`` rooted at
    ``settings.chroma_dir``, get or create the ``settings.chroma_collection``
    collection, and wrap it in
    ``llama_index.vector_stores.chroma.ChromaVectorStore``.

    Parameters
    ----------
    settings : Settings
        Provides ``chroma_dir`` and ``chroma_collection``.

    Returns
    -------
    BasePydanticVectorStore
        A Chroma-backed store ready for indexing.

    Raises
    ------
    NotImplementedError
        Stage not implemented yet.
    """
    raise NotImplementedError("Chroma vector store setup is not implemented yet.")


def build_index(
    nodes: list["TextNode"],
    *,
    vector_store: BasePydanticVectorStore,
    embed_model: "BaseEmbedding",
) -> VectorStoreIndex:
    """Insert chunks into ``vector_store`` and return the built index.

    Parameters
    ----------
    nodes : list[TextNode]
        Output of :func:`rag_chatbot.chunking.split_documents`.
    vector_store : BasePydanticVectorStore
        Target collection from :func:`build_vector_store`.
    embed_model : BaseEmbedding
        Model from :func:`rag_chatbot.embeddings.build_embed_model` used to
        vectorise each node.

    Returns
    -------
    VectorStoreIndex
        Index over the newly stored chunks.

    Raises
    ------
    NotImplementedError
        Stage not implemented yet.
    """
    raise NotImplementedError("Index building is not implemented yet.")


def reset_index(settings: Settings) -> None:
    """Delete the persisted Chroma directory so the next ingest starts clean.

    Parameters
    ----------
    settings : Settings
        Provides ``chroma_dir``.

    Raises
    ------
    NotImplementedError
        Stage not implemented yet.
    """
    raise NotImplementedError("Index reset is not implemented yet.")
