"""Chunking stage: split loaded documents into retrieval-sized text nodes.

Receives documents from :mod:`rag_chatbot.ingestion` and hands nodes to
:mod:`rag_chatbot.vector_store`. Chunk geometry comes from the caller, so this
module holds no configuration of its own.
"""

from __future__ import annotations

from typing import Sequence

from llama_index.core import Document
from llama_index.core.schema import TextNode


def split_documents(
    documents: Sequence[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextNode]:
    """Break documents into overlapping :class:`~llama_index.core.schema.TextNode` chunks.

    Uses a sentence-aware splitter so boundaries fall on sentence ends rather than
    mid-clause, which materially improves retrieved-chunk readability.

    Parameters
    ----------
    documents : Sequence[Document]
        Output of :func:`rag_chatbot.ingestion.load_documents`.
    chunk_size : int
        Maximum tokens per node.
    chunk_overlap : int
        Tokens repeated between consecutive nodes; must be smaller than
        ``chunk_size``.

    Returns
    -------
    list[TextNode]
        Flattened node list preserving each document's source metadata.

    Raises
    ------
    ValueError
        If ``chunk_overlap`` is not smaller than ``chunk_size``.
    NotImplementedError
        Stage not implemented yet.
    """
    raise NotImplementedError("Document chunking is not implemented yet.")
