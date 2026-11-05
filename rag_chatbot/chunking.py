"""Chunking stage: split loaded documents into retrieval-sized text nodes.

Receives documents from :mod:`rag_chatbot.ingestion` and hands nodes to
:mod:`rag_chatbot.vector_store`. Chunk geometry comes from the caller, so this
module holds no configuration of its own.

Splitting uses LlamaIndex's ``SentenceSplitter``, which packs whole sentences up
to the token budget instead of cutting at a fixed character offset. The default
secondary split (``.``) is left in place, so an oversized sentence is broken
between words rather than mid-word.
"""

from __future__ import annotations

from typing import Sequence

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode


def _validate_geometry(chunk_size: int, chunk_overlap: int) -> None:
    """Reject chunk settings that cannot produce a usable split.

    Parameters
    ----------
    chunk_size : int
        Maximum tokens per node.
    chunk_overlap : int
        Tokens repeated between adjacent nodes.

    Raises
    ------
    ValueError
        If either value is out of range, or the overlap leaves no new text per
        chunk.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be at least 1, got {chunk_size}.")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be 0 or greater, got {chunk_overlap}.")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than "
            f"chunk_size ({chunk_size}); otherwise no new text enters a chunk."
        )


def build_splitter(*, chunk_size: int, chunk_overlap: int) -> SentenceSplitter:
    """Construct the sentence-aware splitter used by this stage.

    ``include_metadata`` is enabled because in this LlamaIndex version it governs
    whether a source document's metadata is copied onto its child nodes. Stored
    chunk text is unaffected either way: the metadata is rendered later, from
    ``metadata_template``, when a prompt is assembled. Leaving it off silently
    loses every provenance field, so a retrieved chunk could no longer cite its
    file and page.

    Parameters
    ----------
    chunk_size : int
        Maximum tokens per node.
    chunk_overlap : int
        Tokens repeated between adjacent nodes.

    Returns
    -------
    SentenceSplitter
        Configured splitter, reusable for measuring token counts.

    Raises
    ------
    ValueError
        If the geometry is invalid. See :func:`_validate_geometry`.
    """
    _validate_geometry(chunk_size, chunk_overlap)
    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        include_metadata=True,
    )


def split_documents(
    documents: Sequence[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextNode]:
    """Break documents into overlapping :class:`~llama_index.core.schema.TextNode` chunks.

    Every source document's metadata is copied onto each of its nodes, so a chunk
    retrieved later can still cite the file and page it came from. Documents with
    no text after cleaning are skipped, and non-text nodes are dropped because
    this pipeline ingests text-only formats.

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
        Flattened node list in document order, preserving source metadata.

    Raises
    ------
    ValueError
        If ``chunk_overlap`` is not smaller than ``chunk_size``, or either value
        is out of range.
    """
    splitter = build_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    populated = [doc for doc in documents if doc.text and doc.text.strip()]
    if not populated:
        return []

    nodes = splitter.get_nodes_from_documents(populated)
    return [
        node
        for node in nodes
        if isinstance(node, TextNode) and node.get_content().strip()
    ]
