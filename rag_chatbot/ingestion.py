"""Ingestion stage: discover source documents and load them into LlamaIndex.

Depends only on :class:`config.Settings` values and LlamaIndex document types.
Knows nothing about chunking, embeddings, or storage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Sequence

from llama_index.core import Document

SUPPORTED_EXTENSIONS: Final[tuple[str, ...]] = (".pdf", ".docx", ".txt", ".md")


def discover_files(
    data_dir: Path,
    extensions: Sequence[str] = SUPPORTED_EXTENSIONS,
) -> list[Path]:
    """List ingestable files beneath ``data_dir``.

    Parameters
    ----------
    data_dir : Path
        Directory to scan recursively.
    extensions : Sequence[str]
        Lowercase suffixes to keep, including the leading dot.

    Returns
    -------
    list[Path]
        Matching files in deterministic sorted order. Hidden files and the
        ``.gitkeep`` placeholders are skipped.

    Raises
    ------
    FileNotFoundError
        If ``data_dir`` does not exist.
    NotImplementedError
        Stage not implemented yet.
    """
    raise NotImplementedError("Ingestion file discovery is not implemented yet.")


def file_to_documents(path: Path) -> list[Document]:
    """Load one file into one or more :class:`~llama_index.core.Document` objects.

    Selects the reader by suffix: PDF via ``PDFReader``, DOCX via ``DocxReader``,
    text and Markdown via ``TextReader``. Each document carries its source path in
    ``metadata`` so later stages can attribute answers to a file.

    Parameters
    ----------
    path : Path
        Absolute or relative path to a supported document.

    Returns
    -------
    list[Document]
        Loaded documents with source metadata attached.

    Raises
    ------
    ValueError
        If the suffix is not in :data:`SUPPORTED_EXTENSIONS`.
    NotImplementedError
        Stage not implemented yet.
    """
    raise NotImplementedError("Document loading is not implemented yet.")


def load_documents(paths: Sequence[Path]) -> list[Document]:
    """Load and concatenate every document in ``paths``.

    Parameters
    ----------
    paths : Sequence[Path]
        Files returned by :func:`discover_files`.

    Returns
    -------
    list[Document]
        One flat list ready for the chunking stage.

    Raises
    ------
    NotImplementedError
        Stage not implemented yet.
    """
    raise NotImplementedError("Batch document loading is not implemented yet.")
