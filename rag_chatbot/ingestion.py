"""Ingestion stage: discover source documents and load them into LlamaIndex.

Depends only on :class:`config.Settings` values and LlamaIndex document types.
Knows nothing about chunking, embeddings, or storage.

Reader mapping: ``.pdf`` uses ``PDFReader`` (one document per page), ``.docx``
uses ``DocxReader``, and ``.txt``/``.md`` are read directly as UTF-8.

Plain text and Markdown bypass a reader on purpose: the installed LlamaIndex
build ships no ``TextReader``, and ``MarkdownReader`` needs the unstructured
package. Reading those two formats directly avoids a dependency that adds nothing
to a byte-stream already in the encoding we want.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Sequence

from llama_index.core import Document
from llama_index.readers.file import DocxReader, PDFReader

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: Final[tuple[str, ...]] = (".pdf", ".docx", ".txt", ".md")

PDF_EXTENSION: Final[str] = ".pdf"
DOCX_EXTENSION: Final[str] = ".docx"
PLAIN_TEXT_EXTENSIONS: Final[tuple[str, ...]] = (".txt", ".md")

_FALLBACK_ENCODING: Final[str] = "latin-1"
_MAX_BLANK_LINES: Final[int] = 2

_HORIZONTAL_WS = re.compile(r"[^\S\n]+")
_TRAILING_WS = re.compile(r"[^\S\n]+$", re.MULTILINE)
_BLANK_RUNS = re.compile(r"\n{3,}")


class UnsupportedFileError(ValueError):
    """The file's extension is not one LlamaIndex is configured to read."""


class CorruptDocumentError(RuntimeError):
    """A parser failed to read a file that exists and has a supported type."""


def clean_text(text: str | None) -> str:
    """Normalise whitespace without altering wording.

    Applied in this order, so paragraph structure survives:

    1. ``\\r\\n`` and ``\\r`` become ``\\n``.
    2. Non-breaking spaces become ordinary spaces; zero-width characters vanish.
    3. Runs of tabs and spaces inside a line collapse to one space.
    4. Trailing whitespace is stripped from every line.
    5. Three or more consecutive newlines collapse to a single blank line, which
       keeps paragraphs distinct but removes PDF-extraction noise.
    6. Leading and trailing whitespace is trimmed from the result.

    Sentence punctuation, casing, hyphenation, and line breaks inside a paragraph
    are left alone, so no meaningful content is damaged.

    Parameters
    ----------
    text : str | None
        Raw extracted text. ``None`` and empty strings are tolerated.

    Returns
    -------
    str
        Cleaned text, or ``""`` when there was nothing to keep.
    """
    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ").replace("\u200b", "")
    normalized = _HORIZONTAL_WS.sub(" ", normalized)
    normalized = _TRAILING_WS.sub("", normalized)
    normalized = _BLANK_RUNS.sub("\n" * _MAX_BLANK_LINES, normalized)
    return normalized.strip()


def _source_metadata(path: Path) -> dict[str, str | int]:
    """Build provenance metadata shared by every reader.

    Parameters
    ----------
    path : Path
        The file that was read.

    Returns
    -------
    dict[str, str | int]
        Filename, absolute path, format, byte size, and modification time.
    """
    stat = path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    return {
        "file_name": path.name,
        "file_path": str(path.resolve()),
        "file_type": path.suffix.lower().lstrip(".") or "unknown",
        "file_size_bytes": stat.st_size,
        "last_modified": modified.isoformat(timespec="seconds"),
    }


def _read_pdf(path: Path, base_metadata: dict[str, str | int]) -> list[Document]:
    """Load a PDF as one document per page, numbering each page.

    Parameters
    ----------
    path : Path
        PDF file to parse.
    base_metadata : dict[str, str | int]
        Shared provenance fields from :func:`_source_metadata`.

    Returns
    -------
    list[Document]
        One document per non-empty page, carrying ``page_label``,
        ``page_number``, and ``page_count``.

    Raises
    ------
    CorruptDocumentError
        If ``pypdf`` cannot parse the file.
    """
    try:
        raw = PDFReader().load_data(path, extra_info=dict(base_metadata))
    except Exception as exc:
        raise CorruptDocumentError(f"Could not parse PDF {path.name}: {exc}") from exc

    total = len(raw)
    pages: list[Document] = []
    for number, doc in enumerate(raw, start=1):
        metadata = dict(doc.metadata)
        metadata["page_number"] = number
        metadata["page_count"] = total
        pages.append((clean_text(doc.text), metadata))

    return [
        Document(text=text, metadata=metadata)
        for text, metadata in pages
        if text
    ]


def _read_docx(path: Path, base_metadata: dict[str, str | int]) -> list[Document]:
    """Load a Word document as a single document.

    ``docx2txt`` flattens the body and discards pagination, so no page metadata
    is available for this format.

    Parameters
    ----------
    path : Path
        DOCX file to parse.
    base_metadata : dict[str, str | int]
        Shared provenance fields from :func:`_source_metadata`.

    Returns
    -------
    list[Document]
        Zero or one document, depending on whether the body held text.

    Raises
    ------
    CorruptDocumentError
        If the archive is unreadable or malformed.
    """
    try:
        raw = DocxReader().load_data(path, extra_info=dict(base_metadata))
    except Exception as exc:
        raise CorruptDocumentError(f"Could not parse DOCX {path.name}: {exc}") from exc

    return [
        Document(text=clean_text(doc.text), metadata=dict(doc.metadata))
        for doc in raw
        if clean_text(doc.text)
    ]


def _read_plain_text(
    path: Path, base_metadata: dict[str, str | int]
) -> list[Document]:
    """Load a ``.txt`` or ``.md`` file as a single document.

    Decodes as UTF-8 and falls back to latin-1, which accepts any byte sequence,
    so a stray encoding marker cannot abort an otherwise healthy batch.

    Parameters
    ----------
    path : Path
        Text file to read.
    base_metadata : dict[str, str | int]
        Shared provenance fields from :func:`_source_metadata`.

    Returns
    -------
    list[Document]
        Zero or one document, depending on whether the file held content.

    Raises
    ------
    CorruptDocumentError
        If the file cannot be opened or read at all.
    """
    try:
        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw = path.read_text(encoding=_FALLBACK_ENCODING)
    except OSError as exc:
        raise CorruptDocumentError(f"Could not read {path.name}: {exc}") from exc

    text = clean_text(raw)
    if not text:
        return []
    return [Document(text=text, metadata=dict(base_metadata))]


def file_to_documents(path: Path) -> list[Document]:
    """Load one file into one or more :class:`~llama_index.core.Document` objects.

    Routes by suffix to the matching parser, cleans the extracted text, and
    attaches provenance metadata. A PDF yields one document per page; every other
    format yields one document per file.

    Parameters
    ----------
    path : Path
        Absolute or relative path to a supported document.

    Returns
    -------
    list[Document]
        Non-empty cleaned documents. Empty, whitespace-only, or text-free files
        yield ``[]`` rather than raising, so one blank page never blocks a batch.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    UnsupportedFileError
        If the suffix is outside :data:`SUPPORTED_EXTENSIONS`. Subclasses
        :class:`ValueError`.
    CorruptDocumentError
        If a parser cannot read the file.
    """
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"Document not found: {file}")
    if not file.is_file():
        raise IsADirectoryError(f"Expected a file but found a directory: {file}")

    suffix = file.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileError(
            f"Unsupported file type {suffix!r} for {file.name}. Supported: {supported}."
        )

    metadata = _source_metadata(file)

    if suffix == PDF_EXTENSION:
        return _read_pdf(file, metadata)
    if suffix == DOCX_EXTENSION:
        return _read_docx(file, metadata)
    return _read_plain_text(file, metadata)


def load_documents(
    paths: Sequence[Path],
    *,
    continue_on_error: bool = False,
) -> list[Document]:
    """Load and concatenate every document in ``paths``.

    Parameters
    ----------
    paths : Sequence[Path]
        Files returned by :func:`discover_files`.
    continue_on_error : bool
        When ``False`` (default) the first unreadable file raises. When ``True``
        unreadable files are logged at warning level and skipped, so one damaged
        PDF does not discard an otherwise useful knowledge base.

    Returns
    -------
    list[Document]
        One flat list ready for the chunking stage.

    Raises
    ------
    FileNotFoundError
        If a path is missing and ``continue_on_error`` is ``False``.
    UnsupportedFileError
        If a type is unsupported and ``continue_on_error`` is ``False``.
    CorruptDocumentError
        If a file is damaged and ``continue_on_error`` is ``False``.
    """
    documents: list[Document] = []

    for path in paths:
        try:
            documents.extend(file_to_documents(path))
        except (
            OSError,
            UnsupportedFileError,
            CorruptDocumentError,
        ) as exc:
            if not continue_on_error:
                raise
            logger.warning("Skipping %s: %s", path, exc)

    return documents


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
        Matching files in deterministic sorted order. Hidden files, hidden
        directories, and unsupported types are excluded.

    Raises
    ------
    FileNotFoundError
        If ``data_dir`` does not exist or is not a directory.
    """
    directory = Path(data_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Data directory not found: {directory}")

    wanted = {ext.lower() for ext in extensions}

    def is_ingestable(candidate: Path) -> bool:
        if not candidate.is_file():
            return False
        if candidate.suffix.lower() not in wanted:
            return False
        relative_parts = candidate.relative_to(directory).parts
        return not any(part.startswith(".") for part in relative_parts)

    return sorted(
        (found for found in directory.rglob("*") if is_ingestable(found)),
        key=lambda found: str(found).casefold(),
    )


def ingest_directory(
    data_dir: Path,
    *,
    extensions: Sequence[str] = SUPPORTED_EXTENSIONS,
    continue_on_error: bool = True,
) -> list[Document]:
    """Discover and load every readable document in one directory.

    Convenience wrapper for the two steps the pipeline needs, keeping the call
    order in one place.

    Parameters
    ----------
    data_dir : Path
        Directory holding the source documents.
    extensions : Sequence[str]
        Suffixes to consider ingestable.
    continue_on_error : bool
        Skip unreadable files instead of raising; ``True`` by default because a
        user-managed folder often contains a stray damaged file.

    Returns
    -------
    list[Document]
        Flat list of cleaned documents, empty when nothing was found.

    Raises
    ------
    FileNotFoundError
        If ``data_dir`` does not exist.
    """
    return load_documents(
        discover_files(data_dir, extensions),
        continue_on_error=continue_on_error,
    )
