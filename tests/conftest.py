"""Shared pytest fixtures: real PDF, DOCX, and text files on disk.

Files are generated instead of mocked so the readers, parsers, and metadata
handling are exercised end to end. Every fixture lands in ``tmp_path``, which
pytest removes after the session.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

from config import Settings

# test_geminiapi.py is a stand-alone script, not a pytest module: its body calls
# the live Gemini API at import time, so letting pytest import it would spend a
# real request.
collect_ignore = ["test_geminiapi.py"]

PLACEHOLDER_API_KEY = "test-key-not-a-secret"
DOC_PARAGRAPH = (
    "Retrieval-augmented generation grounds a language model in source documents. "
    "The retriever selects the passages that the generator must obey."
)
SECOND_PARAGRAPH = (
    "Chunk boundaries matter because a passage split across two chunks loses context. "
    "An overlap keeps the linking sentence available to both chunks."
)


def _escape_pdf_text(text: str) -> str:
    """Escape the three characters that are special inside a PDF literal string."""
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(pages: list[str]) -> bytes:
    """Assemble a minimal but structurally valid multi-page PDF.

    Each page renders ``pages[i]`` as one Helvetica 12pt line. Cross-reference
    offsets are measured from the emitted bytes so ``pypdf`` parses the file the
    same way it would parse a real export.

    Parameters
    ----------
    pages : list[str]
        One plain-text line per page.

    Returns
    -------
    bytes
        PDF file contents.
    """
    page_count = len(pages)
    first_page_id = 4  # Catalog=1, Pages=2, Font=3, then page/content pairs.
    object_count = 3 + 2 * page_count

    kids = " ".join(f"{first_page_id + 2 * index} 0 R" for index in range(page_count))
    bodies: dict[int, str] = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>",
        3: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }

    for index, line in enumerate(pages):
        stream = f"BT /F1 12 Tf 14 TL 72 720 Td ({_escape_pdf_text(line)}) Tj ET"
        page_id = first_page_id + 2 * index
        content_id = page_id + 1
        bodies[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        )
        bodies[content_id] = (
            f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"
        )

    chunks: list[bytes] = [b"%PDF-1.4\n"]
    position = len(chunks[0])
    offsets: list[tuple[int, int]] = []

    for number in range(1, object_count + 1):
        offsets.append((number, position))
        data = f"{number} 0 obj\n{bodies[number]}\nendobj\n".encode("latin-1")
        chunks.append(data)
        position += len(data)

    xref_offset = position
    rows = [b"xref\n0 %d\n" % (object_count + 1), b"0000000000 65535 f \n"]
    rows += [b"%010d 00000 n \n" % offset for _number, offset in offsets]

    chunks.extend(rows)
    chunks.append(
        (
            f"trailer\n<< /Size {object_count + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("latin-1")
    )
    return b"".join(chunks)


def build_docx(paragraphs: list[str]) -> bytes:
    """Assemble a ``.docx`` archive that ``docx2txt`` can read.

    Only ``word/document.xml`` is inspected by the parser, but the content types
    and relationship parts are included so the archive resembles a real export.

    Parameters
    ----------
    paragraphs : list[str]
        One ``<w:p>`` element per string.

    Returns
    -------
    bytes
        ZIP archive contents.
    """
    body = "".join(
        f'<w:p><w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'
        for text in paragraphs
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-'
        'package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.'
        'openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
        'relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/'
        'officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()


@pytest.fixture
def sample_text(tmp_path: Path) -> Path:
    """A ``.txt`` file with messy whitespace around two real paragraphs."""
    path = tmp_path / "fieldnotes.txt"
    path.write_text(
        "  Retrieval-augmented   generation grounds\tthe model.\n\n\n\n"
        "Chunk   boundaries matter.\r\n"
        "An overlap keeps context.\n\n\n\n\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """A generated two-page PDF with distinct text on each page."""
    path = tmp_path / "report.pdf"
    path.write_bytes(
        build_pdf(
            [
                "Page one: vector stores keep embeddings nearby.",
                "Page two: ChromaDB persists them between runs.",
            ]
        )
    )
    return path


@pytest.fixture
def sample_docx(tmp_path: Path) -> Path:
    """A generated DOCX with two paragraphs."""
    path = tmp_path / "notes.docx"
    path.write_bytes(build_docx([DOC_PARAGRAPH, SECOND_PARAGRAPH]))
    return path


@pytest.fixture
def long_prose() -> str:
    """Prose long enough to force several 512-token chunks.

    Sentences are numbered so no two chunks can be identical by coincidence,
    which keeps duplicate-content assertions meaningful.
    """
    return " ".join(
        f"Item {index:03d}: the retriever ranks passages by similarity "
        f"and returns only the closest matches to the question."
        for index in range(1, 121)
    )


def build_settings(**overrides: object) -> Settings:
    """Construct :class:`config.Settings` with test-safe defaults.

    Tests build configuration directly instead of calling ``load_settings()``,
    so the suite never depends on a developer's ``.env`` and every credential is
    an obvious placeholder. Pass ``overrides`` to change one field.

    Returns
    -------
    Settings
        Configuration pointing at inert paths.
    """
    base: dict[str, object] = {
        "gemini_api_key": PLACEHOLDER_API_KEY,
        "gemini_model": "gemini-test-model",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "hf_token": None,
        "chroma_collection": "test_collection",
        "data_dir": Path("data"),
        "chroma_dir": Path("vectorstore"),
        "chunk_size": 512,
        "chunk_overlap": 50,
        "similarity_top_k": 3,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture
def make_settings(tmp_path: Path):
    """Return a factory producing settings scoped to ``tmp_path``."""

    def factory(**overrides: object) -> Settings:
        defaults: dict[str, object] = {
            "data_dir": tmp_path / "data",
            "chroma_dir": tmp_path / "chroma",
        }
        defaults.update(overrides)
        return build_settings(**defaults)

    return factory
