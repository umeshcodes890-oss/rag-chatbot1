"""Unit tests for :mod:`rag_chatbot.ingestion`.

Covered: whitespace cleaning rules, file discovery, per-format loading, metadata
preservation, and the four failure classes required of this stage (unsupported
type, empty file, corrupted document, missing file).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from llama_index.core import Document

from conftest import build_docx, build_pdf
from rag_chatbot.ingestion import (
    SUPPORTED_EXTENSIONS,
    CorruptDocumentError,
    UnsupportedFileError,
    clean_text,
    discover_files,
    file_to_documents,
    ingest_directory,
    load_documents,
)


class TestCleanText:
    """Whitespace normalisation must tidy layout without rewriting content."""

    def test_empty_and_none_are_safe(self) -> None:
        assert clean_text(None) == ""
        assert clean_text("") == ""
        assert clean_text("   \n\t ") == ""

    @pytest.mark.parametrize("line_break", ["\r\n", "\r", "\n"])
    def test_line_breaks_normalise_to_lf(self, line_break: str) -> None:
        assert clean_text(f"alpha{line_break}beta") == "alpha\nbeta"

    def test_internal_space_and_tab_runs_collapse(self) -> None:
        assert clean_text("retrieval    augmented\tgeneration") == (
            "retrieval augmented generation"
        )

    def test_trailing_whitespace_is_stripped_per_line(self) -> None:
        assert clean_text("first line    \nsecond line\t\t") == (
            "first line\nsecond line"
        )

    def test_runs_of_blank_lines_collapse_to_one_paragraph_break(self) -> None:
        assert clean_text("para one\n\n\n\n\npara two") == "para one\n\npara two"

    def test_single_newline_inside_a_paragraph_is_kept(self) -> None:
        assert clean_text("line one\nline two") == "line one\nline two"

    def test_non_breaking_space_becomes_ordinary_space(self) -> None:
        assert clean_text("chunk\u00a0overlap") == "chunk overlap"

    def test_zero_width_characters_are_removed(self) -> None:
        assert clean_text("chunk\u200boverlap") == "chunkoverlap"
        assert clean_text("a\u200b  b") == "a b"

    def test_punctuation_and_casing_survive(self) -> None:
        source = "RAG (retrieval-augmented generation) works; it costs 3 cents."
        assert clean_text(source) == source

    def test_outer_whitespace_is_trimmed(self) -> None:
        assert clean_text("\n\n   hello world  \n\n") == "hello world"


class TestDiscoverFiles:
    """Discovery is recursive, deterministic, and type-filtered."""

    def test_finds_supported_types_recursively_in_sorted_order(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "zeta.txt").write_text("z", encoding="utf-8")
        (tmp_path / "alpha.txt").write_text("a", encoding="utf-8")
        nested = tmp_path / "sub" / "deep"
        nested.mkdir(parents=True)
        (nested / "beta.md").write_text("b", encoding="utf-8")

        found = discover_files(tmp_path)

        names = [path.name for path in found]
        assert names == ["alpha.txt", "beta.md", "zeta.txt"]

    def test_skips_unsupported_and_hidden_entries(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("keep", encoding="utf-8")
        (tmp_path / "sheet.xlsx").write_bytes(b"binary")
        (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
        hidden = tmp_path / ".private"
        hidden.mkdir()
        (hidden / "secret.txt").write_text("hide", encoding="utf-8")

        found = discover_files(tmp_path)

        assert [path.name for path in found] == ["notes.txt"]

    def test_honours_custom_extension_filter(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("a", encoding="utf-8")
        (tmp_path / "b.md").write_text("b", encoding="utf-8")

        assert [path.name for path in discover_files(tmp_path, [".md"])] == ["b.md"]

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discover_files(tmp_path / "absent")

    def test_ignores_directories_matching_extension(self, tmp_path: Path) -> None:
        (tmp_path / "looks.like.folder.txt").mkdir()

        assert discover_files(tmp_path) == []


class TestPlainTextLoading:
    """TXT and MD load through a direct read with cleaning applied."""

    def test_txt_loads_with_cleaned_text_and_metadata(
        self, sample_text: Path
    ) -> None:
        documents = file_to_documents(sample_text)

        assert len(documents) == 1
        text = documents[0].text
        assert "Retrieval-augmented generation grounds the model." in text
        assert "Chunk boundaries matter." in text
        assert "\t" not in text and "   " not in text and "\r" not in text

    def test_metadata_records_source_provenance(self, sample_text: Path) -> None:
        metadata = file_to_documents(sample_text)[0].metadata

        assert metadata["file_name"] == "fieldnotes.txt"
        assert metadata["file_path"] == str(sample_text.resolve())
        assert metadata["file_type"] == "txt"
        assert metadata["file_size_bytes"] == sample_text.stat().st_size
        assert "last_modified" in metadata

    def test_markdown_is_treated_as_text(self, tmp_path: Path) -> None:
        path = tmp_path / "guide.md"
        path.write_text("# Title\n\nSome   body text.", encoding="utf-8")

        documents = file_to_documents(path)

        assert documents[0].metadata["file_type"] == "md"
        assert "Some body text." in documents[0].text

    def test_empty_file_yields_no_documents(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.txt"
        path.write_bytes(b"")

        assert file_to_documents(path) == []

    def test_whitespace_only_file_yields_no_documents(self, tmp_path: Path) -> None:
        path = tmp_path / "blank.txt"
        path.write_text("  \n\n\t  \n", encoding="utf-8")

        assert file_to_documents(path) == []

    def test_non_utf8_bytes_fall_back_latin1(self, tmp_path: Path) -> None:
        path = tmp_path / "legacy.txt"
        path.write_bytes("café résumé".encode("latin-1"))

        documents = file_to_documents(path)

        assert "caf" in documents[0].text


class TestPdfLoading:
    """PDFs become one document per page, numbered."""

    def test_each_page_becomes_its_own_document(self, sample_pdf: Path) -> None:
        documents = file_to_documents(sample_pdf)

        assert len(documents) == 2
        assert "Page one" in documents[0].text
        assert "Page two" in documents[1].text

    def test_page_metadata_is_preserved(self, sample_pdf: Path) -> None:
        documents = file_to_documents(sample_pdf)

        first, second = documents
        assert first.metadata["page_number"] == 1
        assert second.metadata["page_number"] == 2
        assert first.metadata["page_label"] == "1"
        assert first.metadata["page_count"] == 2
        assert second.metadata["file_name"] == "report.pdf"

    def test_corrupt_pdf_raises_typed_error(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.pdf"
        path.write_bytes(b"%PDF-1.4\nthis is not a pdf structure\n%%EOF\n")

        with pytest.raises(CorruptDocumentError):
            file_to_documents(path)


class TestDocxLoading:
    """DOCX flattens to one document; pagination is unavailable."""

    def test_paragraphs_load_into_one_document(self, sample_docx: Path) -> None:
        documents = file_to_documents(sample_docx)

        assert len(documents) == 1
        assert "Retrieval-augmented generation grounds a language model" in (
            documents[0].text
        )
        assert "An overlap keeps the linking sentence" in documents[0].text

    def test_metadata_has_no_page_fields(self, sample_docx: Path) -> None:
        metadata = file_to_documents(sample_docx)[0].metadata

        assert metadata["file_type"] == "docx"
        assert "page_number" not in metadata

    def test_empty_document_yields_no_documents(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.docx"
        path.write_bytes(build_docx([""]))

        assert file_to_documents(path) == []

    def test_corrupt_docx_raises_typed_error(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.docx"
        path.write_bytes(b"PK not really a zip archive")

        with pytest.raises(CorruptDocumentError):
            file_to_documents(path)


class TestFileValidation:
    """Unsupported, missing, and directory inputs fail with clear messages."""

    def test_unsupported_extension_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "book.epub"
        path.write_bytes(b"binary")

        with pytest.raises(UnsupportedFileError) as excinfo:
            file_to_documents(path)

        assert isinstance(excinfo.value, ValueError)
        assert ".epub" in str(excinfo.value)

    def test_extensionless_file_is_unsupported(self, tmp_path: Path) -> None:
        path = tmp_path / "README"
        path.write_text("text", encoding="utf-8")

        with pytest.raises(UnsupportedFileError):
            file_to_documents(path)

    def test_unsupported_error_lists_supported_types(self, tmp_path: Path) -> None:
        path = tmp_path / "clip.mp4"
        path.write_bytes(b"binary")

        with pytest.raises(UnsupportedFileError) as excinfo:
            file_to_documents(path)

        for extension in SUPPORTED_EXTENSIONS:
            assert extension in str(excinfo.value)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            file_to_documents(tmp_path / "absent.txt")

    def test_directory_instead_of_file_raises(self, tmp_path: Path) -> None:
        folder = tmp_path / "somedir.txt"
        folder.mkdir()

        with pytest.raises(IsADirectoryError):
            file_to_documents(folder)

    def test_case_insensitive_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "upper.TXT"
        path.write_text("content here", encoding="utf-8")

        documents = file_to_documents(path)

        assert documents[0].metadata["file_type"] == "txt"


class TestLoadDocuments:
    """Batch loading concatenates, and error policy is explicit."""

    def test_concatenates_in_path_order(self, tmp_path: Path) -> None:
        first = tmp_path / "a.txt"
        second = tmp_path / "b.pdf"
        first.write_text("First document body.", encoding="utf-8")
        second.write_bytes(build_pdf(["Second document body."]))

        documents = load_documents([first, second])

        assert len(documents) == 2
        assert documents[0].metadata["file_name"] == "a.txt"
        assert documents[1].metadata["file_name"] == "b.pdf"

    def test_returns_document_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "a.txt"
        path.write_text("body", encoding="utf-8")

        assert all(isinstance(item, Document) for item in load_documents([path]))

    def test_empty_input_returns_empty_list(self) -> None:
        assert load_documents([]) == []

    def test_strict_mode_raises_on_first_corrupt_file(self, tmp_path: Path) -> None:
        good = tmp_path / "good.txt"
        bad = tmp_path / "bad.pdf"
        good.write_text("fine", encoding="utf-8")
        bad.write_bytes(b"%PDF-1.4\nnonsense\n%%EOF\n")

        with pytest.raises(CorruptDocumentError):
            load_documents([good, bad])

    def test_continue_on_error_skips_bad_file_and_keeps_good(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        good = tmp_path / "good.txt"
        missing = tmp_path / "gone.txt"
        bad = tmp_path / "bad.pdf"
        good.write_text("kept content", encoding="utf-8")
        bad.write_bytes(b"%PDF-1.4\nnonsense\n%%EOF\n")

        with caplog.at_level(logging.WARNING, logger="rag_chatbot.ingestion"):
            documents = load_documents(
                [good, bad, missing], continue_on_error=True
            )

        assert [item.metadata["file_name"] for item in documents] == ["good.txt"]
        assert len(caplog.records) == 2
        assert all(record.levelno == logging.WARNING for record in caplog.records)

    def test_blank_files_do_not_interrupt_a_batch(self, tmp_path: Path) -> None:
        blank = tmp_path / "blank.txt"
        filled = tmp_path / "filled.txt"
        blank.write_text("   \n\n", encoding="utf-8")
        filled.write_text("real content", encoding="utf-8")

        documents = load_documents([blank, filled])

        assert len(documents) == 1


class TestIngestDirectory:
    """The convenience wrapper combines discovery and loading."""

    def test_ingests_every_supported_file(
        self, tmp_path: Path, sample_pdf: Path, sample_docx: Path
    ) -> None:
        data_dir = tmp_path / "corpus"
        data_dir.mkdir()
        (data_dir / "note.txt").write_text("A plain note.", encoding="utf-8")
        (data_dir / "report.pdf").write_bytes(sample_pdf.read_bytes())
        (data_dir / "notes.docx").write_bytes(sample_docx.read_bytes())

        documents = ingest_directory(data_dir)

        names = sorted({item.metadata["file_name"] for item in documents})
        assert names == ["note.txt", "notes.docx", "report.pdf"]
        assert len(documents) == 4  # 1 txt + 2 pdf pages + 1 docx

    def test_skips_corrupt_file_by_default(
        self, tmp_path: Path
    ) -> None:
        data_dir = tmp_path / "corpus"
        data_dir.mkdir()
        (data_dir / "good.txt").write_text("usable", encoding="utf-8")
        (data_dir / "bad.pdf").write_bytes(b"%PDF-1.4\njunk\n%%EOF\n")

        documents = ingest_directory(data_dir)

        assert len(documents) == 1

    def test_empty_directory_returns_no_documents(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "nothing"
        data_dir.mkdir()

        assert ingest_directory(data_dir) == []
