"""Unit tests for :mod:`rag_chatbot.chunking`.

Covered: geometry validation, token-budget compliance at the configured 512/50
settings, sentence-boundary integrity, metadata propagation from source
documents, and empty-input handling.
"""

from __future__ import annotations

import pytest
from llama_index.core import Document
from llama_index.core.schema import TextNode

from rag_chatbot.chunking import build_splitter, split_documents

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def make_document(text: str, **metadata: object) -> Document:
    """Build a document with the provenance fields ingestion would attach.

    Parameters
    ----------
    text : str
        Document body.
    metadata : object
        Extra fields such as ``file_name`` or ``page_number``.

    Returns
    -------
    Document
        Ready to split.
    """
    defaults: dict[str, object] = {"file_name": "unit.txt", "file_type": "txt"}
    defaults.update(metadata)
    return Document(text=text, metadata=defaults)


class TestGeometryValidation:
    """Invalid chunk settings must be rejected before any parsing happens."""

    @pytest.mark.parametrize(
        "chunk_size,chunk_overlap",
        [
            (512, 512),
            (512, 600),
            (100, 100),
        ],
    )
    def test_overlap_not_smaller_than_size_raises(
        self, chunk_size: int, chunk_overlap: int
    ) -> None:
        with pytest.raises(ValueError, match="must be smaller"):
            split_documents(
                [make_document("Some text.")],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

    def test_zero_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            split_documents(
                [make_document("Some text.")], chunk_size=0, chunk_overlap=0
            )

    def test_negative_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            split_documents(
                [make_document("Some text.")], chunk_size=512, chunk_overlap=-1
            )

    def test_overlap_equal_to_zero_is_allowed(self) -> None:
        nodes = split_documents(
            [make_document("First sentence. Second sentence.")],
            chunk_size=512,
            chunk_overlap=0,
        )

        assert len(nodes) == 1


class TestBuildSplitter:
    """The splitter factory is the seam the UI and tests share."""

    def test_returns_configured_splitter(self) -> None:
        splitter = build_splitter(chunk_size=512, chunk_overlap=50)

        assert splitter.chunk_size == 512
        assert splitter.chunk_overlap == 50

    def test_metadata_is_copied_but_not_prefixed_into_chunk_text(self) -> None:
        splitter = build_splitter(chunk_size=512, chunk_overlap=50)

        assert splitter.include_metadata is True

        nodes = splitter.get_nodes_from_documents([make_document("A plain body.")])

        assert nodes[0].get_content() == "A plain body."
        assert "file_name" not in nodes[0].get_content()

    def test_invalid_geometry_raises_from_factory(self) -> None:
        with pytest.raises(ValueError):
            build_splitter(chunk_size=10, chunk_overlap=10)


class TestBasicSplitting:
    """Short text stays whole; long text divides."""

    def test_short_document_produces_one_unchanged_node(self) -> None:
        text = "Retrieval keeps the model honest."
        nodes = split_documents(
            [make_document(text)], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

        assert len(nodes) == 1
        assert nodes[0].get_content() == text

    def test_returns_text_nodes(self) -> None:
        nodes = split_documents(
            [make_document("A sentence.")],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        assert all(isinstance(node, TextNode) for node in nodes)

    def test_long_document_splits_into_several_nodes(self, long_prose: str) -> None:
        nodes = split_documents(
            [make_document(long_prose)],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        assert len(nodes) > 1

    def test_no_node_exceeds_the_token_budget(self, long_prose: str) -> None:
        splitter = build_splitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        nodes = splitter.get_nodes_from_documents([make_document(long_prose)])

        sizes = [splitter._token_size(node.get_content()) for node in nodes]
        assert max(sizes) <= CHUNK_SIZE

    def test_small_budget_still_respects_its_own_limit(self, long_prose: str) -> None:
        splitter = build_splitter(chunk_size=64, chunk_overlap=8)

        nodes = splitter.get_nodes_from_documents([make_document(long_prose)])

        assert len(nodes) > 4
        assert all(splitter._token_size(n.get_content()) <= 64 for n in nodes)

    def test_overlap_repeats_text_between_adjacent_nodes(self, long_prose: str) -> None:
        splitter = build_splitter(chunk_size=64, chunk_overlap=8)

        nodes = splitter.get_nodes_from_documents([make_document(long_prose)])
        texts = [node.get_content() for node in nodes]

        tail = texts[0].split()[-8:]
        assert " ".join(tail) in texts[1]

    def test_zero_overlap_does_not_duplicate_content(self, long_prose: str) -> None:
        splitter = build_splitter(chunk_size=64, chunk_overlap=0)

        nodes = splitter.get_nodes_from_documents([make_document(long_prose)])
        texts = [node.get_content() for node in nodes]

        assert texts[0] != texts[1]
        assert not any(text and text == other for text, other in zip(texts, texts[1:]))


class TestBoundaryIntegrity:
    """Chunks break at word and sentence boundaries, never inside a word."""

    def test_nodes_are_verbatim_slices_of_the_source(self, long_prose: str) -> None:
        document = make_document(long_prose)

        nodes = split_documents(
            [document], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

        for node in nodes:
            assert node.get_content() in long_prose

    def test_slice_edges_fall_on_whitespace(self, long_prose: str) -> None:
        """No chunk may start or finish mid-word."""
        nodes = split_documents(
            [make_document(long_prose)],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        for node in nodes:
            text = node.get_content()
            start = long_prose.find(text)
            end = start + len(text)
            if start > 0:
                assert long_prose[start - 1].isspace()
            if end < len(long_prose):
                assert long_prose[end].isspace()

    def test_paragraph_breaks_survive_inside_a_chunk(self) -> None:
        text = "First paragraph body.\n\nSecond paragraph body."
        nodes = split_documents(
            [make_document(text)], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

        assert "\n\n" in nodes[0].get_content()

    def test_chunk_content_has_no_stray_padding(self, long_prose: str) -> None:
        nodes = split_documents(
            [make_document(long_prose)],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        for node in nodes:
            assert node.get_content() == node.get_content().strip()


class TestMetadataPreservation:
    """Every chunk must still cite the file and page it came from."""

    def test_document_metadata_copies_onto_each_node(self, long_prose: str) -> None:
        document = make_document(
            long_prose, file_name="big.txt", page_number=3, page_count=7
        )

        nodes = split_documents(
            [document], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

        assert len(nodes) > 1
        for node in nodes:
            assert node.metadata["file_name"] == "big.txt"
            assert node.metadata["page_number"] == 3
            assert node.metadata["page_count"] == 7

    def test_pdf_pages_keep_their_own_metadata(self) -> None:
        page_one = make_document(
            "Page one body text.", file_name="report.pdf", page_number=1
        )
        page_two = make_document(
            "Page two body text.", file_name="report.pdf", page_number=2
        )

        nodes = split_documents(
            [page_one, page_two], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

        assert [node.metadata["page_number"] for node in nodes] == [1, 2]

    def test_metadata_dict_is_not_shared_between_documents(self) -> None:
        first = make_document("First document text.", file_name="a.txt")
        second = make_document("Second document text.", file_name="b.txt")

        nodes = split_documents(
            [first, second], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

        names = {node.metadata["file_name"] for node in nodes}
        assert names == {"a.txt", "b.txt"}
        assert len(nodes) == 2


class TestEmptyInputHandling:
    """Nothing here may crash on an empty corpus."""

    def test_no_documents_returns_no_nodes(self) -> None:
        assert (
            split_documents([], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            == []
        )

    def test_blank_documents_are_dropped(self) -> None:
        documents = [
            make_document(""),
            make_document("   \n  "),
            make_document("Real sentence here."),
        ]

        nodes = split_documents(
            documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

        assert len(nodes) == 1
        assert nodes[0].get_content() == "Real sentence here."

    def test_whitespace_only_document_produces_no_empty_node(self) -> None:
        nodes = split_documents(
            [make_document("   \n\n   ")],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        assert nodes == []
