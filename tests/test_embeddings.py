"""Unit tests for :mod:`rag_chatbot.embeddings`.

These load a real Sentence Transformers model, so the first run downloads its
weights into the HuggingFace cache and takes about a minute; later runs reuse the
cache and finish in seconds. The model is built once per session.

No test here contacts Gemini, and none reads ``.env`` — settings come from
:func:`conftest.build_settings`, whose only key is a placeholder string.
"""

from __future__ import annotations

import math

import pytest
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from conftest import build_settings
from rag_chatbot.chunking import split_documents
from rag_chatbot.embeddings import EmbeddingModelError, build_embed_model
from rag_chatbot.ingestion import file_to_documents

RELATED_PAIR = ("Dogs are loyal companion animals.", "A dog is a faithful pet.")
UNRELATED_PAIR = (
    "Dogs are loyal companion animals.",
    "Quarterly revenue grew by nine percent year on year.",
)


def cosine(left: list[float], right: list[float]) -> float:
    """Cosine similarity, computed here to keep the test independent of numpy."""
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    return dot / (norm_left * norm_right)


@pytest.fixture(scope="session")
def embed_model() -> HuggingFaceEmbedding:
    """One shared model instance, because loading torch is the slow part."""
    return build_embed_model(build_settings())


class TestModelConstruction:
    """The factory honours configuration and reports usable errors."""

    def test_returns_huggingface_embed_model(self, embed_model) -> None:
        assert isinstance(embed_model, HuggingFaceEmbedding)

    def test_uses_the_configured_model_identifier(self, embed_model) -> None:
        assert embed_model.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_settings_model_name_reaches_the_constructor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify the wiring without downloading a second model."""
        captured: dict[str, object] = {}

        class RecordingEmbedding:
            def __init__(self, model_name: str, **kwargs: object) -> None:
                captured["model_name"] = model_name
                captured["kwargs"] = kwargs

        monkeypatch.setattr(
            "rag_chatbot.embeddings.HuggingFaceEmbedding", RecordingEmbedding
        )
        settings = build_settings(embedding_model="some-org/some-model")

        build_embed_model(settings)

        assert captured["model_name"] == "some-org/some-model"

    def test_unknown_model_raises_wrapped_error(self) -> None:
        settings = build_settings(embedding_model="no-such-org/no-such-model-xyz")

        with pytest.raises(EmbeddingModelError, match="no-such-org"):
            build_embed_model(settings)

    def test_original_failure_is_chained_as_cause(self) -> None:
        settings = build_settings(embedding_model="no-such-org/no-such-model-xyz")

        with pytest.raises(EmbeddingModelError) as excinfo:
            build_embed_model(settings)

        assert excinfo.value.__cause__ is not None

    def test_error_message_names_the_model_and_hints_at_causes(self) -> None:
        settings = build_settings(embedding_model="no-such-org/no-such-model-xyz")

        with pytest.raises(EmbeddingModelError) as excinfo:
            build_embed_model(settings)

        message = str(excinfo.value)
        assert "Check that the identifier is correct" in message
        assert "HF_TOKEN" in message


class TestVectorShape:
    """Every vector must be fixed-width and numerically clean."""

    def test_dimension_is_stable_across_calls(self, embed_model) -> None:
        first = embed_model.get_text_embedding("Alpha.")
        second = embed_model.get_text_embedding("A much longer sentence about ideas.")

        assert len(first) == len(second)
        assert len(first) > 0

    def test_all_values_are_finite(self, embed_model) -> None:
        vector = embed_model.get_text_embedding("ChromaDB persists embeddings.")

        assert all(math.isfinite(value) for value in vector)

    def test_batch_matches_single_embedding_dimensions(self, embed_model) -> None:
        texts = ["First short.", "Second sentence that is noticeably longer."]

        vectors = embed_model.get_text_embedding_batch(
            texts, show_progress_bar=False
        )
        expected_width = len(embed_model.get_text_embedding(texts[0]))

        assert len(vectors) == len(texts)
        assert [len(vector) for vector in vectors] == [expected_width] * len(texts)

    def test_very_long_text_is_still_one_vector(self, embed_model) -> None:
        long_text = "The retriever ranks passages. " * 900

        vector = embed_model.get_text_embedding(long_text)

        assert len(vector) == len(embed_model.get_text_embedding("short"))

    def test_whitespace_text_does_not_crash(self, embed_model) -> None:
        vector = embed_model.get_text_embedding("   \n\t  ")

        assert len(vector) > 0
        assert all(math.isfinite(value) for value in vector)


class TestSemanticQuality:
    """The model must rank related text above unrelated text."""

    def test_vectors_are_deterministic(self, embed_model) -> None:
        first = embed_model.get_text_embedding(RELATED_PAIR[0])
        second = embed_model.get_text_embedding(RELATED_PAIR[0])

        assert cosine(first, second) == pytest.approx(1.0, abs=1e-6)

    def test_related_sentences_score_above_unrelated(self, embed_model) -> None:
        related = cosine(
            embed_model.get_text_embedding(RELATED_PAIR[0]),
            embed_model.get_text_embedding(RELATED_PAIR[1]),
        )
        unrelated = cosine(
            embed_model.get_text_embedding(UNRELATED_PAIR[0]),
            embed_model.get_text_embedding(UNRELATED_PAIR[1]),
        )

        assert related > unrelated

    def test_batch_and_single_agree(self, embed_model) -> None:
        text = "Vector stores keep embeddings near their source text."

        batch = embed_model.get_text_embedding_batch([text], show_progress_bar=False)[0]
        single = embed_model.get_text_embedding(text)

        assert cosine(batch, single) == pytest.approx(1.0, abs=1e-4)


class TestWithRealChunks:
    """End-to-end proof that ingestion and chunking feed this stage directly."""

    def test_every_chunk_vectorises_to_the_same_width(
        self, tmp_path, embed_model
    ) -> None:
        document = tmp_path / "corpus.txt"
        document.write_text(
            " ".join(
                f"Section {index} explains how retrieval augments generation."
                for index in range(1, 200)
            ),
            encoding="utf-8",
        )
        settings = build_settings(data_dir=tmp_path)

        nodes = split_documents(
            file_to_documents(document),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        vectors = embed_model.get_text_embedding_batch(
            [node.get_content() for node in nodes], show_progress_bar=False
        )

        assert len(nodes) > 1
        assert len(vectors) == len(nodes)
        assert {len(vector) for vector in vectors} == {len(vectors[0])}
