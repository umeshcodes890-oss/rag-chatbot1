"""Embedding stage: build the HuggingFace embed model used to vectorise text.

Vector embeddings run locally through ``sentence-transformers``; Gemini is used
only for generation. Keeping this separate from :mod:`rag_chatbot.llm` lets the
embedding model be swapped without touching retrieval or chat logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llama_index.core.embeddings import BaseEmbedding

if TYPE_CHECKING:
    from config import Settings


def build_embed_model(settings: Settings) -> BaseEmbedding:
    """Create the embed model described by ``settings.embedding_model``.

    Intended implementation: return a
    ``llama_index.embeddings.huggingface.HuggingFaceEmbedding`` configured with
    ``settings.embedding_model`` and, when present, ``settings.hf_token`` for
    gated models.

    The model is resolved once at start-up and cached by the caller, because the
    first call downloads the weights and is expensive.

    Parameters
    ----------
    settings : Settings
        Provides ``embedding_model`` and the optional ``hf_token``.

    Returns
    -------
    BaseEmbedding
        A ready-to-use LlamaIndex embed model.

    Raises
    ------
    NotImplementedError
        Stage not implemented yet.
    """
    raise NotImplementedError("Embedding model construction is not implemented yet.")
