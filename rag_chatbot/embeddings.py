"""Embedding stage: build the HuggingFace embed model used to vectorise text.

Vector embeddings run locally through ``sentence-transformers``; Gemini is used
only for generation. Keeping this separate from :mod:`rag_chatbot.llm` lets the
embedding model be swapped without touching retrieval or chat logic.

The first call for a new model downloads its weights to the HuggingFace cache and
is slow; every later call reuses the cached copy. Callers should therefore build
the model once and keep it, which is what ``app.py`` does through
``st.cache_resource``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

if TYPE_CHECKING:
    from config import Settings


class EmbeddingModelError(RuntimeError):
    """The embedding model could not be downloaded, loaded, or initialised."""


def build_embed_model(settings: Settings) -> BaseEmbedding:
    """Create the embed model named by ``settings.embedding_model``.

    ``HuggingFaceEmbedding`` exposes no ``token`` argument, so authentication for
    a gated model is picked up from the environment by ``huggingface_hub``. That
    already happens: :func:`config.load_settings` loads ``.env`` before this runs,
    so ``HF_TOKEN`` is in the process environment by the time a download starts.
    The token is never passed as an argument or stored on the returned object.

    Parameters
    ----------
    settings : Settings
        Provides ``embedding_model``, the Sentence Transformers model identifier
        or a local filesystem path to a model directory.

    Returns
    -------
    BaseEmbedding
        A ready-to-use LlamaIndex embed model.

    Raises
    ------
    EmbeddingModelError
        If the model cannot be fetched or loaded. The cause is chained so the
        original error stays visible, and the message names the model that
        failed.
    """
    try:
        return HuggingFaceEmbedding(model_name=settings.embedding_model)
    except Exception as exc:
        raise EmbeddingModelError(
            f"Could not load embedding model {settings.embedding_model!r}: {exc}. "
            "Check that the identifier is correct, that you are online for the "
            "first download, and that HF_TOKEN is set if the model is gated."
        ) from exc
