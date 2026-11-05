"""Generation stage: build the Gemini chat model behind the assistant.

Single seam between the rest of the codebase and the Google provider SDK, so the
API key is read only from :mod:`config` and never reaches UI or storage modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llama_index.core.llms import LLM

if TYPE_CHECKING:
    from config import Settings


def build_llm(settings: Settings) -> LLM:
    """Create the Gemini LLM named by ``settings.gemini_model``.

    Intended implementation: return a ``llama_index.llms.google_genai.GoogleGenAI``
    instance built with ``settings.gemini_api_key``, ``settings.gemini_model`` and
    a modest ``temperature`` suited to grounded question answering.

    Parameters
    ----------
    settings : Settings
        Provides ``gemini_api_key`` and ``gemini_model``.

    Returns
    -------
    LLM
        A ready-to-use LlamaIndex chat model.

    Raises
    ------
    NotImplementedError
        Stage not implemented yet.
    """
    raise NotImplementedError("Gemini LLM construction is not implemented yet.")
