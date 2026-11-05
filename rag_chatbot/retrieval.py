"""Retrieval stage: fetch relevant chunks and compose a grounded answer.

Produces the :class:`Answer` value object the UI renders. It reads from the index
but never writes to it, and holds no conversation state — that is the chatbot's
job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore

if TYPE_CHECKING:
    from llama_index.core import VectorStoreIndex
    from llama_index.core.llms import LLM
    from config import Settings
    from rag_chatbot.embeddings import BaseEmbedding


@dataclass(frozen=True, slots=True)
class Source:
    """One cited chunk, shown to the user beneath the answer.

    Attributes
    ----------
    document_name : str
        Basename of the file the chunk came from.
    score : float | None
        Relevance score, when the retriever supplies one.
    text : str
        The retrieved chunk verbatim.
    """

    document_name: str
    score: float | None
    text: str


@dataclass(frozen=True, slots=True)
class Answer:
    """A generated reply plus the evidence it was drawn from.

    Attributes
    ----------
    text : str
        The assistant's answer.
    sources : tuple[Source, ...]
        Chunks used as context, ordered by relevance.
    """

    text: str
    sources: tuple[Source, ...]


def build_retriever(index: VectorStoreIndex, *, top_k: int) -> BaseRetriever:
    """Return a vector retriever over ``index`` limited to ``top_k`` hits.

    Parameters
    ----------
    index : VectorStoreIndex
        Index built by :mod:`rag_chatbot.vector_store`.
    top_k : int
        Maximum chunks returned per query.

    Returns
    -------
    BaseRetriever
        Configured retriever used by :func:`retrieve_nodes`.

    Raises
    ------
    NotImplementedError
        Stage not implemented yet.
    """
    raise NotImplementedError("Retriever setup is not implemented yet.")


def retrieve_nodes(retriever: BaseRetriever, query: str) -> list[NodeWithScore]:
    """Run ``query`` against the vector store and return matching chunks.

    Parameters
    ----------
    retriever : BaseRetriever
        From :func:`build_retriever`.
    query : str
        The user's question.

    Returns
    -------
    list[NodeWithScore]
        Ranked nodes with their relevance scores.

    Raises
    ------
    NotImplementedError
        Stage not implemented yet.
    """
    raise NotImplementedError("Node retrieval is not implemented yet.")


def answer_question(
    query: str,
    *,
    index: VectorStoreIndex,
    llm: "LLM",
    settings: Settings,
    history: Sequence[str] = (),
) -> Answer:
    """Retrieve context for ``query`` and generate a cited answer.

    Intended implementation: retrieve nodes, build a prompt that restricts the
    model to the supplied context, call the LLM, and package the reply with the
    chunks actually used.

    Parameters
    ----------
    query : str
        The current user question.
    index : VectorStoreIndex
        Source of retrieved chunks.
    llm : LLM
        Generation model from :mod:`rag_chatbot.llm`.
    settings : Settings
        Provides ``similarity_top_k``.
    history : Sequence[str]
        Prior turns, used to resolve follow-up questions.

    Returns
    -------
    Answer
        Generated text together with its sources.

    Raises
    ------
    NotImplementedError
        Stage not implemented yet.
    """
    raise NotImplementedError("Grounded answer generation is not implemented yet.")
