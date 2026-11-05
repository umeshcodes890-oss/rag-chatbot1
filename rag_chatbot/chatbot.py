"""Chatbot stage: conversation orchestration on top of retrieval.

The only module the UI talks to. It manages turn history and the system prompt,
and delegates the actual lookup and generation work to
:mod:`rag_chatbot.retrieval`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Sequence

if TYPE_CHECKING:
    from llama_index.core import VectorStoreIndex
    from config import Settings
    from rag_chatbot.retrieval import Answer

Role = Literal["user", "assistant"]

DEFAULT_SYSTEM_PROMPT = (
    "You are a document question-answering assistant. Answer only from the "
    "provided context. If the context does not contain the answer, say so "
    "plainly instead of guessing, and cite the source document."
)


@dataclass(frozen=True, slots=True)
class ChatTurn:
    """A single conversation message.

    Attributes
    ----------
    role : Role
        Who produced the message.
    content : str
        Message body.
    """

    role: Role
    content: str


class Chatbot:
    """Stateful assistant bound to one ingested index.

    Typical use from the UI layer::

        bot = Chatbot(settings, index)
        answer = bot.ask("What is the renewal deadline?", history=turns)

    Keeping this as a class rather than module-level functions lets the
    conversation hold its own LLM session without leaking state between requests.
    """

    def __init__(
        self,
        settings: "Settings",
        index: "VectorStoreIndex",
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        """Bind the assistant to configuration and a built index.

        Parameters
        ----------
        settings : Settings
            Validated application configuration.
        index : VectorStoreIndex
            Index produced by :mod:`rag_chatbot.vector_store`.
        system_prompt : str
            Instruction framing every reply.
        """
        self._settings = settings
        self._index = index
        self._system_prompt = system_prompt

    @property
    def system_prompt(self) -> str:
        """The instruction prefixing every generation call."""
        return self._system_prompt

    def ask(self, question: str, history: Sequence[ChatTurn] = ()) -> "Answer":
        """Answer ``question`` using the indexed documents.

        Parameters
        ----------
        question : str
            The user's latest message.
        history : Sequence[ChatTurn]
            Prior turns, oldest first.

        Returns
        -------
        Answer
            The reply text and the chunks it was grounded in.

        Raises
        ------
        NotImplementedError
            Chat logic is not implemented yet; it will call
            :func:`rag_chatbot.retrieval.answer_question`.
        """
        raise NotImplementedError("Chat orchestration is not implemented yet.")
