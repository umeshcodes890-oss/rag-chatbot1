"""Streamlit entry point for the RAG chatbot.

Presentation only: page layout, session state, and rendering. Every pipeline
call goes through :mod:`rag_chatbot.pipeline`, and no stage logic, prompt text,
or credential handling lives in this file.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

from typing import Sequence

import streamlit as st

from config import ConfigError, Settings, load_settings
from rag_chatbot import pipeline
from rag_chatbot.chatbot import ChatTurn
from rag_chatbot.ingestion import SUPPORTED_EXTENSIONS
from rag_chatbot.retrieval import Answer, Source
from rag_chatbot.vector_store import index_exists, reset_index

st.set_page_config(page_title="RAG Chatbot", layout="centered")

_NOT_IMPLEMENTED = (
    "The RAG pipeline is scaffolded but not implemented yet. "
    "Each stage in `rag_chatbot/` still needs its logic."
)


@st.cache_resource(show_spinner="Loading configuration...")
def get_settings() -> Settings:
    """Load and cache configuration, exiting the page cleanly if it is invalid.

    Returns
    -------
    Settings
        Validated configuration, or stops execution with an error message when
        ``.env`` is missing or incomplete.
    """
    try:
        return load_settings()
    except ConfigError as exc:
        st.error(str(exc))
        st.stop()


def _index_ready(settings: Settings) -> bool:
    """Report whether the page can offer chat, tolerating unimplemented stages.

    Parameters
    ----------
    settings : Settings
        Configuration identifying the collection.

    Returns
    -------
    bool
        ``False`` while :func:`rag_chatbot.vector_store.index_exists` is still a
        stub, so the UI shows the "no index" state instead of a traceback.
    """
    try:
        return index_exists(settings)
    except NotImplementedError:
        return False


def render_sidebar(settings: Settings) -> None:
    """Draw the configuration panel and the ingest/reset controls.

    Parameters
    ----------
    settings : Settings
        Configuration to display and act on.
    """
    st.sidebar.header("Knowledge base")
    st.sidebar.caption(f"Documents: `{settings.data_dir}`")
    st.sidebar.caption(f"Collection: `{settings.chroma_collection}`")
    st.sidebar.caption(f"Chunk: {settings.chunk_size} / overlap {settings.chunk_overlap}")
    st.sidebar.caption(f"Top-K: {settings.similarity_top_k}")
    st.sidebar.caption(f"Embedding: `{settings.embedding_model}`")
    st.sidebar.caption(f"Chat model: `{settings.gemini_model}`")
    st.sidebar.caption(f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}")

    if st.sidebar.button("Ingest documents", use_container_width=True):
        with st.spinner("Indexing documents..."):
            try:
                pipeline.ingest_documents(settings)
            except NotImplementedError:
                st.info(_NOT_IMPLEMENTED)
            else:
                st.cache_resource.clear()
                st.sidebar.success("Index rebuilt.")

    if st.sidebar.button("Clear index", use_container_width=True):
        try:
            reset_index(settings)
        except NotImplementedError:
            st.info(_NOT_IMPLEMENTED)
        else:
            st.cache_resource.clear()
            st.sidebar.success("Index deleted.")

    st.sidebar.divider()
    st.sidebar.caption(
        "Your API key is read from `.env` at start-up and never leaves "
        ":mod:`config`."
    )


def render_message(
    role: str, content: str, sources: Sequence[Source] = ()
) -> None:
    """Render one chat bubble, with its sources attached to the assistant turn.

    Parameters
    ----------
    role : str
        ``"user"`` or ``"assistant"``.
    content : str
        Message body.
    sources : Sequence[Source]
        Retrieved chunks backing an assistant reply; ignored for user turns.
    """
    with st.chat_message(role):
        st.markdown(content)
        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for source in sources:
                    label = (
                        source.document_name
                        if source.score is None
                        else f"{source.document_name} · {source.score:.3f}"
                    )
                    st.markdown(f"**{label}**")
                    st.caption(source.text)


def ask_question(settings: Settings, question: str) -> Answer:
    """Send one turn through the chatbot and return its answer.

    Parameters
    ----------
    settings : Settings
        Configuration used to assemble the assistant.
    question : str
        The user's message.

    Returns
    -------
    Answer
        Reply text plus the chunks it was grounded in.

    Raises
    ------
    FileNotFoundError
        If the knowledge base has not been ingested yet.
    """
    history = tuple(
        ChatTurn(role=turn["role"], content=turn["content"])
        for turn in st.session_state.messages
    )
    bot = pipeline.build_chatbot(settings)
    return bot.ask(question, history=history)


def main() -> None:
    """Build the page and drive the chat loop."""
    settings = get_settings()
    render_sidebar(settings)

    st.title("Document Q&A")
    st.caption("Answers are generated from your ingested documents only.")

    for turn in st.session_state.get("messages", []):
        render_message(turn["role"], turn["content"], turn.get("sources", ()))

    if not _index_ready(settings):
        st.info("No index yet — add documents to the data folder and ingest them.")
        return

    question = st.chat_input("Ask a question about your documents")
    if not question:
        return

    render_message("user", question)
    st.session_state.setdefault("messages", []).append(
        {"role": "user", "content": question}
    )

    try:
        with st.spinner("Thinking..."):
            answer = ask_question(settings, question)
    except NotImplementedError:
        st.info(_NOT_IMPLEMENTED)
        return
    except FileNotFoundError:
        st.warning("The index disappeared. Please ingest documents again.")
        return

    render_message("assistant", answer.text, answer.sources)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer.text, "sources": answer.sources}
    )


if __name__ == "__main__":
    main()
