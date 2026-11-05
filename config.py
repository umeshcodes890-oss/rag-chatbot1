"""Central application configuration.

Loads environment variables once, validates them, and exposes an immutable
:class:`Settings` object. No other module reads ``os.environ`` directly, and no
API key is ever hardcoded or logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent

DEFAULT_CHROMA_COLLECTION: Final[str] = "rag_documents"
DEFAULT_GEMINI_MODEL: Final[str] = "gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE: Final[int] = 512
DEFAULT_CHUNK_OVERLAP: Final[int] = 50
DEFAULT_SIMILARITY_TOP_K: Final[int] = 3


class ConfigError(RuntimeError):
    """Raised when a required setting is missing or an optional value is invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated, immutable snapshot of the runtime configuration.

    Attributes
    ----------
    data_dir : Path
        Directory holding the source documents to ingest.
    chroma_dir : Path
        Filesystem path where ChromaDB persists its collection.
    """

    gemini_api_key: str
    gemini_model: str
    embedding_model: str
    hf_token: str | None
    chroma_collection: str
    data_dir: Path
    chroma_dir: Path
    chunk_size: int
    chunk_overlap: int
    similarity_top_k: int

    def __repr__(self) -> str:
        """Return the settings with the API key masked."""
        return (
            f"Settings(gemini_model={self.gemini_model!r}, "
            f"embedding_model={self.embedding_model!r}, "
            f"gemini_api_key={'***' if self.gemini_api_key else None}, "
            f"chroma_collection={self.chroma_collection!r}, "
            f"data_dir={self.data_dir!s}, chroma_dir={self.chroma_dir!s}, "
            f"chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}, "
            f"similarity_top_k={self.similarity_top_k})"
        )

    __str__ = __repr__


def _require(name: str) -> str:
    """Return a non-empty environment value or raise :class:`ConfigError`."""
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(
            f"Missing required environment variable {name!r}. "
            "Copy .env.example to .env and fill in your credentials."
        )
    return value


def _positive_int(name: str, default: int) -> int:
    """Read an optional integer setting, enforcing a value of at least 1."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}.") from exc
    if value < 1:
        raise ConfigError(f"{name} must be a positive integer, got {value}.")
    return value


def _path(name: str, default: Path) -> Path:
    """Read an optional filesystem path, falling back to ``default``."""
    raw = os.getenv(name, "").strip()
    return Path(raw) if raw else default


def _optional(name: str, default: str = "") -> str:
    """Read an optional string setting, trimmed and substituted if blank."""
    return os.getenv(name, "").strip() or default


def load_settings(env_file: Path | None = None) -> Settings:
    """Load ``.env`` and build the application :class:`Settings`.

    Parameters
    ----------
    env_file : Path | None
        Explicit dotenv path. Defaults to ``<project root>/.env``.

    Returns
    -------
    Settings
        Validated configuration with secrets already resolved.

    Raises
    ------
    ConfigError
        If a required variable is absent or a numeric setting is invalid.
    """
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)

    chunk_size = _positive_int("CHUNK_SIZE", DEFAULT_CHUNK_SIZE)
    chunk_overlap = _positive_int("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)
    if chunk_overlap >= chunk_size:
        raise ConfigError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    return Settings(
        gemini_api_key=_require("GEMINI_API_KEY"),
        gemini_model=_optional("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        embedding_model=_optional("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        hf_token=_optional("HF_TOKEN") or None,
        chroma_collection=_optional("CHROMA_COLLECTION", DEFAULT_CHROMA_COLLECTION),
        data_dir=_path("DATA_DIR", PROJECT_ROOT / "data"),
        chroma_dir=_path("CHROMA_DIR", PROJECT_ROOT / "vectorstore"),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        similarity_top_k=_positive_int("SIMILARITY_TOP_K", DEFAULT_SIMILARITY_TOP_K),
    )
