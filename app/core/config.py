from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true/false, yes/no, on/off, or 1/0.")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def _project_path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default)).expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    chroma_path: Path
    chroma_space: str
    embed_model: str
    chunk_size: int
    chunk_overlap: int
    max_upload_mb: int
    rerank_enabled: bool
    reranker_model: str
    ollama_host: str
    llm_provider: str
    llm_model: str
    api_base_url: str
    api_key: str

    @classmethod
    def from_env(cls) -> "Settings":
        chroma_space = os.getenv("CHROMA_SPACE", "cosine").strip().lower()
        if chroma_space != "cosine":
            raise ValueError("CHROMA_SPACE must be 'cosine' for this project configuration.")
        chunk_size = _env_int("CHUNK_SIZE", 200)
        chunk_overlap = _env_int("CHUNK_OVERLAP", 40)
        max_upload_mb = _env_int("MAX_UPLOAD_MB", 10)
        if not 1 <= chunk_size <= 255:
            raise ValueError("CHUNK_SIZE must be between 1 and 255 tokens.")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE.")
        if max_upload_mb < 1:
            raise ValueError("MAX_UPLOAD_MB must be at least 1.")
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip().rstrip("/")
        parsed_host = urlparse(ollama_host)
        if parsed_host.scheme not in {"http", "https"} or not parsed_host.netloc:
            raise ValueError("OLLAMA_HOST must be a valid HTTP or HTTPS URL.")
        llm_provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
        if llm_provider not in {"ollama", "openai_compatible"}:
            raise ValueError("LLM_PROVIDER must be 'ollama' or 'openai_compatible'.")
        return cls(
            project_root=PROJECT_ROOT,
            chroma_path=_project_path("CHROMA_PATH", "chroma_db"),
            chroma_space=chroma_space,
            embed_model=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip(),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_upload_mb=max_upload_mb,
            rerank_enabled=_env_bool("RERANK_ENABLED", False),
            reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2").strip(),
            ollama_host=ollama_host,
            llm_provider=llm_provider,
            llm_model=os.getenv("LLM_MODEL", "qwen2.5:7b").strip(),
            api_base_url=os.getenv("API_BASE_URL", "").strip(),
            api_key=os.getenv("API_KEY", ""),
        )


settings = Settings.from_env()
