"""Runtime configuration. Everything comes from the environment -- no secrets in code.

Copy .env.example to .env and fill it in, or export the variables directly.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- Qdrant -----------------------------------------------------------
    qdrant_url: str = Field(..., alias="QDRANT_URL")
    qdrant_api_key: SecretStr = Field(..., alias="QDRANT_API_KEY")
    qdrant_timeout_s: float = Field(30.0, alias="QDRANT_TIMEOUT_S")

    # Section-level collection used for retrieval and full-plan reads.
    policy_collection: str = Field("policy_units_v2", alias="POLICY_COLLECTION")
    # Flat one-row-per-plan feature table, used for structured filtering.
    feature_collection: str = Field("migration_payload", alias="FEATURE_COLLECTION")
    faq_collection: str = Field("faq_collection", alias="FAQ_COLLECTION")

    # ---- embeddings -------------------------------------------------------
    # MUST match the model the collections were built with. The stored vectors
    # are 1536-dimensional; `text-embedding-3-small` and the older
    # `text-embedding-ada-002` both produce 1536 dims, so dimension agreement
    # alone does NOT prove the model matches. Run
    #     python -m src.rag.cli verify-embeddings
    # to confirm against a stored vector before trusting retrieval quality.
    openai_api_key: SecretStr | None = Field(None, alias="OPENAI_API_KEY")
    embedding_model: str = Field("text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(1536, alias="EMBEDDING_DIM")
    embedding_timeout_s: float = Field(20.0, alias="EMBEDDING_TIMEOUT_S")

    # ---- answer model -----------------------------------------------------
    # Which provider synthesises answers. "anthropic" grounds answers with
    # native document citations (the API reports which document each span came
    # from). "openai" has no such mechanism, so it emits quotes that are then
    # verified server-side against the source -- weaker, and with no prompt
    # caching, but it runs on a single OpenAI key. See answer_base.py.
    answer_provider: str = Field("openai", alias="ANSWER_PROVIDER")

    anthropic_api_key: SecretStr | None = Field(None, alias="ANTHROPIC_API_KEY")
    answer_model: str = Field("claude-opus-5", alias="ANSWER_MODEL")
    # Used when answer_provider is "openai". Confirm the exact string is
    # available on your account before relying on it:
    #     python -m src.rag.cli verify-answer-model
    openai_answer_model: str = Field("gpt-5.4-mini", alias="OPENAI_ANSWER_MODEL")
    answer_effort: str = Field("high", alias="ANSWER_EFFORT")
    # Adaptive thinking is billed inside max_tokens, and at effort=high it
    # can consume thousands of tokens before the answer begins. 4096 risked
    # truncating mid-answer; 16000 leaves room to think while keeping a
    # non-streaming request under the SDK HTTP timeout.
    answer_max_tokens: int = Field(16000, alias="ANSWER_MAX_TOKENS")
    answer_timeout_s: float = Field(120.0, alias="ANSWER_TIMEOUT_S")

    # ---- retrieval tuning -------------------------------------------------
    # Sections pulled for a broad (cross-plan) search before reranking.
    search_top_k: int = Field(24, alias="SEARCH_TOP_K")
    # Distinct plans carried forward from a broad search.
    max_plans: int = Field(4, alias="MAX_PLANS")
    # Below this cosine score we treat retrieval as having found nothing.
    min_score: float = Field(0.25, alias="MIN_SCORE")
    # Hard ceiling on characters of context handed to the model.
    max_context_chars: int = Field(280_000, alias="MAX_CONTEXT_CHARS")

    # ---- policy -----------------------------------------------------------
    # `coverage_scope` has no payload index on the cluster, so Qdrant refuses
    # to filter on it (strict mode). We drop group products client-side after
    # retrieval instead. Set false only if a payload index is added later.
    exclude_group_client_side: bool = Field(True, alias="EXCLUDE_GROUP_CLIENT_SIDE")
    # Products that exist in the corpus but are no longer sold.
    exclude_withdrawn_products: bool = Field(True, alias="EXCLUDE_WITHDRAWN")
    # IRDAI-mandated standard products (Arogya Sanjeevani, Saral Suraksha
    # Bima) are still sold, so they are NOT withdrawn. Their wording is
    # identical across insurers, so excluding them is a product decision,
    # not a data fix. Default off -- hiding currently-sold plans should be
    # an explicit choice.
    exclude_mandated_standard: bool = Field(
        False, alias="EXCLUDE_MANDATED_STANDARD"
    )

    # ---- service ----------------------------------------------------------
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    request_timeout_s: float = Field(150.0, alias="REQUEST_TIMEOUT_S")


    @property
    def answer_provider_normalised(self) -> str:
        value = (self.answer_provider or "").strip().lower()
        if value not in {"anthropic", "openai"}:
            raise ValueError(
                f"ANSWER_PROVIDER must be 'anthropic' or 'openai', got "
                f"{self.answer_provider!r}"
            )
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


class ConfigError(RuntimeError):
    """Configuration is missing or invalid, with a human-readable explanation."""


def load_settings() -> Settings:
    """get_settings() with a readable error instead of a pydantic traceback.

    Missing configuration is the most common first-run failure, so it is worth
    saying which variables are missing and where they go rather than printing a
    validation dump.
    """
    try:
        return get_settings()
    except ValidationError as exc:
        missing = [
            str(err["loc"][0]) for err in exc.errors()
            if err.get("type") == "missing"
        ]
        other = [
            f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
            for err in exc.errors() if err.get("type") != "missing"
        ]
        env_path = Path.cwd() / ".env"
        lines = ["Configuration is incomplete."]
        if missing:
            lines.append("")
            lines.append("Missing required settings: " + ", ".join(missing))
        if other:
            lines.append("")
            lines.extend("  " + o for o in other)
        lines += [
            "",
            f"{'Edit' if env_path.exists() else 'Create'} {env_path} — start from"
            " the template:",
            "",
            "    cp .env.example .env      # PowerShell: copy .env.example .env",
            "",
            "At minimum QDRANT_URL and QDRANT_API_KEY must be set. Answering"
            " also needs OPENAI_API_KEY",
            "(or ANTHROPIC_API_KEY with ANSWER_PROVIDER=anthropic).",
        ]
        raise ConfigError("\n".join(lines)) from exc


def running_in_venv() -> bool:
    """True when the interpreter is the project virtualenv.

    Every dependency is installed into ./venv, so running the CLI with a bare
    system `python` fails on the first third-party import. Detecting it lets us
    say so instead of surfacing a ModuleNotFoundError.
    """
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)
