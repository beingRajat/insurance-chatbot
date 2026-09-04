"""Query embedding.

The Qdrant collections were built by an existing pipeline with 1536-dimensional
vectors. Both `text-embedding-3-small` and the older `text-embedding-ada-002`
produce 1536 dims, so matching dimensions does NOT prove the query model
matches the index model -- and a mismatch degrades retrieval silently rather
than raising. `verify_against_stored()` settles it empirically.
"""
from __future__ import annotations

import logging
import math

from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import Settings

log = logging.getLogger(__name__)


class EmbeddingUnavailable(RuntimeError):
    """No embedding credential configured."""


class Embedder:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncOpenAI | None = None
        if settings.openai_api_key:
            self._client = AsyncOpenAI(
                api_key=settings.openai_api_key.get_secret_value(),
                timeout=settings.embedding_timeout_s,
                max_retries=0,  # tenacity owns retries so backoff is uniform
            )

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def model(self) -> str:
        return self._settings.embedding_model

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=12),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
        reraise=True,
    )
    async def embed(self, text: str) -> list[float]:
        if self._client is None:
            raise EmbeddingUnavailable(
                "OPENAI_API_KEY is not set, so queries cannot be embedded. "
                "Set it, or use the /plan endpoint which needs no embedding."
            )
        text = (text or "").strip()
        if not text:
            raise ValueError("Cannot embed empty text")

        resp = await self._client.embeddings.create(
            model=self._settings.embedding_model,
            input=text,
        )
        vec = resp.data[0].embedding
        if len(vec) != self._settings.embedding_dim:
            raise RuntimeError(
                f"Embedding model {self._settings.embedding_model} returned "
                f"{len(vec)} dims but the collection expects "
                f"{self._settings.embedding_dim}. Retrieval would fail."
            )
        return vec

    async def verify_against_stored(
        self, stored_text: str, stored_vector: list[float]
    ) -> float:
        """Re-embed text taken from a stored point and compare to its vector.

        Cosine near 1.0 means the configured model is the one that built the
        index. Anything materially below that means a different model was used
        and retrieval is running degraded.
        """
        fresh = await self.embed(stored_text)
        return _cosine(fresh, stored_vector)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
