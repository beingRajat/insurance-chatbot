"""OpenAI answer backend, with server-side quote verification.

Use this when only an OpenAI key is available. It is not a drop-in equal of the
Anthropic backend: OpenAI has no native document-citation mechanism, so
grounding is established differently and slightly more weakly.

The model is required (via strict structured output) to return each claim with
the id of the source it came from and a verbatim quote. `verify_quotes` then
checks that the quote actually occurs in that source. Quotes that fail are
dropped and counted in `Answer.unverified_quotes`, so a model that invents a
citation is visible rather than convincing.

Trade-offs against the Anthropic backend, all of which are consequences of the
API rather than choices:

* Grounding is verified after the fact, not produced by the platform.
* Structured output means one JSON response, so citations and a machine-readable
  answer coexist -- which the Anthropic path cannot do in a single call.
* No prompt caching, so repeated questions about the same plan pay full input
  price every time. On a ~30K-token plan that is the main cost difference.
"""
from __future__ import annotations

import json
import logging

from openai import (
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from . import canonical
from .answer_base import (
    SYSTEM_PROMPT,
    Answer,
    AnswerUnavailable,
    Citation,
    verify_quotes,
)
from .config import Settings
from .store import Section

log = logging.getLogger(__name__)

# Strict JSON schema: every property required, no extras. Anything less and the
# model omits `claims` on hard questions, which silently loses all grounding.
RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answerable", "answer", "claims"],
    "properties": {
        "answerable": {
            "type": "boolean",
            "description": (
                "False when the supplied documents do not contain the answer. "
                "Prefer false over a partially-supported answer."
            ),
        },
        "answer": {
            "type": "string",
            "description": (
                "The answer in plain prose. When answerable is false, state "
                "plainly that the supplied policy sections do not cover it."
            ),
        },
        "claims": {
            "type": "array",
            "description": (
                "One entry per factual claim in the answer, each backed by a "
                "verbatim quote. Quotes are checked against the source and "
                "discarded if they do not match, so paraphrasing loses the "
                "citation."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["source_id", "quote"],
                "properties": {
                    "source_id": {
                        "type": "integer",
                        "description": "The [SOURCE n] number the claim came from.",
                    },
                    "quote": {
                        "type": "string",
                        "description": (
                            "Text copied exactly from that source, word for "
                            "word, no ellipses or edits."
                        ),
                    },
                },
            },
        },
    },
}

_EXTRA_RULES = """
Return JSON matching the required schema.

Every factual sentence in `answer` must be supported by an entry in `claims`
whose `quote` is copied EXACTLY from the source you cite -- word for word, with
no ellipses, no reformatting and no paraphrasing. Quotes are automatically
checked against the source text and silently discarded when they do not match,
so an approximate quote is the same as no citation at all.

Cite sources by the [SOURCE n] number shown above each document.
"""


class OpenAIAnswerer:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._client: AsyncOpenAI | None = None
        if settings.openai_api_key:
            self._client = AsyncOpenAI(
                api_key=settings.openai_api_key.get_secret_value(),
                timeout=settings.answer_timeout_s,
                max_retries=0,  # tenacity owns retries so backoff is uniform
            )

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def provider(self) -> str:
        return "openai"

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()

    def _documents(self, sections: list[Section]) -> tuple[str, list[Section]]:
        """Render sections as numbered sources, budget-capped.

        Returns the prompt text and the sections actually included, in order --
        `source_id` indexes into the returned list, so a section skipped for
        budget must not shift the numbering of the rest.
        """
        parts: list[str] = []
        used: list[Section] = []
        budget = self._s.max_context_chars
        skipped = 0
        for sec in sections:
            if not sec.text:
                continue
            if len(sec.text) > budget:
                skipped += 1
                continue
            budget -= len(sec.text)
            parts.append(f"[SOURCE {len(used)}] {sec.label}\n{sec.text}")
            used.append(sec)
        if skipped:
            log.warning(
                "context budget %d chars exhausted; %d of %d sections omitted "
                "(raise MAX_CONTEXT_CHARS or lower MAX_PLANS)",
                self._s.max_context_chars, skipped, len(sections),
            )
        return "\n\n".join(parts), used

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
        reraise=True,
    )
    async def _call(self, system: str, user: str):
        assert self._client is not None
        return await self._client.chat.completions.create(
            model=self._s.openai_answer_model,
            max_completion_tokens=self._s.answer_max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "grounded_policy_answer",
                    "strict": True,
                    "schema": RESPONSE_SCHEMA,
                },
            },
        )

    async def answer(
        self,
        question: str,
        sections: list[Section],
        *,
        cache_documents: bool = False,  # accepted for interface parity; unused
        extra_context: str | None = None,
    ) -> Answer:
        if self._client is None:
            raise AnswerUnavailable(
                "OPENAI_API_KEY is not set, so answers cannot be generated."
            )

        docs_text, used = self._documents(sections)
        if not used:
            return Answer(
                text="I have no policy text to work from, so I cannot answer this.",
                model=self._s.openai_answer_model,
                provider="openai",
                stop_reason="no_context",
            )

        blocks = [docs_text]
        if extra_context:
            blocks.append(
                "Structured plan attributes extracted from the same source "
                "documents. Values marked as not stated are genuinely absent "
                "upstream -- do not read them as zero:\n\n" + extra_context
            )
        blocks.append(f"Question: {question}")

        resp = await self._call(SYSTEM_PROMPT + _EXTRA_RULES, "\n\n".join(blocks))
        choice = resp.choices[0]

        if getattr(choice.message, "refusal", None):
            return Answer(
                text="", model=resp.model, provider="openai",
                stop_reason="refusal", refusal=choice.message.refusal,
                documents_supplied=len(used),
            )

        try:
            payload = json.loads(choice.message.content or "{}")
        except json.JSONDecodeError as exc:
            log.error("model returned unparseable JSON: %s", exc)
            return Answer(
                text="", model=resp.model, provider="openai",
                stop_reason="bad_json",
                refusal="the model returned malformed JSON",
                documents_supplied=len(used),
            )

        raw_claims = payload.get("claims") or []
        pairs = [
            (int(c.get("source_id", -1)), str(c.get("quote", "")))
            for c in raw_claims
            if isinstance(c, dict)
        ]
        # The model asserting a source is not evidence. Check every quote.
        good, rejected = verify_quotes(pairs, used)
        if rejected:
            log.warning(
                "%d of %d quotes did not appear in the source cited and were "
                "discarded", rejected, len(pairs),
            )

        citations = [
            Citation(
                text=quote,
                document_title=used[idx].label,
                document_index=idx,
                page=canonical.locate_page(used[idx].text, quote),
                verified=True,
            )
            for idx, quote in good
        ]

        text = str(payload.get("answer") or "").strip()
        if payload.get("answerable") is False and not text:
            text = "The supplied policy sections do not state this."

        usage = resp.usage
        return Answer(
            text=text,
            citations=citations,
            model=resp.model,
            provider="openai",
            stop_reason=choice.finish_reason or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            documents_supplied=len(used),
            unverified_quotes=rejected,
        )
