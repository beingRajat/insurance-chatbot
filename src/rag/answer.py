"""Anthropic answer backend, grounded by native document citations.

Retrieved sections are passed as `document` content blocks with citations
enabled, so the response comes back segmented and each cited span reports which
document it came from. Every sentence is therefore attributable to a named
section of a named plan, and unattributable claims are visible because they
carry no citation at all.

The corpus has no page field, but the ingest pipeline left "### PAGE n" markers
inside the section text for ~52% of sections, so `canonical.locate_page` lifts
a page number out of the text where one is recoverable. Citations read
"Plan — Exclusions, page 14" when the page is known and "Plan — Exclusions"
when it is not; a page is never guessed.

`output_config.format` is deliberately not used: structured output and
citations are mutually exclusive on the API and would return 400. When a
structured comparison table is needed, run a second call without citations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import anthropic
from anthropic import AsyncAnthropic

from . import canonical
from .answer_base import (
    SYSTEM_PROMPT,
    Answer,
    AnswerUnavailable,
    Citation,
)
from .config import Settings
from .store import Section

log = logging.getLogger(__name__)

class Answerer:
    """Anthropic backend. Grounding comes from native document citations, so
    the document-to-quote mapping is produced by the platform rather than
    asserted by the model -- see answer_base for how the OpenAI backend
    differs."""

    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._client: AsyncAnthropic | None = None
        if settings.anthropic_api_key:
            self._client = AsyncAnthropic(
                api_key=settings.anthropic_api_key.get_secret_value(),
                timeout=settings.answer_timeout_s,
                max_retries=2,
            )

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def provider(self) -> str:
        return "anthropic"

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()

    def _documents(
        self, sections: list[Section], cache_last: bool
    ) -> tuple[list[dict], list[Section]]:
        """Build citation-enabled document blocks, budget-capped.

        Returns the blocks *and* the sections they were built from, in the same
        order. The API reports citations by `document_index`, so the caller must
        resolve that index against exactly this list; resolving it against the
        original `sections` mis-attributes every citation once anything is
        skipped for budget.

        A section that does not fit whole is skipped rather than truncated --
        half a clause is worse than none, because the model cannot tell it is
        reading a fragment.
        """
        blocks: list[dict] = []
        used: list[Section] = []
        budget = self._s.max_context_chars
        skipped = 0
        for sec in sections:
            if not sec.text:
                continue
            if len(sec.text) > budget:
                skipped += 1
                continue
            body = sec.text
            budget -= len(body)
            used.append(sec)
            blocks.append({
                "type": "document",
                "title": sec.label,
                "source": {
                    "type": "content",
                    "content": [{"type": "text", "text": body}],
                },
                # All-or-none across the request; enabling per block is required.
                "citations": {"enabled": True},
            })
        if skipped:
            log.warning(
                "context budget %d chars exhausted; %d of %d sections omitted "
                "(raise MAX_CONTEXT_CHARS or lower MAX_PLANS)",
                self._s.max_context_chars, skipped, len(sections),
            )
        # Cache the document prefix so follow-up questions about the same plan
        # re-read it at a fraction of the input cost.
        if blocks and cache_last:
            blocks[-1]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
        return blocks, used

    async def answer(
        self,
        question: str,
        sections: list[Section],
        *,
        cache_documents: bool = False,
        extra_context: str | None = None,
    ) -> Answer:
        if self._client is None:
            raise AnswerUnavailable(
                "ANTHROPIC_API_KEY is not set, so answers cannot be generated. "
                "Retrieval-only endpoints still work."
            )
        documents, used_sections = self._documents(
            sections, cache_last=cache_documents
        )
        if not documents:
            return Answer(
                text="I have no policy text to work from, so I cannot answer this.",
                model=self._s.answer_model,
                stop_reason="no_context",
            )

        content: list[dict] = [*documents]
        if extra_context:
            content.append({
                "type": "text",
                "text": (
                    "Structured plan attributes extracted from the same source "
                    "documents. Values marked as not stated are genuinely absent "
                    "upstream -- do not read them as zero:\n\n" + extra_context
                ),
            })
        content.append({"type": "text", "text": f"Question: {question}"})

        try:
            resp = await self._client.beta.messages.create(
                model=self._s.answer_model,
                max_tokens=self._s.answer_max_tokens,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": content}],
                thinking={"type": "adaptive"},
                output_config={"effort": self._s.answer_effort},
                # Rescue a policy refusal on a same-call fallback model rather
                # than returning nothing to the caller.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.BadRequestError:
            # Most likely cause: this deployment does not have the fallback beta
            # enabled. Retry once on the plain path so a beta gate cannot take
            # the service down.
            log.warning("beta call rejected; retrying without server-side fallbacks")
            resp = await self._client.messages.create(
                model=self._s.answer_model,
                max_tokens=self._s.answer_max_tokens,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": content}],
                thinking={"type": "adaptive"},
                output_config={"effort": self._s.answer_effort},
            )

        return _parse(resp, len(documents), used_sections)


def _parse(resp, documents_supplied: int, sections: list[Section]) -> Answer:
    parts: list[str] = []
    citations: list[Citation] = []
    # `sections` here is exactly the list _documents() built blocks from, so
    # the API's document_index lines up one-to-one.
    sourced = sections

    for block in resp.content:
        if getattr(block, "type", None) != "text":
            continue
        parts.append(block.text)
        for c in getattr(block, "citations", None) or []:
            quoted = getattr(c, "cited_text", "") or ""
            doc_i = getattr(c, "document_index", -1)
            doc_i = -1 if doc_i is None else doc_i
            page = None
            if 0 <= doc_i < len(sourced):
                page = canonical.locate_page(sourced[doc_i].text, quoted)
            citations.append(Citation(
                text=quoted,
                document_title=getattr(c, "document_title", "") or "",
                document_index=doc_i,
                page=page,
            ))

    usage = resp.usage
    stop_reason = getattr(resp, "stop_reason", "") or ""
    refusal = None
    if stop_reason == "refusal":
        details = getattr(resp, "stop_details", None)
        refusal = getattr(details, "explanation", None) or "declined by safety policy"

    return Answer(
        text="".join(parts).strip(),
        citations=citations,
        model=getattr(resp, "model", ""),
        provider="anthropic",
        stop_reason=stop_reason,
        refusal=refusal,
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        documents_supplied=documents_supplied,
    )
