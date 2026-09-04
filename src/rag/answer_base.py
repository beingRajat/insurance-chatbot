"""Provider-agnostic answer types, prompt, and grounding verification.

Two answer backends implement `SupportsAnswer`:

* `answer.Answerer`         -- Anthropic, using native document citations
* `answer_openai.OpenAIAnswerer` -- OpenAI, using structured output plus
                              server-side quote verification

They are not equivalent in how grounding is established, and the difference
matters enough to be explicit about:

Anthropic computes citations itself. Documents are passed as `document`
content blocks with `citations: {enabled: true}` and the API returns, for each
cited span, the exact source text and which document it came from. The mapping
is produced by the platform, not asserted by the model.

OpenAI has no such mechanism, so the model is asked to emit each claim with a
`source_id` and a verbatim `quote`, and `verify_quotes()` then checks that the
quote really occurs in the document it names. Quotes that fail are discarded
rather than shown, and the answer is marked ungrounded if none survive. That
makes the model's citation a *claim to be checked* instead of a fact to be
trusted -- which is the right posture, but it is verification after the fact
rather than provenance by construction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .store import Section

SYSTEM_PROMPT = """\
You answer questions about Indian health-insurance policy wordings, using only \
the policy documents supplied in this request.

Rules you must follow:

1. Ground every factual claim in the supplied documents. If the documents do \
not contain the answer, say so plainly -- "The supplied policy sections do not \
state this" -- and stop. Never fill a gap from general knowledge about \
insurance, and never infer a number that is not written down.

2. Insurance answers are usually composed from several clauses. Before \
answering a coverage question, check for and report all of: the coverage \
grant, any waiting period (initial, pre-existing disease, and specific \
disease), sub-limits or capping, co-payment, room-rent limits and any \
proportionate deduction, and the exclusions list. An answer that gives the \
coverage grant while omitting an applicable waiting period, sub-limit or \
co-payment is wrong, even if every word of it is accurate.

3. If a value is described as not stated in the source data, report it as not \
stated. Do not treat it as zero, unlimited, or absent.

4. Answer questions about what a document says. Do not recommend which policy \
the reader should buy, and do not assess their personal situation -- that is \
regulated advice. If asked, explain that you can compare what the wordings say \
and leave the choice to them.

5. Name the plan and insurer you are answering about. When comparing plans, \
attribute every statement to a specific plan.

6. Be concise and concrete. Lead with the direct answer, then the conditions \
that qualify it. Use plain language; expand jargon such as PED on first use.
"""


@dataclass(slots=True)
class Citation:
    text: str
    document_title: str
    document_index: int
    # Recovered from "### PAGE n" markers left in the section text by the
    # ingest pipeline (present for ~52% of sections). None when the section
    # carries no markers or the quote could not be located -- never guessed.
    page: int | None = None
    # True when the quote was checked against the cited document rather than
    # taken on trust. Native-citation backends are verified by construction.
    verified: bool = True

    @property
    def source_label(self) -> str:
        return (f"{self.document_title}, page {self.page}"
                if self.page is not None else self.document_title)


@dataclass(slots=True)
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    stop_reason: str = ""
    refusal: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    documents_supplied: int = 0
    # Quotes the model attributed to a document that do not appear in it.
    # Non-zero means the model fabricated or paraphrased a citation.
    unverified_quotes: int = 0

    @property
    def is_grounded(self) -> bool:
        return bool(self.citations)


class AnswerUnavailable(RuntimeError):
    """No credential configured for the selected answer provider."""


@runtime_checkable
class SupportsAnswer(Protocol):
    """Interface both answer backends satisfy."""

    @property
    def available(self) -> bool: ...

    @property
    def provider(self) -> str: ...

    async def answer(
        self,
        question: str,
        sections: list[Section],
        *,
        cache_documents: bool = False,
        extra_context: str | None = None,
    ) -> Answer: ...

    async def aclose(self) -> None: ...


def _normalise(s: str) -> str:
    """Collapse whitespace and case so quote matching survives reformatting."""
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def verify_quotes(
    quotes: list[tuple[int, str]], sections: list[Section]
) -> tuple[list[tuple[int, str]], int]:
    """Keep only quotes that genuinely occur in the document they cite.

    Returns (verified, rejected_count). A model asserting a source is not
    evidence; this makes it checkable. Matching is whitespace- and
    case-insensitive because models reflow quoted text, but it is otherwise a
    substring test -- a paraphrase does not pass.
    """
    flat = [_normalise(s.text) for s in sections]
    verified: list[tuple[int, str]] = []
    rejected = 0
    for idx, quote in quotes:
        probe = _normalise(quote)
        if not probe or not (0 <= idx < len(flat)):
            rejected += 1
            continue
        if probe in flat[idx]:
            verified.append((idx, quote))
        else:
            rejected += 1
    return verified, rejected
