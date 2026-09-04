"""Retrieval orchestration.

Two query paths, because insurance questions come in two shapes:

* `ask_plan` -- a question about one named plan. All nine of that plan's
  sections are read in full. This is the reliable path: an insurance answer is
  composed from clauses scattered across the document (coverage grant, waiting
  periods, sub-limits, co-pay, exclusions), so top-k retrieval within a single
  plan systematically drops some of them. Reading the plan whole removes that
  failure mode, and it is affordable -- nine sections is a small context.

* `ask_broad` -- a question across plans. Vector search proposes candidate
  plans, then each candidate is read in full before answering, so retrieval
  narrows the field but never decides the answer on a partial view.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from . import canonical
from .answer import Answerer
from .answer_base import Answer, SupportsAnswer
from .answer_openai import OpenAIAnswerer
from .config import Settings
from .embeddings import Embedder
from .resolver import PlanResolver, ResolvedPlan
from .store import PolicyStore, Section

log = logging.getLogger(__name__)


def build_answerer(settings) -> SupportsAnswer:
    """Pick the answer backend from ANSWER_PROVIDER.

    Both satisfy the same interface but do not ground answers the same way --
    Anthropic citations are computed by the platform, OpenAI quotes are
    verified after the fact. answer_base.py documents the difference.
    """
    provider = settings.answer_provider_normalised
    backend: SupportsAnswer = (
        OpenAIAnswerer(settings) if provider == "openai" else Answerer(settings)
    )
    log.info("answer backend: %s (%s)", provider,
             settings.openai_answer_model if provider == "openai"
             else settings.answer_model)
    return backend

# Feature flags worth surfacing alongside a plan answer, and the wording used
# for them. Keeps the -1 sentinel from ever being shown as a real number.
FEATURE_SUMMARY: dict[str, tuple[str, str]] = {
    "initial_waiting_period_days": ("Initial waiting period", "days"),
    "ped_waiting_period_months": ("Pre-existing disease waiting period", "months"),
    "specific_disease_waiting_period_months": ("Specific-disease waiting period", "months"),
    "maternity_waiting_period_months": ("Maternity waiting period", "months"),
    "copay_percentage": ("Co-payment", "%"),
    "min_sum_insured": ("Minimum sum insured", ""),
    "max_sum_insured": ("Maximum sum insured", ""),
    "pre_hospitalization_days": ("Pre-hospitalisation cover", "days"),
    "post_hospitalization_days": ("Post-hospitalisation cover", "days"),
    "grace_period_days": ("Grace period", "days"),
    "free_look_period_days": ("Free-look period", "days"),
}


@dataclass(slots=True)
class RetrievalTrace:
    """Everything needed to audit or debug one answer."""

    mode: str
    question: str
    plans_considered: list[str] = field(default_factory=list)
    sections_used: list[str] = field(default_factory=list)
    top_score: float = 0.0
    dropped_group: int = 0
    dropped_withdrawn: int = 0
    dropped_low_score: int = 0
    dropped_standard: int = 0
    # Which source PDF was read, and what else carried the same plan name.
    # DEFECT 7: plan_name alone is not a document key, so the choice is
    # recorded rather than made invisibly.
    document_read: str = ""
    alternative_documents: list[str] = field(default_factory=list)
    retrieval_ms: int = 0
    answer_ms: int = 0


@dataclass(slots=True)
class Result:
    answer: Answer | None
    trace: RetrievalTrace
    sections: list[Section]
    message: str | None = None


class RagService:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self.store = PolicyStore(settings)
        self.embedder = Embedder(settings)
        self.answerer: SupportsAnswer = build_answerer(settings)
        self.resolver = PlanResolver()

    async def warmup(self) -> dict:
        """Load the plan-name resolver. Cheap, and it surfaces the cross-
        collection naming mismatch as a number rather than as silent misses."""
        await self.resolver.load(self.store)
        return self.resolver.join_report

    async def aclose(self) -> None:
        await self.store.aclose()
        await self.embedder.aclose()
        await self.answerer.aclose()

    # ---- filtering -------------------------------------------------------
    def _apply_policy(
        self, sections: list[Section], trace: RetrievalTrace
    ) -> list[Section]:
        """Client-side filters standing in for absent payload indexes."""
        kept: list[Section] = []
        for sec in sections:
            payload = {
                "coverage_scope": sec.coverage_scope,
                "plan_name": sec.plan_name,
            }
            if self._s.exclude_group_client_side and canonical.is_group_product(payload):
                trace.dropped_group += 1
                continue
            if self._s.exclude_withdrawn_products and canonical.is_withdrawn_product(payload):
                trace.dropped_withdrawn += 1
                continue
            if (self._s.exclude_mandated_standard
                    and canonical.is_mandated_standard_product(payload)):
                trace.dropped_standard += 1
                continue
            kept.append(sec)
        return kept

    async def _feature_context(self, resolved: ResolvedPlan) -> str | None:
        # Must query with the feature collection's own spelling of the name.
        if not resolved.feature_name:
            return None
        features = await self.store.plan_features(resolved.feature_name)
        if not features:
            return None
        lines = [f"Plan: {resolved.display_name}"]
        for key, (label, unit) in FEATURE_SUMMARY.items():
            if key in features:
                lines.append(f"  {label}: {canonical.describe_number(features[key], unit)}")
        flags = sorted(
            k.removeprefix("has_").replace("_", " ")
            for k, v in features.items()
            if k.startswith("has_") and v is True
        )
        if flags:
            lines.append("  Present per extracted attributes: " + ", ".join(flags))
        return "\n".join(lines)

    # ---- path 1: one named plan -----------------------------------------
    async def ask_plan(
        self, question: str, plan_name: str, *, insurer: str | None = None
    ) -> Result:
        trace = RetrievalTrace(mode="plan", question=question)
        t0 = time.perf_counter()

        await self.resolver.load(self.store)
        resolved = self.resolver.resolve(plan_name)
        if resolved is None or not resolved.usable:
            trace.retrieval_ms = int((time.perf_counter() - t0) * 1000)
            hints = ", ".join(self.resolver.suggestions(plan_name))
            return Result(
                answer=None, trace=trace, sections=[],
                message=f"No plan matching {plan_name!r}. Closest names: {hints}",
            )

        docs = await self.store.fetch_plan_documents(resolved.section_name)
        if insurer:
            want = canonical.canonical_insurer(insurer)
            docs = [d for d in docs if d.insurer == want] or docs

        if len({d.insurer for d in docs}) > 1:
            # Same plan name across insurers (mandated standard products).
            # Answering from an arbitrary one would be a silent wrong answer.
            trace.retrieval_ms = int((time.perf_counter() - t0) * 1000)
            names = sorted({d.insurer for d in docs})
            return Result(
                answer=None, trace=trace, sections=[],
                message=(
                    f"{resolved.display_name!r} is offered by more than one "
                    f"insurer ({', '.join(names)}). Re-ask with the insurer "
                    "specified so the answer comes from one policy."
                ),
            )

        chosen = docs[0] if docs else None
        sections = self._apply_policy(chosen.sections if chosen else [], trace)
        if chosen:
            trace.document_read = chosen.source_pdf
            trace.alternative_documents = [d.source_pdf for d in docs[1:]]
            if trace.alternative_documents:
                log.warning(
                    "plan %r has %d source documents; read %r, ignored %s",
                    resolved.display_name, len(docs), chosen.source_pdf,
                    trace.alternative_documents,
                )
        trace.retrieval_ms = int((time.perf_counter() - t0) * 1000)

        if not sections:
            return Result(
                answer=None, trace=trace, sections=[],
                message=(
                    f"{resolved.display_name!r} matched the corpus but has no "
                    "retail sections left after filtering — it is likely a "
                    "group or withdrawn product."
                ),
            )

        trace.plans_considered = [f"{resolved.insurer} — {resolved.display_name}"]
        trace.sections_used = [s.section_id for s in sections]

        extra = await self._feature_context(resolved)
        t1 = time.perf_counter()
        answer = await self.answerer.answer(
            question, sections, cache_documents=True, extra_context=extra
        )
        trace.answer_ms = int((time.perf_counter() - t1) * 1000)
        return Result(answer=answer, trace=trace, sections=sections)

    # ---- path 2: across plans -------------------------------------------
    async def ask_broad(
        self, question: str, *, insurers: list[str] | None = None
    ) -> Result:
        trace = RetrievalTrace(mode="broad", question=question)
        t0 = time.perf_counter()

        vector = await self.embedder.embed(question)
        hits = await self.store.search_sections(
            vector, top_k=self._s.search_top_k, insurers=insurers
        )
        if hits:
            trace.top_score = hits[0].score

        before = len(hits)
        hits = [h for h in hits if h.score >= self._s.min_score]
        trace.dropped_low_score = before - len(hits)
        hits = self._apply_policy(hits, trace)

        if not hits:
            trace.retrieval_ms = int((time.perf_counter() - t0) * 1000)
            return Result(
                answer=None, trace=trace, sections=[],
                message=(
                    "Nothing in the policy corpus matched that question closely "
                    "enough to answer from. Rather than guess, I am declining. "
                    "Try naming a specific plan, or rephrasing with policy terms."
                ),
            )

        # Rank plans by their best-matching section, then read those plans whole.
        best: dict[tuple[str, str], float] = {}
        for h in hits:
            key = h.plan_key
            best[key] = max(best.get(key, 0.0), h.score)
        ranked = sorted(best, key=lambda k: -best[k])[: self._s.max_plans]

        # Scope each candidate to the insurer whose section actually matched.
        # fetch_plan_sections alone picks the first document for a plan name,
        # and mandated products share names across insurers -- a hit on Star
        # Health's "Arogya Sanjeevani Policy" would otherwise be answered from
        # Aditya Birla's document.
        sections: list[Section] = []
        for insurer, plan_name in ranked:
            docs = await self.store.fetch_plan_documents(plan_name)
            scoped = [d for d in docs if d.insurer == insurer] or docs
            sections.extend(self._apply_policy(scoped[0].sections, trace))
            if len(scoped) > 1:
                trace.alternative_documents.extend(d.source_pdf for d in scoped[1:])

        trace.plans_considered = [f"{i} — {p}" for i, p in ranked]
        trace.sections_used = sorted({s.section_id for s in sections})
        trace.retrieval_ms = int((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        answer = await self.answerer.answer(question, sections, cache_documents=False)
        trace.answer_ms = int((time.perf_counter() - t1) * 1000)
        return Result(answer=answer, trace=trace, sections=sections)

    # ---- dry run: what would be sent, without calling the model ---------
    async def retrieve_only(
        self, question: str, *, plan_name: str | None = None
    ) -> tuple[list[Section], RetrievalTrace, str | None]:
        """Exercise the full retrieval path without spending an answer call.

        Useful for tuning MIN_SCORE / MAX_PLANS, for CI, and for running the
        pipeline before answer credentials are provisioned.
        """
        if plan_name:
            trace = RetrievalTrace(mode="plan-dry", question=question)
            t0 = time.perf_counter()
            await self.resolver.load(self.store)
            resolved = self.resolver.resolve(plan_name)
            if resolved is None or not resolved.usable:
                trace.retrieval_ms = int((time.perf_counter() - t0) * 1000)
                return [], trace, None
            sections = self._apply_policy(
                await self.store.fetch_plan_sections(resolved.section_name), trace
            )
            trace.retrieval_ms = int((time.perf_counter() - t0) * 1000)
            trace.plans_considered = [
                f"{resolved.insurer} — {resolved.display_name} "
                f"(match: {resolved.confidence})"
            ]
            trace.sections_used = [s.section_id for s in sections]
            return sections, trace, await self._feature_context(resolved)

        trace = RetrievalTrace(mode="broad-dry", question=question)
        t0 = time.perf_counter()
        vector = await self.embedder.embed(question)
        hits = await self.store.search_sections(vector, top_k=self._s.search_top_k)
        if hits:
            trace.top_score = hits[0].score
        before = len(hits)
        hits = [h for h in hits if h.score >= self._s.min_score]
        trace.dropped_low_score = before - len(hits)
        hits = self._apply_policy(hits, trace)
        trace.plans_considered = [f"{i} — {p}" for i, p in dict.fromkeys(
            h.plan_key for h in hits)][: self._s.max_plans]
        trace.sections_used = [h.section_id for h in hits]
        trace.retrieval_ms = int((time.perf_counter() - t0) * 1000)
        return hits, trace, None

    # ---- path 3: structured feature filter (no LLM) ---------------------
    async def plans_with_features(self, flags: list[str]) -> list[dict]:
        """Exact structured filter. Every has_*/cover_type_* flag is indexed,
        so this is server-side and exact -- the right tool for "which plans
        cover X", where similarity search would only approximate."""
        rows = await self.store.filter_plans_by_features(flags_true=flags)
        out = []
        for r in rows:
            out.append({
                "insurer": canonical.canonical_insurer(r.get("insurer_name")),
                "plan_name": canonical.clean_text(r.get("plan_name")),
                "coverage_scope": r.get("coverage_scope"),
                "attributes": {
                    label: canonical.describe_number(r.get(key), unit)
                    for key, (label, unit) in FEATURE_SUMMARY.items()
                    if key in r
                },
            })
        return out
