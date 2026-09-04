"""Plan-name resolution across collections with inconsistent naming.

DEFECT 6: the section collection and the feature collection use different
`plan_name` conventions.

    policy_units_v2   insurer_name="Aditya Birla Health Insurance"
                      plan_name   ="Activ One Max+"
    migration_payload plan_name   ="Aditya Birla Activ One Max+"

`plan_name` is indexed on both, and the cluster's strict mode forbids
unindexed filtering, so there is no server-side fuzzy match. Joining the two
collections on the raw string matches only 37 of 127 plans -- silently.

This module folds both name lists to a comparable key and resolves a caller's
plan name to the exact stored string in each collection. It also gives users
tolerant lookup: "activ one max" finds "Activ One Max+".

Folding is deliberately conservative. An earlier version stripped insurer brand
tokens from the plan name and keyed on the fold alone, which collapsed seven
distinct plans ("Acko Arogya Sanjeevani Policy", "National Arogya Sanjeevani
Policy", ...) onto one key and made six of them unreachable. Two more folded to
an empty string and vanished entirely. The key is therefore
(canonical insurer, folded plan) and the fold never returns empty.
"""
from __future__ import annotations

import asyncio
import difflib
import logging
import re
from dataclasses import dataclass, field

from . import canonical

log = logging.getLogger(__name__)

# Generic insurance vocabulary that carries no distinguishing information.
# Applied only when doing so leaves something behind.
_NOISE = re.compile(
    r"\b(policy|policies|plan|plans|insurance|assurance|company|limited|ltd|"
    r"co|the|cover|scheme|product)\b"
)


def fold_plan(name: str | None, *, strip_brand: bool = True) -> str:
    """Reduce a plan name to a comparable key.

    "+" and "plus" are preserved as a distinct token: "Activ One Max" and
    "Activ One Max+" are different products, and dropping the "+" merged them.

    Insurer brand tokens are stripped so the feature collection's
    "Aditya Birla Activ One Max+" folds onto the section collection's
    "Activ One Max+". Brand stripping is skipped when it would leave nothing --
    "National Health Policy" is entirely brand and noise words, and folding it
    to "" made it unreachable.
    """
    raw = (name or "")
    # Normalise the plus marker before punctuation is discarded.
    s = re.sub(r"\+", " plus ", raw)
    s = canonical._fold(s)
    if not s:
        return ""

    stages = []
    if strip_brand:
        brandless = s
        for token in _BRAND_TOKENS:
            brandless = re.sub(rf"\b{re.escape(token)}\b", " ", brandless)
        stages.append(re.sub(r"\s+", " ", brandless).strip())
    stages.append(s)

    for candidate in stages:
        denoised = re.sub(r"\s+", " ", _NOISE.sub(" ", candidate)).strip()
        if denoised:
            return denoised
        if candidate:
            return candidate
    return s


@dataclass(slots=True)
class ResolvedPlan:
    insurer: str
    display_name: str
    section_name: str | None  # exact plan_name in the section collection
    feature_name: str | None  # exact plan_name in the feature collection
    confidence: str           # "exact" | "folded" | "fuzzy"
    ambiguous_with: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.section_name is not None


class PlanResolver:
    """Built once per process; the underlying name lists are small and static."""

    def __init__(self) -> None:
        # (insurer, fold) -> raw section plan_name
        self._sections: dict[tuple[str, str], str] = {}
        # fold -> {insurer: raw feature plan_name}; the feature collection has
        # no separate insurer field, so its insurer is inferred from the name.
        self._features: dict[str, dict[str, str]] = {}
        self._features_any: dict[str, str] = {}
        self._raw_sections: dict[str, tuple[str, str]] = {}
        self._loaded = False
        self._lock = asyncio.Lock()

    async def load(self, store) -> None:
        """Populate from both collections. Concurrency-safe and idempotent."""
        if self._loaded:
            return
        async with self._lock:
            if self._loaded:  # another task won the race
                return
            for insurer, plan in await store.list_plan_names_raw():
                key = (insurer, fold_plan(plan))
                if key[1]:
                    self._sections.setdefault(key, plan)
                self._raw_sections[plan.strip().lower()] = (insurer, plan)

            for plan in await store.list_feature_plan_names_raw():
                fold = fold_plan(plan)
                if not fold:
                    continue
                # The feature collection bakes the insurer into the name, so
                # recover it from the un-stripped string.
                insurer = canonical.canonical_insurer(plan)
                self._features.setdefault(fold, {}).setdefault(insurer, plan)
                self._features_any.setdefault(fold, plan)

            self._loaded = True
            r = self.join_report
            log.info(
                "plan resolver: %d section plans, %d feature folds, %d joinable "
                "(raw-string join: %d, recovered by folding: %d)",
                r["section_plans"], r["feature_folds"], r["joinable_after_folding"],
                r["joinable_on_raw_string"], r["recovered_by_folding"],
            )

    def _feature_for(self, insurer: str, fold: str) -> str | None:
        by_insurer = self._features.get(fold) or {}
        return by_insurer.get(insurer) or self._features_any.get(fold)

    def resolve(self, name: str, insurer: str | None = None) -> ResolvedPlan | None:
        raw = (name or "").strip()
        if not raw:
            return None
        want = canonical.canonical_insurer(insurer) if insurer else None

        # 1. exact stored string in the section collection
        hit = self._raw_sections.get(raw.lower())
        if hit and (want is None or hit[0] == want):
            ins, stored = hit
            fold = fold_plan(stored)
            return ResolvedPlan(ins, stored, stored,
                                self._feature_for(ins, fold), "exact")

        # 2. folded match, scoped by insurer when one was given
        fold = fold_plan(raw)
        matches = [(i, f) for (i, f) in self._sections if f == fold]
        if want:
            matches = [m for m in matches if m[0] == want] or matches
        if matches:
            ins, f = matches[0]
            others = [f"{i} — {self._sections[(i, f)]}" for i, f in matches[1:]]
            return ResolvedPlan(ins, self._sections[(ins, f)],
                                self._sections[(ins, f)],
                                self._feature_for(ins, f), "folded", others)

        # 3. fuzzy over folds, tolerating typos and partial names
        all_folds = sorted({f for _, f in self._sections})
        close = difflib.get_close_matches(fold, all_folds, n=1, cutoff=0.82)
        if not close:
            partial = [f for f in all_folds if fold and fold in f]
            close = partial[:1] if len(partial) == 1 else []
        if close:
            cand = [(i, f) for (i, f) in self._sections if f == close[0]]
            if want:
                cand = [c for c in cand if c[0] == want] or cand
            ins, f = cand[0]
            others = [f"{i} — {self._sections[(i, f)]}" for i, f in cand[1:]]
            return ResolvedPlan(ins, self._sections[(ins, f)],
                                self._sections[(ins, f)],
                                self._feature_for(ins, f), "fuzzy", others)
        return None

    def suggestions(self, name: str, n: int = 5) -> list[str]:
        fold = fold_plan(name)
        all_folds = sorted({f for _, f in self._sections})
        keys = difflib.get_close_matches(fold, all_folds, n=n, cutoff=0.4)
        out = [self._sections[(i, f)] for (i, f) in self._sections if f in keys]
        if not out:
            out = list(self._sections.values())[:n]
        return out[:n]

    @property
    def join_report(self) -> dict:
        folds = {f for _, f in self._sections}
        joinable = {f for f in folds if f in self._features}
        raw_values = set(self._features_any.values())
        raw_join = {k for k, v in self._sections.items() if v in raw_values}
        return {
            "section_plans": len(self._sections),
            "feature_folds": len(self._features),
            "joinable_after_folding": len(joinable),
            "joinable_on_raw_string": len(raw_join),
            "recovered_by_folding": len(joinable) - len(raw_join),
            "unreachable_empty_fold": sum(
                1 for v in self._raw_sections.values() if not fold_plan(v[1])
            ),
        }


# Brand tokens that appear as an insurer prefix on feature rows. Derived from
# the canonical insurer map so the two stay in step. Defined after fold_plan so
# the module reads top-down, but evaluated at import time.
_BRAND_TOKENS = sorted(
    {t for name in set(canonical.canonical_names())
     for t in canonical._fold(name).split() if len(t) > 2},
    key=len,
    reverse=True,
)
