"""Qdrant access layer.

Two collections carry the load:

* the section collection (`policy_units_v2`) -- one point per (plan, section),
  nine sections per plan, used for semantic retrieval and for reading a plan in
  full;
* the feature collection (`migration_payload`) -- one point per plan with ~70
  indexed boolean and numeric fields, used for structured filtering.

Only `insurer_name`, `plan_name` and `section_id` are indexed on the section
collection, and the cluster runs strict mode (`unindexed_filtering_retrieve`
disabled), so a filter on anything else returns HTTP 400 rather than falling
back to a scan. Filters here stay inside that set; everything else is applied
client-side in `canonical.py`.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
)

from . import canonical
from .config import Settings

log = logging.getLogger(__name__)

# Section order as it appears in a policy document, so a full-plan read is
# assembled in a sensible reading order rather than in retrieval order.
SECTION_ORDER = [
    "plan_overview",
    "eligibility",
    "benefits",
    "optional_benefits",
    "exclusions",
    "terms_and_conditions",
    "claim_process",
    "value_added_services",
    "customer_support",
]
_SECTION_RANK = {s: i for i, s in enumerate(SECTION_ORDER)}


@dataclass(slots=True)
class Section:
    """One retrieved (plan, section) unit."""

    insurer: str
    plan_name: str
    section_id: str
    section_name: str
    text: str
    source_pdf: str
    coverage_scope: str
    score: float = 0.0
    structured: dict = field(default_factory=dict)

    @property
    def plan_key(self) -> tuple[str, str]:
        return (self.insurer, self.plan_name)

    @property
    def label(self) -> str:
        return f"{self.insurer} — {self.plan_name} — {self.section_name}"


# Filename signals that one document supersedes another for the same plan.
_REVISED_HINT = re.compile(r"\brevised\b|\bnew\b|\bupdated\b|\bv\.?\s*\d+\b", re.I)
_DUP_SUFFIX = re.compile(r"\(\d+\)\s*\.pdf$", re.I)


@dataclass(slots=True)
class PlanDocument:
    """One source PDF for one plan -- the real unit of policy identity."""

    insurer: str
    plan_name: str
    source_pdf: str
    sections: list[Section]

    @property
    def total_chars(self) -> int:
        return sum(len(s.text) for s in self.sections)

    @property
    def revision_rank(self) -> int:
        """Higher wins. Prefers an explicit revision, penalises "(1).pdf" dupes."""
        rank = 0
        if _REVISED_HINT.search(self.source_pdf):
            rank += 2
        years = re.findall(r"20(\d{2})", self.source_pdf)
        if years:
            rank += 1
        if _DUP_SUFFIX.search(self.source_pdf):
            rank -= 3
        return rank

    @property
    def label(self) -> str:
        return f"{self.insurer} — {self.plan_name} [{self.source_pdf}]"


def _to_section(payload: dict, score: float = 0.0) -> Section:
    return Section(
        insurer=canonical.canonical_insurer(payload.get("insurer_name")),
        plan_name=canonical.clean_text(payload.get("plan_name")) or "Unknown plan",
        section_id=(payload.get("section_id") or "").strip(),
        section_name=canonical.clean_text(payload.get("section_name")) or "Section",
        text=canonical.clean_text(payload.get("text")),
        source_pdf=(payload.get("source_pdf") or "").strip(),
        coverage_scope=(payload.get("coverage_scope") or "").strip(),
        score=score,
        structured=payload.get("section_json") or {},
    )


class PolicyStore:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        # check_compatibility is disabled deliberately: the pinned client may
        # sit a minor version ahead of the managed server, which is a warning
        # rather than an incompatibility for the calls used here.
        self._client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key.get_secret_value(),
            timeout=int(settings.qdrant_timeout_s),
            check_compatibility=False,
        )
        self._insurer_spellings: dict[str, list[str]] | None = None

    async def aclose(self) -> None:
        await self._client.close()

    # ---- health ----------------------------------------------------------
    async def health(self) -> dict:
        cols = await self._client.get_collections()
        names = {c.name for c in cols.collections}
        out: dict = {"collections": sorted(names), "checks": {}}
        for label, name in (
            ("sections", self._s.policy_collection),
            ("features", self._s.feature_collection),
        ):
            if name not in names:
                out["checks"][label] = {"collection": name, "status": "ABSENT"}
                continue
            info = await self._client.get_collection(name)
            out["checks"][label] = {
                "collection": name,
                "status": str(info.status),
                "points": info.points_count,
                "indexed_payload_fields": sorted((info.payload_schema or {}).keys()),
            }
        return out

    async def sample_point_with_vector(self) -> tuple[str, list[float]] | None:
        """A stored (text, vector) pair, for the embedding-model verifier."""
        points, _ = await self._client.scroll(
            collection_name=self._s.policy_collection,
            limit=1,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            return None
        payload = points[0].payload or {}
        vector = points[0].vector
        if not isinstance(vector, list):  # named-vector collections
            return None
        # Verify against the *raw* stored text, not the cleaned form -- the
        # index was built from the raw value.
        return payload.get("text") or "", vector

    async def load_insurer_spellings(self) -> dict[str, list[str]]:
        """canonical insurer -> every raw `insurer_name` value in the corpus.

        `insurer_name` is indexed, so an insurer filter must enumerate the exact
        stored strings. Guessing them from a canonical name does not work: the
        corpus holds "IFFCO Tokio" and "IFFCO-Tokio", neither of which a
        case/punctuation permutation of "IFFCO-Tokio General Insurance"
        reliably reproduces. There are only 16 distinct values, so they are read
        once and cached.
        """
        if self._insurer_spellings is not None:
            return self._insurer_spellings
        seen: set[str] = set()
        offset = None
        while True:
            points, offset = await self._client.scroll(
                collection_name=self._s.policy_collection,
                limit=256, offset=offset,
                with_payload=["insurer_name"], with_vectors=False,
            )
            for p in points:
                value = ((p.payload or {}).get("insurer_name") or "").strip()
                if value:
                    seen.add(value)
            if offset is None:
                break
        mapping: dict[str, list[str]] = {}
        for value in sorted(seen):
            mapping.setdefault(canonical.canonical_insurer(value), []).append(value)
        self._insurer_spellings = mapping
        log.info("insurer spellings: %d raw values -> %d canonical insurers",
                 len(seen), len(mapping))
        return mapping

    # ---- plan discovery --------------------------------------------------
    async def list_plans(self) -> list[tuple[str, str]]:
        """Every (insurer, plan) pair in the section collection."""
        plans: set[tuple[str, str]] = set()
        offset = None
        while True:
            points, offset = await self._client.scroll(
                collection_name=self._s.policy_collection,
                limit=256,
                offset=offset,
                with_payload=["insurer_name", "plan_name", "coverage_scope"],
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                if self._s.exclude_group_client_side and canonical.is_group_product(payload):
                    continue
                # Apply the same withdrawn-product rule the query paths use, so
                # /plans cannot advertise a plan that /ask then refuses to read.
                if self._s.exclude_withdrawn_products and canonical.is_withdrawn_product(payload):
                    continue
                plans.add(canonical.plan_key(payload))
            if offset is None:
                break
        return sorted(plans)

    async def list_plan_names_raw(self) -> list[tuple[str, str]]:
        """(canonical insurer, RAW stored plan_name) from the section collection.

        Raw names are required: `plan_name` is indexed, so filters must use the
        exact stored string.
        """
        out: list[tuple[str, str]] = []
        offset = None
        while True:
            points, offset = await self._client.scroll(
                collection_name=self._s.policy_collection,
                limit=256, offset=offset,
                with_payload=["insurer_name", "plan_name"], with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                plan = (payload.get("plan_name") or "").strip()
                if plan:
                    out.append(
                        (canonical.canonical_insurer(payload.get("insurer_name")), plan)
                    )
            if offset is None:
                break
        return out

    async def list_feature_plan_names_raw(self) -> list[str]:
        """RAW stored plan_name values from the feature collection."""
        out: list[str] = []
        offset = None
        while True:
            points, offset = await self._client.scroll(
                collection_name=self._s.feature_collection,
                limit=256, offset=offset,
                with_payload=["plan_name"], with_vectors=False,
            )
            for p in points:
                plan = ((p.payload or {}).get("plan_name") or "").strip()
                if plan:
                    out.append(plan)
            if offset is None:
                break
        return out

    # ---- retrieval -------------------------------------------------------
    async def search_sections(
        self,
        vector: list[float],
        *,
        top_k: int,
        insurers: list[str] | None = None,
        section_ids: list[str] | None = None,
    ) -> list[Section]:
        """Semantic search over sections.

        `insurers` are canonical names; every raw spelling that maps to them is
        enumerated, because filtering on one spelling silently drops the rest.
        """
        must: list[FieldCondition] = []
        if insurers:
            known = await self.load_insurer_spellings()
            spellings: list[str] = []
            for name in insurers:
                spellings.extend(known.get(canonical.canonical_insurer(name), []))
            if not spellings:
                log.warning("no stored insurer_name matches %s; ignoring the "
                            "insurer filter rather than returning nothing", insurers)
            else:
                must.append(
                    FieldCondition(key="insurer_name",
                                   match=MatchAny(any=sorted(set(spellings))))
                )
        if section_ids:
            must.append(FieldCondition(key="section_id", match=MatchAny(any=section_ids)))

        result = await self._client.query_points(
            collection_name=self._s.policy_collection,
            query=vector,
            limit=top_k,
            with_payload=True,
            query_filter=Filter(must=must) if must else None,
        )
        return [_to_section(p.payload or {}, p.score) for p in result.points]

    async def fetch_plan_documents(self, plan_name: str) -> list[PlanDocument]:
        """All distinct source documents carrying this plan name.

        DEFECT 7: `plan_name` alone does not identify a document.

        * Mandated standard products share a name across insurers --
          "Arogya Sanjeevani Policy" returns 27 sections spanning United India,
          Star Health and Aditya Birla.
        * Six plans were ingested twice from different PDFs, including a
          literal "corona-kavach-prospectus (1).pdf" duplicate, a filename
          truncated to "uvaan Health Insurance Policy.pdf", and -- worst --
          "Family Medicare Policy.pdf" alongside "Revised Family Medicare
          Policy (New policies issued after 1st April 2024).pdf".

        Grouping by (insurer, source_pdf) yields exactly nine sections every
        time, so that triple is the real document key. Merging the groups would
        blend superseded wording with its own revision, so the caller picks one.

        `source_pdf` is not indexed, so grouping happens client-side over the
        at-most-27 rows the indexed `plan_name` filter returns.
        """
        points, _ = await self._client.scroll(
            collection_name=self._s.policy_collection,
            limit=128,
            with_payload=True,
            with_vectors=False,
            scroll_filter=Filter(
                must=[FieldCondition(key="plan_name", match=MatchValue(value=plan_name))]
            ),
        )
        grouped: dict[tuple[str, str], list[Section]] = {}
        for p in points:
            payload = p.payload or {}
            key = (
                canonical.canonical_insurer(payload.get("insurer_name")),
                (payload.get("source_pdf") or "").strip(),
            )
            grouped.setdefault(key, []).append(_to_section(payload))

        docs: list[PlanDocument] = []
        for (insurer, pdf), sections in grouped.items():
            sections.sort(key=lambda s: _SECTION_RANK.get(s.section_id, 99))
            docs.append(PlanDocument(
                insurer=insurer, plan_name=plan_name, source_pdf=pdf,
                sections=sections,
            ))
        # Most likely-current document first: an explicitly revised or newer
        # filename wins, then the more completely extracted one.
        docs.sort(key=lambda d: (-d.revision_rank, -d.total_chars, d.source_pdf))
        return docs

    async def fetch_plan_sections(self, plan_name: str) -> list[Section]:
        """Sections of the single best-matching document for a plan name."""
        docs = await self.fetch_plan_documents(plan_name)
        return docs[0].sections if docs else []

    async def plan_features(self, plan_name: str) -> dict | None:
        """The flat feature row for a plan, with -1 sentinels normalised away."""
        points, _ = await self._client.scroll(
            collection_name=self._s.feature_collection,
            limit=1,
            with_payload=True,
            with_vectors=False,
            scroll_filter=Filter(
                must=[FieldCondition(key="plan_name", match=MatchValue(value=plan_name))]
            ),
        )
        if not points:
            return None
        payload = dict(points[0].payload or {})
        payload.pop("text", None)  # large duplicate of the section text
        for key, value in list(payload.items()):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                payload[key] = canonical.clean_number(value)
        return payload

    async def filter_plans_by_features(
        self, *, flags_true: list[str], limit: int = 50
    ) -> list[dict]:
        """Structured filter over the feature collection.

        Every `has_*` / `cover_type_*` flag is indexed there, so this is served
        server-side and is exact -- the right tool for "which plans cover X",
        far more reliable than similarity search.
        """
        must = [FieldCondition(key=f, match=MatchValue(value=True)) for f in flags_true]
        points, _ = await self._client.scroll(
            collection_name=self._s.feature_collection,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            scroll_filter=Filter(must=must) if must else None,
        )
        out = []
        for p in points:
            payload = dict(p.payload or {})
            payload.pop("text", None)
            if self._s.exclude_group_client_side and canonical.is_group_product(payload):
                continue
            out.append(payload)
        return out


# NOTE: an earlier version guessed raw insurer spellings from folded canonical
# keys by permuting case and punctuation. It generated 0 of 2 matches for
# "IFFCO Tokio" / "IFFCO-Tokio", so insurer-filtered searches silently returned
# nothing for that insurer. Spellings are now read from the collection itself
# (there are only 16 distinct values) -- see PolicyStore.load_insurer_spellings.
