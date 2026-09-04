# Policy Collections Audit — `policy_units_v2` + `migration_payload_v2`

**Date:** 2026-09-04
**Scope:** Qdrant Cloud collections backing the planned policy finding/answering agent.
**Method:** Read-only. Every point in both collections was scrolled (not sampled) — 1,826 + 156.
**Status:** This repo's `src/rag/` already reads both collections. Of the defects below, the
insurer-spelling split is already handled in `src/rag/canonical.py`; **the truncation defect
(§4) is not.**

This document exists so the measurement exercise does not have to be repeated. Every figure
below was measured on the dates above; the reproduction scripts are in the last section.
Numbers will drift if the collections are re-ingested — re-run the scripts, don't trust these.

---

## 1. Connection

```
QDRANT_URL        = https://<cluster-id>.europe-west3-0.gcp.cloud.qdrant.io   # real value in .env
QDRANT_API_KEY    = <in .env — not recorded here; see .env.example, key needs rotating>
POLICY_COLLECTION = policy_units_v2
FEATURE_COLLECTION= migration_payload_v2
```

Reachable, key valid, both collections `green`. Nine collections exist on the cluster:
`user_policies`, `policy_sections`, `policy_section_updated`, `policy_units_v2`,
`user_purchased_policies`, `migration_payload_v2`, `faq_collection`, `migration_payload`,
`policy_units_fuller`. Only the two named above were audited.

**Repo state:** `.env` is fully configured for this cluster — `QDRANT_URL`, `QDRANT_API_KEY`,
`POLICY_COLLECTION`, `FEATURE_COLLECTION`, plus `EMBEDDING_MODEL=text-embedding-3-small`,
`EMBEDDING_DIM=1536`, `SEARCH_TOP_K=24`, `MIN_SCORE=0.25`, `MAX_PLANS=4`,
`ANSWER_MODEL=claude-opus-5`.

An answering pipeline already exists in `src/rag/` (2,860 lines):
`store.py` (Qdrant access), `resolver.py` (plan resolution), `canonical.py` (defect shims),
`service.py`, `answer.py` / `answer_openai.py`, `api.py`, `cli.py`, `embeddings.py`,
`config.py`, plus `src/qdrant_health.py`.

`canonical.py` already documents five upstream defects it compensates for:

| | defect | status |
|---|---|---|
| 1 | `insurer_name` — 16 spellings for ~11 companies, collections disagree | **handled** |
| 2 | numeric feature fields use `-1` for "unknown" rather than null | **handled** |
| 3 | `coverage_scope` has no payload index (Qdrant strict mode) | **handled** |
| 4 | COVID-era standard products still in corpus | **handled** |
| 5 | cp1252 text decoded as UTF-8 → mojibake | **handled** |
| **6** | **`text` truncated at ~20,033 chars (§4)** | **NOT handled — new** |

Verified by grep: no source file in `src/` references the truncation marker, and no code reads
the `full_*` backfill fields. `store.py:121` reads `section_json` as `structured`, but the
HEAD/CONT rule (§3) is not used.

---

## 2. Collection shapes

| | `policy_units_v2` | `migration_payload_v2` |
|---|---|---|
| Points | 1,826 | 156 |
| Vector | 1536-d, Cosine | 1536-d, Cosine |
| `indexed_vectors_count` | 0 | 0 |
| Payload indexes | 4 | 91 |
| Grain | one point per (plan, section, chunk) | one point per plan |
| Payload fields | 14 | 129 |
| Distinct `plan_name` | 154 | 154 |
| Distinct `insurer_name` | 13 | 12 |

`indexed_vectors_count: 0` is **not a fault.** The cluster's `indexing_threshold` is 10,000 and
both collections are below it, so search runs exact brute-force — slower at scale, more accurate
than HNSW, irrelevant at 1,826 points.

### `policy_units_v2` — indexed payload keys
`plan_name`, `plan_type`, `section_id`, `insurer_name` (all `keyword`).

### Section distribution (sums to 1,826)

| `section_id` | points |
|---|---|
| `benefits` | 374 |
| `terms_and_conditions` | 206 |
| `exclusions` | 204 |
| `optional_benefits` | 184 |
| `claim_process` | 184 |
| `value_added_services` | 173 |
| `eligibility` | 171 |
| `customer_support` | 169 |
| `plan_overview` | 161 |

Nine values, fixed taxonomy. This matters for retrieval design — see §6.

### `migration_payload_v2` — field families
- **Typed scalars:** `ped_waiting_period_months`, `initial_waiting_period_days`,
  `copay_percentage`, `min_sum_insured`, `max_sum_insured`, `grace_period_days`,
  `free_look_period_days`, entry ages, pre/post hospitalization days.
- **Tri-state booleans:** every `has_X` is paired with `has_X_known`.
  `has_maternity_cover: false` + `has_maternity_cover_known: true` = genuinely absent;
  `_known: false` = not extracted. **Collapsing "unknown" into "no" is the compliance failure
  mode here.** Do not let the agent do it.
- **Structured blobs:** `full_benefits`, `full_exclusions`, `full_terms_and_conditions`,
  `full_optional_benefits`, `full_eligibility`, `full_claim_process`,
  `full_value_added_services`, `full_customer_support`.
- **Flat text:** `text` (embedded) and `full_text` (payload only).

Text field sizes:

| field | min | median | max |
|---|---|---|---|
| `text` (embedded) | 1,325 | 1,419 | 1,631 |
| `full_text` (not embedded) | 16,674 | 85,998 | 249,328 |

The embedded field is a ~1,400-char summary. Vector search on this collection matches
*plan-level gist*, never clause detail.

---

## 3. The `policy_units_v2` payload rule (HEAD / CONT)

The payload looks heterogeneous — `section_json` is populated on some points and `NULL` on
others, `chunk_ordinal` is present on some and absent on others. It resolves to one rule,
**verified with zero violations across all 1,826 points**:

```
section_json PRESENT  ⟺  chunk_ordinal is None  OR  chunk_ordinal == 0    → HEAD
section_json NULL     ⟺  chunk_ordinal >= 1                               → CONT
```

| | count | text: min / median / p90 / max |
|---|---|---|
| **HEAD** — canonical record for a (plan, section) | 1,441 | 52 / 6,774 / 20,033 / 20,033 |
| **CONT** — continuation chunk, text only | 385 | 49 / 1,893 / 3,912 / 4,039 |

Supporting counts: 425 points carry `chunk_ordinal`; 40 of those also carry `section_json`
(all at `chunk_ordinal == 0`); 425 − 40 = 385 = the CONT count exactly.

**So `bool(section_json)` is a perfect discriminator.** No heuristics needed.

Most sections are a single point — 1,300 of the 1,386 (plan, section) groups. The tail is
long: max 75 points in one group (`Acko Health II` / `benefits`), then 68, 48, 21, 17, 16.

### `text` already contains `section_json`
- `--- Structured Extract ---` marker appears in **1,098** points
- `Additional Unstructured Data:` appears in **1,038** points
- `_verbatim_md_text` appears in **1,099** of the 1,441 HEAD points

`text` = verbatim markdown + a flattened rendering of `section_json`. So `section_json` is
**redundant for retrieval** — its value is as **typed fields for citation**, not as extra
context to feed the model.

---

## 4. Truncation — the one real data defect

**358 points in `policy_units_v2` have their `text` cut off mid-sentence**, ending with the
literal marker:

```
...[TRUNCATED DUE TO LENGTH]...
```

| measure | value |
|---|---|
| Points carrying the marker | **358** of 1,826 (20%) |
| Of those, exactly 20,033 chars | 349 |
| The other 9, at lengths | 17,167 · 17,176 · 17,179 · 17,408 · 17,686 · 17,951 · 19,114 · 19,233 |
| Plans affected | **135** of 154 |
| Unique (plan, insurer, section) | 352 |
| Unique (plan, section) | 348 ← *collapses duplicate-named plans; do not use this key* |

All truncated points are HEAD points with `chunk_ordinal: None` — **no continuation chunks
exist for them.** The missing tail is not stored anywhere in `policy_units_v2`.

### By section (sums to 358)

| section | truncated |
|---|---|
| `benefits` | 139 |
| `terms_and_conditions` | 99 |
| `exclusions` | 75 |
| `claim_process` | 19 |
| `value_added_services` | 13 |
| `optional_benefits` | 8 |
| `customer_support` | 3 |
| `eligibility` | 2 |

The three worst-hit are exactly the sections an insurance answer depends on.

### By insurer (sums to 358)

Star Health 67 · Aditya Birla 57 · Oriental 43 · National 38 · Shriram General 36 ·
IFFCO TOKIO 33 · Universal Sompo 31 · Acko 18 · Galaxy 18 · United India 17

`Care`, `Tata AIG` and `TATA AIG` are unaffected.

### Why it matters

For a truncated `exclusions` section the agent sees a list that stops mid-sentence. It cannot
distinguish "this exclusion does not exist" from "this exclusion was in the part that got cut."
Answering "yes, that's covered" from a truncated exclusions list is a mis-selling exposure.

### Not every large point is truncated

Point `001f0d48-4cc3-4c83-68aa-411556290231` (`Activ One Max+` / `optional_benefits`) is
16,738 chars, ends cleanly, and is **complete**. Only the marker indicates truncation — size
alone does not. `Activ One Max+` does have 4 other sections truncated, just not that one.

---

## 5. Truncation is recoverable — no PDF re-ingest needed

`migration_payload_v2` was scanned exhaustively: every string in every one of the 129 fields,
across all 156 points.

**The only field carrying the truncation marker is `full_text`** (139 of 156 points, 137 plans).
All eight structured `full_*` blobs are **clean everywhere**.

And they hold far more than what was cut:

| plan | section | `policy_units_v2` | `migration_payload_v2` |
|---|---|---|---|
| Activ One Max+ | `benefits` | 20,033 (cut) | `full_benefits` — 132,827 clean |
| Activ One Max+ | `terms_and_conditions` | 20,033 (cut) | `full_terms_and_conditions` — 52,851 clean |
| Galaxy Promise | `benefits` | 20,033 (cut) | `full_benefits` — 168,794 clean |
| Galaxy Promise | `exclusions` | 20,033 (cut) | `full_exclusions` — 55,588 clean |
| Star Critical Illness Multipay | `benefits` | 20,033 (cut) | `full_benefits` — 43,728 clean |

### Exhaustive verification

Every truncated `(plan_name, insurer_name, section_id)` was checked against its counterpart
field — not spot-checked:

```
truncated triples ............ 352
  recoverable, clean ......... 352      ← 100%
  also truncated there ....... 0
  no counterpart field ....... 0

recovered blob size: min 16,842 · median 34,781 · max 240,167 chars   (vs 20,033 cut)
```

Median recovery is **~74% more text** than the truncated version held, up to 240k in the worst
cases. Every affected `(plan_name, insurer_name)` pair exists in `migration_payload_v2` with a
matching `insurer_name` spelling — the Tata mismatch (§6) does not touch the affected set.

### Section → field mapping for the backfill

| `section_id` | `migration_payload_v2` field |
|---|---|
| `benefits` | `full_benefits` |
| `exclusions` | `full_exclusions` |
| `terms_and_conditions` | `full_terms_and_conditions` |
| `optional_benefits` | `full_optional_benefits` |
| `eligibility` | `full_eligibility` |
| `claim_process` | `full_claim_process` |
| `value_added_services` | `full_value_added_services` |
| `customer_support` | `full_customer_support` |
| `plan_overview` | *(never truncated — no mapping needed)* |

**The fix is a lookup, not a re-ingest.** Retrieve from `policy_units_v2` as normal; if the hit
contains the marker, swap in the matching `full_*` blob from `migration_payload_v2`.

---

## 6. Join keys — two traps

### Trap 1: `insurer_name` does not match across collections

| plan | `policy_units_v2` | `migration_payload_v2` |
|---|---|---|
| Tata Aig Medicare Plus | `Tata AIG` | `Tata` |
| Tata Aig Medicare Select | `Tata AIG` | `Tata` |
| Tata Aig Medicare Premier | `TATA AIG` | `Tata` |

Note the split *within* `policy_units_v2` itself — `Tata AIG` and `TATA AIG` are two distinct
keyword values, so even a correct in-collection filter silently drops Medicare Premier. This is
why the insurer counts differ (13 vs 12) despite identical plan coverage.

Tata is the only cross-collection mismatch found in the two audited collections.

**Already handled** — `src/rag/canonical.py` DEFECT 1 maps both `tata aig` and `tata` to
`Tata AIG General Insurance`, matching on a punctuation-stripped lowercase form (longest key
first), so the case split resolves too. Its map is broader than the audited set (16 spellings,
~11 companies, including insurers not present in these two collections) and also catches
`Activ`, an Aditya Birla product line that leaked into the insurer field. Use
`canonical.py` — do not write a second normalisation map.

### Trap 2: `plan_name` is not unique

`Arogya Sanjeevani Policy` is **three different insurers' plans** — the IRDAI-standard product:

| insurer | `migration_payload_v2` id | `source_pdf` |
|---|---|---|
| Aditya Birla | 393289914609651 | `Arogya Sanjeevani.pdf` |
| United India | 775692590724550 | `Arogya Sanjeevani Policy.pdf` |
| Star Health | 842668743960080 | `Prospectus_Arogya_Sanjeevani_Policy_V_8_9a95b2c87f.pdf` |

Each has 9 sections in `policy_units_v2` (27 points total). This is why
`migration_payload_v2` has 156 points but 154 distinct plan names.

**Always key on `(plan_name, insurer_name)`.** Keying on `plan_name` alone returns the wrong
insurer's document.

### What does match
`plan_name` overlap is exact: 154 in both, 0 only in `policy_units_v2`, 0 only in
`migration_payload_v2`.

---

## 7. Retrieval implications

**Do not lead with vector search.**

1. **HEAD text is too large for one vector.** Median 6,774 chars, p90 20,033 — all averaged into
   a single 1536-d vector. For "maternity waiting period in Care Supreme," the relevant clause is
   ~2% of the embedded text; the vector is dominated by everything else.
2. **`section_id` is a fixed 9-value taxonomy and an indexed keyword.** So the primary path
   should be *classify the question → `section_id` → filter-fetch*, not similarity search.
   Vector search becomes the fallback for "which section covers this?", not the default.
3. **`migration_payload_v2` answers structured questions with no vector at all** — 91 indexed
   payload keys cover "which plans cover X with waiting period under N".

### Embedding model — resolved

`.env` declares `EMBEDDING_MODEL=text-embedding-3-small` and `EMBEDDING_DIM=1536`, which
matches the 1536-d vectors on both collections. Query embedding is already wired through
`src/rag/embeddings.py`.

Keep the two in sync: `text-embedding-3-large` defaults to 3072-d and would fail on dimension,
but a *same-dimension, different-model* swap (e.g. `ada-002`, also 1536-d) fails **silently** —
it returns plausible nonsense rather than an error.

---

## 8. Reproduction

Read-only. Requires `QDRANT_URL` and `QDRANT_API_KEY` in the environment.

```python
import json, os, time, urllib.request
from collections import Counter

B = os.environ["QDRANT_URL"].rstrip("/")
H = {"api-key": os.environ["QDRANT_API_KEY"], "Content-Type": "application/json"}
MARK = "[TRUNCATED DUE TO LENGTH]"

def scroll(coll, keys=True, limit=60):
    """Page a whole collection. Small limit: payloads reach 250 KB and large
    pages hit IncompleteRead on this cluster."""
    out, off = [], None
    while True:
        body = {"limit": limit, "with_payload": keys, "with_vector": False}
        if off:
            body["offset"] = off
        for attempt in range(5):
            try:
                r = urllib.request.Request(f"{B}/collections/{coll}/points/scroll",
                                           data=json.dumps(body).encode(), headers=H)
                d = json.loads(urllib.request.urlopen(r, timeout=200).read().decode())["result"]
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2)
        out += [p["payload"] for p in d["points"]]
        off = d.get("next_page_offset")
        if not off:
            return out

pu = scroll("policy_units_v2", ["plan_name", "insurer_name", "section_id",
                                "text", "chunk_ordinal", "section_json"])

# HEAD/CONT invariant — expect 0
violations = [p for p in pu
              if bool(p.get("section_json")) ==
                 (p.get("chunk_ordinal") is not None and p["chunk_ordinal"] >= 1)]

# Truncation census — expect 358 points / 135 plans / 352 triples
cut = [p for p in pu if MARK in (p.get("text") or "")]
print(len(cut), len({p["plan_name"] for p in cut}),
      len({(p["plan_name"], p["insurer_name"], p["section_id"]) for p in cut}))
print(Counter(p["section_id"] for p in cut).most_common())
```

Full-field truncation scan of `migration_payload_v2` (use `limit=8`; payloads are large):

```python
def walk(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)

hits = Counter()
for p in scroll("migration_payload_v2", True, 8):
    for k, v in p.items():
        if any(MARK in s for s in walk(v)):
            hits[k] += 1
# expect: {'full_text': 139} and nothing else
```

---

## 9. Decisions still open

1. **Truncation (§4/§5)** — the only unhandled defect. Backfill from `migration_payload_v2` at
   query time (recommended: verified complete for all 352, no re-ingest) or re-ingest the 358
   sections properly? The query-time fix belongs in `canonical.py` as DEFECT 6, alongside the
   five shims already there.
2. **Agent shape** — a ReAct agent over this data would sit on top of the existing `src/rag/`
   pipeline, not replace it. Proposed tools: `find_plans` (payload filter on
   `migration_payload_v2`), `get_plan_facts` (typed fields for citation), `read_section`
   (HEAD + CONT reassembly with truncation backfill), `search_clauses` (vector, fallback only).
   Decide whether these wrap `store.py`/`resolver.py` or bypass them.

**Resolved since first draft:** embedding model (`text-embedding-3-small`, 1536-d, in `.env`);
insurer-name normalisation (already in `canonical.py` DEFECT 1).
