# Implementation reference — Policy Answer Service

What each module does, why it does it that way, and what is deliberately not
done. Companion to `rag-service.md` (how to run it) and
`rag-architecture.html` (why this shape).

Audited 21 Aug 2026. The audit log is at the end, including six defects found
in my own first pass.

---

## Module map

```
src/rag/
  config.py        Settings from env. No secrets in code.
  canonical.py     Normalisation shims, one per known data defect.
  resolver.py      Plan-name resolution across the two collections.
  store.py         Qdrant access. All filters stay inside the indexed fields.
  embeddings.py    Query embedding + the model-match verifier.
  answer_base.py   Shared types, system prompt, quote verification.
  answer.py        Anthropic backend -- native document citations.
  answer_openai.py OpenAI backend -- structured output + verified quotes.
  service.py       Orchestration. Two query paths + a dry-run path.
  api.py           FastAPI surface.
  cli.py           Operational commands.
```

Both answer backends satisfy `answer_base.SupportsAnswer`, selected by
`ANSWER_PROVIDER` via `service.build_answerer`. They are **not** equivalent in
how grounding is established, and the difference is the most important thing to
understand about this service:

* **Anthropic** passes sections as `document` blocks with `citations: enabled`.
  The API returns, for each cited span, the source text and which document it
  came from. Provenance is produced by the platform.
* **OpenAI** has no such mechanism. The model is required by strict JSON schema
  to return each claim with a `source_id` and a verbatim `quote`, and
  `answer_base.verify_quotes` then checks the quote genuinely occurs in the
  document named. Failures are discarded and counted in
  `Answer.unverified_quotes`. Provenance is verified after the fact.

The second is weaker in kind, not just degree: it can only catch a bad citation
after the model makes one, and it cannot detect a claim the model simply chose
not to cite. It is nonetheless a real check -- matching is whitespace- and
case-insensitive so reflowed quotes pass, but paraphrase does not.

Dependency direction is strictly one-way — `api → service → {store, answer,
embeddings, resolver} → canonical → config`. Nothing imports upward, so any
module can be tested with only its own dependencies present.

---

## `config.py`

Pydantic-settings, reading `.env` or the process environment. Credentials are
`SecretStr`, so they cannot be printed by accident.

Two values that are easy to get wrong:

- **`answer_max_tokens: 16000`.** Adaptive thinking is billed *inside*
  `max_tokens`. At `effort=high` thinking can consume thousands of tokens
  before the answer begins, so the original 4096 risked truncating mid-answer.
  16000 leaves room to think while keeping a non-streaming request under the
  SDK's HTTP timeout.
- **`min_score: 0.25`.** This is the "decline rather than guess" threshold and
  it is **not calibrated** — see Known gaps.

---

## `canonical.py` — one shim per data defect

Every function compensates for something wrong upstream and names the defect
in a comment, so it can be deleted when the data is fixed rather than becoming
permanent folklore.

| Function | Defect it exists for |
|---|---|
| `canonical_insurer` | 16 insurer spellings for ~11 companies; `Activ` is a product line, not an insurer |
| `clean_number` / `describe_number` | Missing numbers stored as `-1`, not null |
| `is_group_product` | `coverage_scope` unindexed, so group products cannot be filtered server-side |
| `is_withdrawn_product` | Corona Kavach / Rakshak withdrawn in 2022 but still present |
| `is_mandated_standard_product` | Arogya Sanjeevani and Saral Suraksha **are still sold** — separated so they are not wrongly hidden |
| `clean_text` | cp1252 bytes decoded as UTF-8, leaving U+FFFD |
| `locate_page` | No page field, but `### PAGE n` markers survive in 51.7% of section text |

`describe_number` renders an unknown as **"not stated in the source data"**
rather than as a number. This matters more than it looks: the model is
explicitly instructed never to read that phrase as zero, which is what stops
`maternity_waiting_period_months = -1` (112 of 156 plans) from becoming "no
waiting period".

`locate_page` returns `None` rather than guessing. When an exact substring
match fails it retries against a whitespace-collapsed copy of the text while
maintaining an exact index mapping — never an estimated offset, because a
wrong page citation is worse than no page citation.

---

## `resolver.py` — the cross-collection join

The section collection stores `plan_name = "Activ One Max+"` with the insurer
in a separate field. The feature collection stores
`plan_name = "Aditya Birla Activ One Max+"`. Joining on the raw string matches
only **42 of 149 plans (28%)** and fails silently for the rest; folding
recovers 56 more, to 98.

`fold_plan()` reduces both to a comparable key, and the resolver maps a
caller's plan name to the exact stored string in each collection. It also gives
users tolerant lookup — "activ one max" finds "Activ One Max+".

Three things the fold has to get right, all of which the first version got
wrong (see Audit finding 1):

- **The key is `(canonical insurer, folded plan)`, not the fold alone.** Seven
  distinct plans are named "…Arogya Sanjeevani Policy"; keying on the fold
  collapsed them onto one entry and made six unreachable.
- **`+` is preserved as a `plus` token.** "Activ One Max" and "Activ One Max+"
  are different products and dropping the `+` merged them.
- **Brand-token stripping is skipped when it would empty the string.**
  "National Health Policy" is entirely brand and noise words; stripping
  everything folded it to `""` and it vanished.

`load()` is guarded by an `asyncio.Lock` and an idempotence check, so
concurrent first requests do not each scroll both collections.

**Result: all 149 plans resolve to themselves.** Verified as an assertion, not
an assumption. (149 rather than the 127 quoted before the fix: the key is now
`(insurer, fold)`, so plans that previously collided onto one key are counted
— and reachable — separately.)

---

## `store.py` — Qdrant access

Only `insurer_name`, `plan_name` and `section_id` are indexed on the section
collection, and the cluster runs strict mode
(`unindexed_filtering_retrieve=false`), so a filter on anything else returns
**HTTP 400 rather than falling back to a scan**. Every server-side filter here
stays inside that set; everything else is applied client-side.

### `PlanDocument` and why `plan_name` is not a document key

`fetch_plan_documents()` groups by `(insurer, plan_name, source_pdf)`, which
yields exactly nine sections every time. `plan_name` alone does not, for two
different reasons:

- Mandated products share a name across insurers — "Arogya Sanjeevani Policy"
  returns 27 sections spanning three insurers.
- Six plans were ingested twice from different PDFs, including a literal
  `corona-kavach-prospectus (1).pdf`, a filename truncated to
  `uvaan Health Insurance Policy.pdf`, and — worst — `Family Medicare
  Policy.pdf` alongside `Revised Family Medicare Policy (New policies issued
  after 1st April 2024).pdf`.

Merging the groups would blend a superseded wording with its own revision, so
one document is chosen: `revision_rank` prefers an explicit "Revised"/"New"
filename and penalises `(N).pdf` duplicates, then the more completely extracted
one wins. **The choice and the ignored alternatives are recorded in the trace**
and logged at WARNING, so a human can see a decision was made.

A plan name spanning multiple insurers does not pick one — it **refuses** and
asks which insurer, because guessing there is a silent wrong answer.

### Insurer spellings are read, not guessed

`load_insurer_spellings()` reads the 16 distinct `insurer_name` values from the
collection and maps them to canonical names. The first version instead
permuted case and punctuation from a canonical name to guess the stored
spellings — it generated **0 of 2** matches for `IFFCO Tokio` / `IFFCO-Tokio`,
so insurer-filtered searches silently returned nothing for that insurer
(Audit finding 2).

When an insurer filter matches no stored spelling, the filter is **dropped with
a warning** rather than applied, so a bad filter degrades to a broader search
instead of to zero results.

---

## `embeddings.py`

`text-embedding-3-small` by default, with retries via tenacity (the SDK's own
retries are disabled so backoff is uniform) and a dimension guard.

`verify_against_stored()` is the important part. The collections were built by
another pipeline; `text-embedding-3-small` and `text-embedding-ada-002` both
produce 1536 dimensions, so **dimension agreement does not prove the model
matches**, and a mismatch degrades retrieval silently rather than raising. The
verifier re-embeds text taken from a stored point and compares against that
point's own vector — cosine above 0.99 confirms the match.

Run `python -m src.rag.cli verify-embeddings` before trusting `/ask`.
`/plan/ask` needs no embeddings and is unaffected.

---

## `answer.py`

Sections become `document` content blocks with `citations: {enabled: true}`, so
each claim comes back attached to the section it was drawn from, and an
unattributed claim is visible because it carries no citation. `grounded: false`
on the response means nothing was cited at all.

Three API constraints shaped this file:

- **`output_config.format` is never used.** Structured output and citations are
  mutually exclusive and return 400 together. A structured comparison table
  needs a second call without citations.
- **`document_index` must resolve against the blocks actually sent.**
  `_documents()` returns both the blocks and the sections they were built from,
  because a section skipped for budget shifts every subsequent index. Resolving
  against the original list mis-attributes every citation after the first skip
  (Audit finding 3).
- **A section that does not fit the budget is skipped, not truncated.** Half a
  clause is worse than none — the model cannot tell it is reading a fragment.
  Omissions are logged with a WARNING naming the setting to raise.

Prompt caching puts a 1h-TTL breakpoint on the last document block for
single-plan reads, so follow-ups re-read the plan at roughly a tenth of the
input cost. The system prompt carries its own breakpoint.

Server-side fallbacks (`fallbacks: "default"`) are enabled so a policy refusal
is rescued on a fallback model inside the same request. If that beta is not
enabled on the account, a `BadRequestError` triggers one retry on the plain
endpoint — a beta gate cannot take the service down.

### The system prompt

Rule 2 is the one that earns its place. It requires the model to check for the
coverage grant, all three waiting-period types, sub-limits, co-payment,
room-rent linkage and exclusions before answering a coverage question, and
states explicitly that an answer giving the coverage grant while omitting an
applicable sub-limit **is wrong even if every word is accurate**. That is the
failure mode chunk-based RAG produces by construction, and reading the plan
whole only removes the retrieval half of it — the model still has to be told to
compose.

Rule 4 keeps the service on the right side of regulated advice: it answers what
a document says and declines to recommend what to buy.

---

## `service.py`

Three paths.

**`ask_plan`** — resolve the name, load one document, read all nine sections,
append structured attributes, answer. Roughly 26–47K tokens. This is the
reliable path and should be the UI default.

**`ask_broad`** — embed, search sections, drop below `min_score`, rank plans by
their best-matching section, then **read each candidate plan in full** before
answering. Retrieval narrows the field; it never decides the answer from a
partial view. Each candidate is scoped to the insurer whose section actually
matched — without that, a hit on Star Health's "Arogya Sanjeevani Policy" would
be answered from Aditya Birla's document (Audit finding 4).

**`retrieve_only`** — the same retrieval, no model call. For tuning
`MIN_SCORE`/`MAX_PLANS`, for CI, and for running the pipeline before answer
credentials exist. This is how the retrieval path was validated here without
either API key.

`RetrievalTrace` accompanies every answer: plans considered, sections used,
document read, alternatives ignored, drop counts by reason, latency split
retrieval-vs-answer, and token usage including cache hits. Without it a wrong
answer is not debuggable.

---

## `api.py`

Nine routes' worth of behaviour, all verified:

| Route | Behaviour |
|---|---|
| `GET /health` | Cluster reachability + per-collection indexed-field list |
| `GET /plans` | 128 retail plans, group and withdrawn filtered |
| `POST /plan/ask` | One plan, read whole. `insurer` disambiguates shared names |
| `POST /ask` | Across plans |
| `POST /plans/by-feature` | Exact structured filter, no model involved |

Status codes are meaningful rather than decorative: `404` with the closest
matching plan names when a name misses, `404` with the insurer list when a name
is ambiguous, `422` when the model declines, `503` with the missing variable
named when a credential is absent, `400` when a feature flag is not indexed,
and `502` — not 400 — when the cluster itself faults (Audit finding 5).

Every response carries a `caveat` naming the corpus limitation, and every
request gets an `x-request-id` echoed into both the response header and the log
line.

`anthropic` and `qdrant-client` are called directly. No orchestration framework
in the latency path, per the house rule in `CLAUDE.md` — the same reason it
applies to the finance agent.

---

## Audit log

Six defects found in my own first implementation, all fixed and re-verified.
Each was found by measuring rather than by re-reading code, which is why they
are listed with numbers.

**1. Resolver fold collisions — ~20 plans unreachable.** Keying on the folded
plan name alone collapsed 14 groups; "Arogya Sanjeevani" merged 7 plans into 1.
Two plans folded to `""` and vanished. Fixed by keying on
`(insurer, fold)`, preserving `+`, and never folding to empty.
*Verified: 0 of 149 plans unresolvable, down from ~20.*

**2. Guessed insurer spellings — 0/2 match for IFFCO.** Permuting case and
punctuation to guess stored values missed both `IFFCO Tokio` and
`IFFCO-Tokio`, so insurer-filtered search silently returned nothing for that
insurer. Fixed by reading the 16 actual values from the collection.
*Verified: 16 raw values mapped, 0 guessed.*

**3. Wrongly excluded 12 currently-sold plans.** I had classified Arogya
Sanjeevani and Saral Suraksha Bima as "withdrawn". They are IRDAI-mandated
standard products that insurers are still required to offer. This was the same
silent-loss failure the module exists to prevent. Fixed by splitting withdrawn
(5 Corona products) from mandated-standard (12 plans, kept by default).
*Verified: retail plans 115 → 128.*

**4. Broad search could answer from the wrong insurer's document.**
`ask_broad` ranked plans by matching section, then re-fetched by `plan_name`
alone — which picks the first document for that name. For plans shared across
insurers the answer could come from a different insurer than the one that
matched. Fixed by scoping each candidate to its matching insurer.

**5. `document_index` mis-mapping, and silent truncation.** Citations were
resolved against the original section list, so any section skipped for budget
shifted every later index and mis-attributed the quote. Separately, a section
that did not fit was truncated mid-clause and included silently. Fixed:
`_documents()` returns the sections it actually used, and an oversized section
is skipped with a WARNING.

**6. Smaller corrections.** `max_tokens` 4096 → 16000 (thinking shares the
budget); `2e+07` rendered as `20,000,000`; cluster faults now `502` not `400`;
`locate_page`'s proportional offset estimate replaced with exact index mapping;
`asyncio.Lock` on resolver load; `/plans` now applies the same withdrawn filter
the query paths use, so it cannot advertise a plan `/ask` then refuses.

### Verification state

| | |
|---|---|
| Route checks | 9/9 pass |
| Plans resolving to themselves | 127/127 |
| Plans returning exactly 9 sections | 128/128, 0 exceptions |
| Modules compile | clean |
| Credentials in tracked files | none |
| `/ask` answer path | **unverified — needs `OPENAI_API_KEY` + `ANTHROPIC_API_KEY`** |
| `/plan/ask` answer path | **unverified — needs `ANTHROPIC_API_KEY`** |

---

## Known gaps

Ordered by how much they would change answer quality.

1. **No eval set.** 150-ish golden question/answer pairs with verified answers
   is the highest-leverage missing piece. Until it exists, every tuning
   decision below is guesswork, including `min_score`.
2. **`min_score = 0.25` is uncalibrated.** Observed real scores ran 0.91–1.00,
   so this threshold almost certainly never fires and the "decline rather than
   guess" guardrail is currently inert on `/ask`. It needs setting from the
   score distribution of genuinely-unanswerable questions, which needs the eval
   set. Treat the guardrail as untested until then.
3. **No reranker.** Plans are ranked by their single best-matching section. A
   cross-encoder pass is the obvious next quality win.
4. **No hybrid search.** Insurance queries are dense with exact terms — UIN,
   PED, AYUSH, "sub-limit", "room rent", rupee figures — where BM25 beats dense
   retrieval. Qdrant supports sparse vectors; these collections have none.
5. **Three residual same-insurer fold collisions** — `Activ Care`/`Activ
   Health`, and two National Insurance pairs. Exact-name lookup resolves them
   and `ResolvedPlan.ambiguous_with` reports the ambiguity, so nothing is lost,
   but fuzzy lookup on those names picks one arbitrarily.
6. **`fetch_plan_documents` caps at 128 points.** Adequate today (27 is the
   observed maximum) but it would truncate silently if a name were ever shared
   more widely.
7. **No conversation memory.** Each request is independent. Prompt caching
   makes follow-ups cheap, but the caller must resend context.
8. **The cluster is in `europe-west3`.** Policy text is public; logged user
   questions about their own health conditions are not. Worth a DPDP review
   before this carries real traffic.
