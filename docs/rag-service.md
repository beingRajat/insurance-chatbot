# Policy Answer Service

Grounded question answering over the health-policy corpus already in Qdrant.
FastAPI in front, `qdrant-client` and `anthropic` called directly — no
orchestration framework in the latency path, per the house rule in `CLAUDE.md`.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env      # OPENAI_API_KEY is the only key strictly needed
uvicorn src.rag.api:app --reload
```

Then open:

- **<http://127.0.0.1:8000/>** — the chat interface
- <http://127.0.0.1:8000/docs> — generated API reference

## The chat interface

A single self-contained page at `src/rag/static/chat.html`, served same-origin
by the app itself so it needs no CORS configuration and no host to point at.
It cannot be hosted anywhere else without opening CORS — and it deliberately
loads no scripts from anywhere, so there is nothing to pin or audit beyond the
one file.

What it surfaces, and why each of these is on screen rather than buried:

- **A grounded / ungrounded badge on every answer.** An uncited answer is
  called out in red rather than shown as though it were sourced. This is the
  single most important thing on the page.
- **`unverified_quotes`, when non-zero.** On the OpenAI backend this means the
  model cited text that is not in the document it named. Silently dropping
  those without saying so would hide exactly the failure worth knowing about.
- **Expandable evidence** — every quoted passage with its section and, where
  recoverable, its page number.
- **The full trace** in the side panel: mode, provider, sections used, top
  score, what was dropped and why, retrieval-vs-answer latency split, tokens
  and cache hits.
- **Which document was read**, plus any same-plan documents that were ignored.
- **Benefit filtering** driven by `/plans/by-feature` — an exact structured
  filter with no model involved, labelled as such so it is not mistaken for a
  model answer.
- **Insurer disambiguation.** When a plan name is shared across insurers the
  API returns 404 rather than guessing; the page reveals the insurer selector
  and explains why.

Rendering uses `textContent` throughout — no `innerHTML` — so policy text
cannot inject markup.

## Providers — one key or two

`ANSWER_PROVIDER` selects the answer backend. Default is `openai`, so a single
`OPENAI_API_KEY` covers both query embedding and answer synthesis.

| | `openai` (default) | `anthropic` |
|---|---|---|
| Answer model | `OPENAI_ANSWER_MODEL`, default `gpt-5.4-mini` | `claude-opus-5` |
| Keys needed | `OPENAI_API_KEY` | `OPENAI_API_KEY` + `ANTHROPIC_API_KEY` |
| How grounding works | Model emits quotes; **the service verifies each one against the source** and discards those that do not match | The API reports which document each cited span came from |
| Fabricated citation | Caught, discarded, counted in `unverified_quotes` | Cannot arise — the mapping is not the model's to assert |
| Prompt caching | None — repeat questions on a plan pay full input price | ~10x cheaper input on follow-ups (1h TTL) |
| Structured output + citations together | Yes, one JSON response | No — mutually exclusive on that API |

The verification step is what makes the OpenAI path defensible: a model
asserting a source is treated as a claim to be checked, not a fact. Quotes are
matched case- and whitespace-insensitively, so reflowed text passes, but a
paraphrase does not. Watch `unverified_quotes` in the response — non-zero means
the model cited text that is not in the document it named.

Verify both models before trusting either path:

```bash
python -m src.rag.cli verify-answer-model   # model exists + does strict JSON schema
python -m src.rag.cli verify-embeddings     # query model matches the index
```

## Two query paths, because insurance questions come in two shapes

**`POST /plan/ask` — one named plan, read whole.** All nine sections of the
plan go into context (~26–47K tokens). This is the reliable path and should be
the default in any UI.

The reason is that an insurance answer is composed from clauses that sit far
apart in the document. "Am I covered for a knee replacement?" needs the
coverage grant, the specific-disease waiting period, the pre-existing-disease
waiting period, any sub-limit, the co-payment, the room-rent linkage, and the
exclusions list. Top-k chunk retrieval reliably returns two or three of those
and silently drops the rest, producing an answer that is fluent, traceable to
what it found, and materially wrong. Reading the plan whole removes the failure
mode entirely, and at this size it is cheap.

**`POST /ask` — across plans.** Vector search proposes candidate plans, then
each candidate is read *in full* before answering. Retrieval narrows the field;
it never decides the answer from a partial view.

**`POST /plans/by-feature` — exact structured filter, no model involved.** All
70 `has_*` / `cover_type_*` / numeric fields on `migration_payload` are
indexed, so "which plans cover maternity and OPD" is an exact server-side
filter rather than an approximation. Prefer this over semantic search whenever
the question is really a filter.

## Grounding

Retrieved sections are passed as `document` blocks with citations enabled, so
each claim comes back attached to the section it came from — and a claim with
no citation is visibly ungrounded. The response carries `grounded: false` when
nothing was cited at all.

The corpus has no page field, but the ingest pipeline left `### PAGE n` markers
inside the text of 735 of 1,422 sections (51.7%). `canonical.locate_page`
recovers a page from those where it can, so citations read
`"Plan — Exclusions, page 14"` when the page is knowable and
`"Plan — Exclusions"` when it is not. A page is never guessed.

## Verify the embedding model before trusting `/ask`

```bash
python -m src.rag.cli verify-embeddings
```

The collections were built by another pipeline with 1536-dimensional vectors.
`text-embedding-3-small` and `text-embedding-ada-002` both produce 1536 dims,
so matching dimensions is **not** proof the model matches — and a mismatch
degrades retrieval silently rather than raising. This command re-embeds text
taken from a stored point and compares against that point's own vector. Cosine
above 0.99 confirms the match; anything materially lower means the wrong model
is configured and `/ask` is running degraded.

`/plan/ask` needs no embeddings, so it is unaffected either way.

## Data defects this service works around

Each shim is documented at its definition with the defect it compensates for,
so it can be deleted when the upstream data is fixed rather than quietly
outliving its purpose.

| # | Defect | Workaround | Where |
|---|--------|-----------|-------|
| 1 | `coverage_scope` has no payload index, and the cluster's strict mode rejects unindexed filters — so group products cannot be filtered server-side (171 of 1,422 sections) | Dropped client-side after retrieval; counted in `trace.dropped_group` | `canonical.is_group_product` |
| 2 | 16 insurer spellings for ~11 companies, differing between collections; `Activ` appears as an insurer but is an Aditya Birla product line | Canonical map, applied on read; insurer filters enumerate every spelling | `canonical.canonical_insurer` |
| 3 | Missing numbers stored as `-1`, not null — `maternity_waiting_period_months` is `-1` on 112 of 156 plans | Normalised to `None`; rendered "not stated in the source data" so the model cannot read it as zero | `canonical.clean_number` |
| 4 | Withdrawn COVID-era and mandated standard products still present | Excluded by default (`EXCLUDE_WITHDRAWN`) | `canonical.is_withdrawn_product` |
| 5 | cp1252 bytes decoded as UTF-8, leaving U+FFFD in place of bullets and dashes | Folded to ASCII on read | `canonical.clean_text` |
| 6 | The two collections use different `plan_name` conventions — `"Activ One Max+"` in the section collection, `"Aditya Birla Activ One Max+"` in the feature collection. **A raw-string join matches only 42 of 149 plans (28%).** | Names folded to a comparable key, keyed by `(insurer, fold)`; the resolver also gives users tolerant lookup. Recovers 56 further joins, to 98 | `resolver.PlanResolver` |
| 7 | `plan_name` is not a document key. `Arogya Sanjeevani Policy` spans 3 insurers (27 sections), and 6 plans were ingested twice from different PDFs — including a `(1).pdf` duplicate, a filename truncated to `uvaan Health…`, and `Family Medicare Policy.pdf` alongside its own `Revised …` version | Grouped by `(insurer, plan, source_pdf)`, which yields exactly 9 sections every time. One document is chosen, revisions preferred and `(1).pdf` duplicates penalised; the choice and the ignored alternatives are recorded in the trace. A plan name shared across insurers **refuses** and asks which insurer | `store.fetch_plan_documents` |

Note that #4 was initially over-broad on my part: I had also classified Arogya
Sanjeevani and Saral Suraksha Bima as withdrawn. Both are IRDAI-mandated
standard products still on sale, so they are now classified separately and kept
by default (`EXCLUDE_MANDATED_STANDARD=false`). See the audit log in
`rag-implementation.md`.

Run `python -m src.rag.cli health` for live index coverage,
`src/qdrant_health.py` for the full per-field filterability probe, and
`docs/rag-implementation.md` for the module-by-module reference and audit log.

## Guardrails

- **Declines rather than guesses.** Below `MIN_SCORE` the service returns "not
  found" instead of answering from weak context.
- **Wording questions, not advice.** The system prompt forbids recommending
  which policy to buy — that is regulated distribution activity.
- **Every response carries a caveat** naming the corpus limitation.
- **Full trace on every answer**: plans considered, sections used, document
  read, alternatives ignored, drop counts, latency split, token usage
  including cache hits.
- **Refusals surface as 422**, not as an empty 200.
- **Server-side fallbacks are enabled** on the answer call, so a policy
  refusal is rescued on a fallback model within the same request rather than
  failing the caller. If the beta is not enabled on your account the code
  retries once on the plain endpoint, so a beta gate cannot take the service
  down.

## Known gaps

- **Nothing is reranked.** `/ask` ranks plans by their best-matching section.
  A cross-encoder reranker is the obvious next quality win.
- **No hybrid search.** Insurance queries are full of exact terms (UIN, PED,
  AYUSH, "sub-limit", "room rent") where BM25 beats dense retrieval. Qdrant
  supports sparse vectors; the collections have none.
- **No eval set.** Until 150-ish golden question/answer pairs exist, retrieval
  tuning has no target. This is the highest-leverage missing piece.
- **`MIN_SCORE` is uncalibrated.** Observed scores ran 0.91–1.00 against a
  default of 0.25, so the "decline rather than guess" guardrail on `/ask`
  almost certainly never fires today. Calibrating it needs the eval set.
- **No conversation memory.** Each request is independent. Prompt caching
  makes follow-ups on the same plan cheap, but the caller must resend context.
- **The cluster is in `europe-west3`.** Policy text is public, but logged user
  questions about their own health are not — worth a DPDP review before this
  carries real traffic.
