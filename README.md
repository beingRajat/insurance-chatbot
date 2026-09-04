# IRDAI Health Policy Corpus Builder

Production-oriented initial scraper for the IRDAI Health Insurance Products catalogue.

## What it does

1. Uses a real Chromium browser via Playwright (not guessed IRDAI API endpoints).
2. Reads the IRDAI catalogue pages and extracts:
   - archive status
   - financial year
   - insurer
   - UIN
   - product name
   - approval date
   - product type
   - document link
3. Builds a complete local catalogue first, and verifies coverage against
   the portal's own "Showing X - Y of N results" total before proceeding.
4. Filters to the configured 10 target insurers **by UIN insurer code**
   (see "Why identity comes from the UIN" below).
5. Keeps non-archived individual health policies using the UIN's structural
   fields, and excludes group, add-on/rider, travel, personal-accident and
   micro-insurance filings.
6. Deduplicates by UIN, then by UIN version, then by product name.
7. Ranks products deterministically and selects up to 100.
8. Downloads the official documents.
9. Validates PDF signatures and computes SHA-256.
10. Stores resumable metadata in SQLite.
11. Never overwrites a file whose content hash differs.
12. Writes failures to a retryable queue.

## Important selection rule

IRDAI does NOT publish an official "top 100 health plans" ranking. This project therefore does not pretend one exists.

The insurer list is explicitly configurable in `config.yaml`. The default list is a practical major-health-insurer starting set, not an IRDAI ranking. Review/change it before a production corpus run.

For a defensible production ranking, replace `target_insurers` with your approved list and document the ranking methodology.

## Why identity comes from the UIN, not the insurer/type columns

IRDAI's free-text columns are too dirty to filter on:

- **`Name of the Insurer`** has **266 distinct spellings for ~30 companies**,
  including typos (`Insuarnce`, `Insurrance`, `Cholamanandlam`, `Royal
  Sundram`, `Helath`). Substring matching returned **zero** rows for Star
  Health, Care Health and ManipalCigna.
- **`Type Of Product`** is **blank on 689** of 1,819 rows and says
  `Revision` on **527**. A "Revision" is a re-filed retail product, so a
  `[Individual, Retail, New]` whitelist discards most of the real universe.

The UIN is structured and stable:

```text
BAJHLIP23020V012223
BAJ  insurer code   (chars 1-3)
HL   category       (HL health, PA accident, TI/TG travel, HM micro)
IP   policy type    (IP individual, GP group, IA/GA/AP add-on & rider)
23020 serial   V01 version   2223 financial year
```

`config.yaml` therefore keys `target_insurers` on `codes`, including
predecessor entities whose book of business the canonical insurer now
carries (Max Bupa -> Niva Bupa, Religare -> Care, Apollo Munich / HDFC ERGO
Health -> HDFC ERGO, CignaTTK -> ManipalCigna). `aliases` are a fallback for
legacy `IRDA/NL-HLT/...` UINs that carry no 3-char code.

## Known upstream data defects

These are IRDAI's, not the scraper's. Each is detected and reported rather
than silently absorbed:

- **1 catalogue row has no UIN at all** (ICICI Lombard "Group Criti Shield
  Plus", FY2022-23). It cannot be keyed, so it is counted and logged.
- **At least one row carries the wrong UIN.** `ADIHLIP18076V011718` is
  listed with an individual `IP` UIN, but its document is
  `ABHI_GroupActiveSecure_2017-2018.pdf` and the PDF's own internal UIN
  reads `IRDAI/HLT/ABHI/P-G/...` (`P-G` = group). Because the catalogue's
  UIN is itself wrong, the UIN filter cannot see it;
  `reject_document_filename_patterns` cross-checks the document filename,
  which is an independent signal, and drops it.
- **Products are re-filed under new UIN serials** in later financial years,
  so the same product appears more than once (e.g. "Arogya Plus Policy" as
  both `SBIHLIP21334V02` and `SBIHLIP22135V03`). Only the most recently
  approved filing is kept.

## Catalogue freshness (read before building a corpus)

The newest financial year in this IRDAI listing is **2022-2023** (25 rows);
there is nothing for FY2023-24 onward, and every row is `Non-Archived`, so
`include_archived` is effectively a no-op for this URL.

**A complete scrape of this page does not give you currently-sold products.**
Validate against insurer-issued wording/CIS and effective dates before using
the corpus for anything advice-shaped.

## Pagination

The portal's `Page 1 of 91` label is at its **default** page size of 20. At
`page_size: 60` there are 31 pages. Two traps this scraper handles:

- The `delta` parameter must be applied to **page 1 too**. Loading the bare
  URL for page 1 yields records 1-20 while page 2 at `delta=60` starts at
  record 61 -- silently losing records 21-60 (39 unique UINs, all of them
  the newest FY2021-22/FY2022-23 retail filings).
- Liferay **clamps** an out-of-range `cur` to the last page and re-serves
  it, so looping to 91 re-fetches page 31 sixty times. The loop stops when
  the result window stops advancing.

## Run

### 1. Install

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
playwright install chromium
```

### 2. Catalogue only (recommended first run)

```bash
python -m src.scraper catalogue
```

This creates `data/catalogue.csv` and `data/catalogue.sqlite3`.

### 3. Select the 100 products

```bash
python -m src.scraper select
```

This creates `data/selected_100.csv`.

### 4. Download

```bash
python -m src.scraper download
```

### 5. Retry failures

```bash
python -m src.scraper retry
```

## Production safeguards

- Coverage gate: scrape is reconciled against the portal's reported total.
- Respectful concurrency: default 2.
- Exponential backoff.
- Request timeout.
- Browser context with a normal user-agent.
- PDF magic-byte validation (`%PDF`).
- SHA-256 content hashing.
- SQLite uniqueness on UIN/document hash.
- Atomic `.part` downloads followed by rename.
- Existing identical files are skipped.
- Changed content is preserved as a new version.
- Structured logs.
- No hard-coded PDF URLs.

## Output

```text
data/
  catalogue.sqlite3
  catalogue.csv
  selected_100.csv
  downloads.csv
  failed_downloads.csv
  policies/
    <insurer_slug>/
      <uin>/
        <document>.pdf
        metadata.json
```

Insurer directories use the **canonical** name, so one company gets one
directory instead of one per IRDAI spelling variant. Each `metadata.json`
keeps `insurer_as_published` alongside it for provenance.

## Legal/operational note

Only use publicly accessible documents and respect IRDAI/insurer website terms, robots policies, rate limits and applicable law. This tool is for building a research/knowledge corpus; it does not establish that a product is currently sold merely because an IRDAI record is non-archived. For production insurance advice, validate current insurer-issued policy wording/CIS and effective dates as a second source.
