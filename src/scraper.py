from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import yaml
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from tenacity import retry, stop_after_attempt, wait_exponential


ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

DATA_DIR = ROOT / CFG["paths"]["data_dir"]
POLICY_DIR = ROOT / CFG["paths"]["policy_dir"]
DATA_DIR.mkdir(parents=True, exist_ok=True)
POLICY_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("irdai-scraper")


@dataclass
class Product:
    archive_status: str
    financial_year: str
    insurer: str
    uin: str
    product_name: str
    approval_date: str
    product_type: str
    document_url: str
    document_text: str = ""
    # Derived from the UIN, which is far more reliable than IRDAI's
    # free-text insurer/product-type columns. See parse_uin().
    insurer_code: str = ""
    uin_category: str = ""
    uin_type: str = ""
    uin_serial: str = ""
    uin_version: str = ""


def norm(s) -> str:
    # Coerce non-str input: pandas types numeric-looking CSV columns
    # (uin_serial, uin_version) as int/float, and norm() is called on them.
    if s is None:
        return ""
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = s.replace("\xa0", " ")
    # IRDAI serves a few cells with mis-encoded bytes that decode to U+FFFD
    # and to smart punctuation. Product names become corpus metadata, so
    # fold them to ASCII equivalents rather than shipping mojibake.
    s = s.replace("�", "").replace("–", "-").replace("—", "-")
    s = s.replace("‘", "'").replace("’", "'")
    s = s.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip()


def product_key(name: str) -> str:
    """Normalised product name used to collapse re-filings of one product.

    IRDAI issues a NEW UIN serial when a product is re-filed in a later
    financial year (e.g. "Arogya Plus Policy" is SBIHLIP21334V02 in FY21 and
    SBIHLIP22135V03 in FY22), so UIN-version dedup alone leaves superseded
    wordings in the corpus.
    """
    n = norm(name).lower()
    n = re.sub(r"[^a-z0-9]+", " ", n)
    n = re.sub(r"\b(policy|plan|insurance|cover|the|a)\b", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:100]


# IRDAI UIN grammar: <INSURER:3><CATEGORY:0-5><TYPE:2><SERIAL:3-6>V<VER:2><FY:4>
# e.g. BAJHLIP23020V012223 -> BAJ | HL | IP | 23020 | 01 | 2223
# Older filings vary (1-3 char category, truncated version), and FY2013-14
# rows use a completely different "IRDA/NL-HLT/<INSURER>/P-H/..." form.
UIN_TYPES = ("IP", "GP", "IA", "GA", "IO", "OP", "DP", "BP", "SP",
             "AP", "SA", "OA", "GT", "BA")
UIN_RE = re.compile(
    r"^([A-Z]{3})([A-Z]{0,5}?)(" + "|".join(UIN_TYPES) + r")?"
    r"(\d{3,6})(?:V(\d{1,2})(\d{2,4})?)?"
)


def parse_uin(uin: str) -> dict:
    """Split an IRDAI UIN into its structural parts.

    Returns empty strings for anything that cannot be determined; callers
    fall back to the free-text columns in that case.
    """
    u = re.sub(r"[^A-Za-z0-9/.-]", "", (uin or "")).upper()
    blank = {"insurer_code": "", "uin_category": "", "uin_type": "",
             "uin_serial": "", "uin_version": ""}

    # Legacy IRDA/NL-HLT/<INSURER>/P-H/V.I/<n>/13-14 form carries no
    # 3-char insurer code, so leave resolution to the alias fallback.
    if u.startswith("IRDA") or "/" in u:
        return dict(blank, uin_category="LEGACY")

    m = UIN_RE.match(u)
    if not m:
        return dict(blank)
    return {
        "insurer_code": m.group(1),
        "uin_category": m.group(2) or "",
        "uin_type": m.group(3) or "",
        "uin_serial": m.group(4) or "",
        "uin_version": m.group(5) or "",
    }


def parse_approval_date(s: str) -> str:
    """IRDAI prints DD-MM-YYYY. Return ISO YYYY-MM-DD so it sorts correctly.

    The previous code sorted the raw DD-MM-YYYY string, which ranked by
    day-of-month before year.
    """
    s = norm(s)
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%y", "%d.%m.%Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    m = re.search(r"(20\d{2})", s)
    return f"{m.group(1)}-01-01" if m else ""


PRODUCT_COLUMNS = [
    "uin", "archive_status", "financial_year", "insurer", "product_name",
    "approval_date", "approval_date_iso", "product_type", "document_url",
    "document_text", "insurer_code", "uin_category", "uin_type",
    "uin_serial", "uin_version", "scraped_at",
]


def init_db() -> sqlite3.Connection:
    db = sqlite3.connect(DATA_DIR / "catalogue.sqlite3")
    db.execute("""
    CREATE TABLE IF NOT EXISTS products (
        uin TEXT PRIMARY KEY,
        archive_status TEXT,
        financial_year TEXT,
        insurer TEXT,
        product_name TEXT,
        approval_date TEXT,
        product_type TEXT,
        document_url TEXT,
        document_text TEXT,
        scraped_at TEXT NOT NULL
    )
    """)

    # Migrate older databases in place rather than forcing a re-scrape.
    existing = {r[1] for r in db.execute("PRAGMA table_info(products)")}
    for col in PRODUCT_COLUMNS:
        if col not in existing:
            db.execute(f"ALTER TABLE products ADD COLUMN {col} TEXT")

    db.execute("""
    CREATE TABLE IF NOT EXISTS downloads (
        uin TEXT,
        url TEXT,
        path TEXT,
        sha256 TEXT,
        bytes INTEGER,
        downloaded_at TEXT,
        status TEXT,
        error TEXT,
        PRIMARY KEY (uin, url)
    )
    """)
    db.commit()
    return db


async def extract_table(page, stats: dict | None = None) -> list[Product]:
    products = []
    stats = stats if stats is not None else {}
    # IRDAI renders the catalogue as an HTML table. We deliberately parse
    # semantic table content instead of relying on brittle CSS class names.
    tables = page.locator("table")
    count = await tables.count()

    for ti in range(count):
        table = tables.nth(ti)
        rows = table.locator("tr")
        if await rows.count() < 2:
            continue

        header_cells = rows.nth(0).locator("th,td")
        headers = [norm(await header_cells.nth(i).inner_text()) for i in range(await header_cells.count())]
        joined = " | ".join(h.lower() for h in headers)

        if "uin" not in joined or "product name" not in joined:
            continue

        for ri in range(1, await rows.count()):
            row = rows.nth(ri)
            cells = row.locator("td,th")
            n = await cells.count()
            if n < 6:
                continue

            vals = [norm(await cells.nth(i).inner_text()) for i in range(n)]

            # Find columns by header text rather than assuming fixed indexes.
            idx = {h.lower(): i for i, h in enumerate(headers)}

            def get(*names):
                for name in names:
                    for h, i in idx.items():
                        if name in h:
                            return vals[i] if i < len(vals) else ""
                return ""

            uin = get("uin")
            product_name = get("product name")
            insurer = get("name of the insurer", "insurer")
            fy = get("financial year")
            approval = get("date of approval", "approval")
            ptype = get("type of product", "product type")
            archive = get("archive / non archive", "archive")

            if not uin or not product_name:
                # IRDAI itself publishes rows with no UIN (e.g. ICICI
                # Lombard "Group Criti Shield Plus", FY2022-23). They cannot
                # be keyed or deduplicated, so count them and move on rather
                # than reporting the scrape as short.
                if any(v for v in vals):
                    stats["skipped_no_uin"] = stats.get("skipped_no_uin", 0) + 1
                    stats.setdefault("skipped_rows", []).append(
                        {"insurer": insurer, "product_name": product_name,
                         "financial_year": fy, "uin": uin}
                    )
                continue

            # The document column can contain an anchor to a PDF or a
            # document-detail page. We keep the href and resolve it later.
            anchors = row.locator("a[href]")
            document_url = ""
            document_text = ""
            for ai in range(await anchors.count()):
                a = anchors.nth(ai)
                href = await a.get_attribute("href")
                text = norm(await a.inner_text())
                if not href:
                    continue
                if ".pdf" in href.lower() or ".pdf" in text.lower() or "document" in href.lower():
                    document_url = urljoin(CFG["source"]["url"], href)
                    document_text = text
                    break

            products.append(Product(
                archive_status=archive,
                financial_year=fy,
                insurer=insurer,
                uin=uin,
                product_name=product_name,
                approval_date=approval,
                product_type=ptype,
                document_url=document_url,
                document_text=document_text,
                **parse_uin(uin),
            ))
        break

    return products


PORTLET = "com_irdai_document_media_IRDAIDocumentMediaPortlet"
SHOWING_RE = re.compile(
    r"Showing\s+([\d,]+)\s*-\s*([\d,]+)\s+of\s+([\d,]+)\s+results", re.I
)


def paginated_url(page_no: int) -> str:
    """Build the Liferay search-container URL for a page.

    This is applied to page 1 too. Previously page 1 loaded the bare URL,
    which uses IRDAI's *default* delta of 20 -- so page 1 returned records
    1-20 while page 2 (delta=60) started at record 61. Records 21-60 were
    never requested and 39 unique UINs were silently lost, all of them the
    newest FY2021-22/FY2022-23 retail filings.
    """
    return (
        CFG["source"]["url"]
        + f"?_{PORTLET}_cur={page_no}"
        + f"&_{PORTLET}_delta={CFG['source']['page_size']}"
        + f"&p_p_id={PORTLET}"
        + "&p_p_lifecycle=0&p_p_state=normal&p_p_mode=view"
    )


async def read_showing(page) -> tuple[int, int, int] | None:
    """Parse the portal's own 'Showing X - Y of N results' counter."""
    try:
        body = norm(await page.locator("body").inner_text())
    except PlaywrightTimeoutError:
        return None
    m = SHOWING_RE.search(body)
    if not m:
        return None
    return tuple(int(g.replace(",", "")) for g in m.groups())


async def catalogue() -> None:
    db = init_db()
    all_products: dict[str, Product] = {}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=CFG["runtime"]["headless"])
        context = await browser.new_context(
            user_agent=CFG["runtime"]["user_agent"],
            accept_downloads=True,
        )
        page = await context.new_page()
        page.set_default_timeout(CFG["source"]["navigation_timeout_ms"])

        delta = CFG["source"]["page_size"]
        stats: dict = {}
        expected_total = 0
        rows_seen = 0
        empty_pages: list[int] = []
        page_no = 1

        while page_no <= CFG["source"]["max_pages"]:
            await page.goto(paginated_url(page_no), wait_until="domcontentloaded")
            try:
                await page.wait_for_load_state("networkidle")
            except PlaywrightTimeoutError:
                pass

            showing = await read_showing(page)
            rows = await extract_table(page, stats)

            if showing:
                first, last, total = showing
                if not expected_total:
                    expected_total = total
                    log.info(
                        "IRDAI reports %s records; %s pages at delta=%s",
                        total, -(-total // delta), delta
                    )
                # Liferay CLAMPS an out-of-range `cur` to the last page and
                # re-serves it, so a naive loop over the default-delta page
                # count (91) silently re-fetches page 31 sixty times. Stop
                # as soon as the window no longer advances.
                if last >= total and page_no > 1:
                    log.info("page=%s rows=%s (final window %s-%s of %s)",
                             page_no, len(rows), first, last, total)
                    for item in rows:
                        old = all_products.get(item.uin)
                        if old is None or (not old.document_url and item.document_url):
                            all_products[item.uin] = item
                    rows_seen += len(rows)
                    break
                log.info("page=%s rows=%s (%s-%s of %s)",
                         page_no, len(rows), first, last, total)
            else:
                log.warning("page=%s rows=%s (no result counter found)",
                            page_no, len(rows))

            if not rows:
                # Transient render/timeout failure, not the end of the
                # catalogue. Retry once before giving up on the page.
                empty_pages.append(page_no)
                log.warning("page=%s returned 0 rows; retrying once", page_no)
                await asyncio.sleep(2)
                await page.goto(paginated_url(page_no), wait_until="domcontentloaded")
                try:
                    await page.wait_for_load_state("networkidle")
                except PlaywrightTimeoutError:
                    pass
                rows = await extract_table(page, stats)
                if rows:
                    empty_pages.pop()
                    log.info("page=%s recovered with %s rows", page_no, len(rows))

            for item in rows:
                # Keep the first occurrence unless the later row has a
                # usable document URL.
                old = all_products.get(item.uin)
                if old is None or (not old.document_url and item.document_url):
                    all_products[item.uin] = item

            rows_seen += len(rows)
            if expected_total and rows_seen >= expected_total:
                break

            page_no += 1
            await asyncio.sleep(CFG["runtime"].get("delay_ms_between_pages", 400) / 1000)

        await context.close()
        await browser.close()

    # Completeness gate: the portal tells us how many rows exist, so a
    # short scrape is a hard signal, not something to discover later.
    no_uin = stats.get("skipped_no_uin", 0)
    if expected_total:
        accounted = rows_seen + no_uin
        if no_uin:
            log.warning(
                "%s catalogue row(s) published by IRDAI with no UIN were "
                "skipped (unkeyable): %s",
                no_uin,
                "; ".join(
                    f"{r['insurer']} / {r['product_name']} ({r['financial_year']})"
                    for r in stats.get("skipped_rows", [])
                ),
            )
        if accounted < expected_total:
            log.error(
                "INCOMPLETE: accounted for %s of %s catalogue rows "
                "(%s scraped + %s unkeyable); missing %s. Empty pages: %s",
                accounted, expected_total, rows_seen, no_uin,
                expected_total - accounted, empty_pages or "none",
            )
        else:
            log.info(
                "Coverage OK: %s of %s catalogue rows accounted for "
                "(%s scraped + %s unkeyable)",
                accounted, expected_total, rows_seen, no_uin,
            )
    if empty_pages:
        log.error("Pages that never yielded rows: %s", empty_pages)

    now = datetime.now(timezone.utc).isoformat()
    cols = ",".join(PRODUCT_COLUMNS)
    marks = ",".join("?" * len(PRODUCT_COLUMNS))
    updates = ",".join(f"{c}=excluded.{c}" for c in PRODUCT_COLUMNS if c != "uin")
    for pdt in all_products.values():
        db.execute(
            f"INSERT INTO products ({cols}) VALUES ({marks}) "
            f"ON CONFLICT(uin) DO UPDATE SET {updates}",
            (
                pdt.uin, pdt.archive_status, pdt.financial_year, pdt.insurer,
                pdt.product_name, pdt.approval_date,
                parse_approval_date(pdt.approval_date), pdt.product_type,
                pdt.document_url, pdt.document_text, pdt.insurer_code,
                pdt.uin_category, pdt.uin_type, pdt.uin_serial,
                pdt.uin_version, now,
            ),
        )
    db.commit()

    df = pd.read_sql_query(
        f"SELECT {cols} FROM products ORDER BY insurer, product_name", db
    )
    df.to_csv(DATA_DIR / "catalogue.csv", index=False)
    log.info("Saved %s unique UIN records (%s catalogue rows scraped)",
             len(df), rows_seen)
    db.close()


def document_filename(url: str) -> str:
    """Return the *.pdf path segment of a Liferay document URL.

    Liferay puts the real filename mid-path and appends a UUID, so the last
    path segment is the UUID, not the filename:
      /documents/37343/931203/ABHI_GroupActiveSecure_2017-2018.pdf/<uuid>?...
    """
    for seg in norm(url).split("?")[0].split("/"):
        if seg.lower().endswith(".pdf"):
            return seg
    return ""


def resolve_insurer(insurer_code: str, insurer_text: str) -> str:
    """Map a catalogue row to one of the configured canonical insurers.

    The UIN insurer code is authoritative (it covers 1,751 of 1,819 rows and
    is immune to IRDAI's 266 free-text spellings). The lowercase alias
    fallback exists only for legacy 'IRDA/NL-HLT/...' UINs that carry no
    code. Returns "" when the row is not a target insurer.
    """
    code = norm(insurer_code).upper()
    text = norm(insurer_text).lower()
    for target in CFG["selection"]["target_insurers"]:
        if code and code in {c.upper() for c in target.get("codes", [])}:
            return target["canonical"]
    for target in CFG["selection"]["target_insurers"]:
        if any(a.lower() in text for a in target.get("aliases", [])):
            return target["canonical"]
    return ""


def score_product(row) -> int:
    """Auditable score. Recency is bounded so it cannot swamp policy signals.

    The previous version added approval_year * 2 (~4,044) on top of weights
    of 100/30/25/10, which meant approval year alone decided the ranking.
    """
    cfg = CFG["selection"]["score"]
    score = 0
    if norm(row.uin_type).upper() in {
        t.upper() for t in CFG["selection"]["uin_type"]["accepted"]
    }:
        score += cfg["individual_policy"]
    if norm(row.uin_category).upper() in {
        c.upper() for c in CFG["selection"]["uin_category"]["accepted"]
    }:
        score += cfg["health_category"]
    if getattr(row, "is_latest_version", False):
        score += cfg["latest_version"]
    if norm(row.document_url):
        score += cfg["has_document"]

    year = norm(row.approval_date_iso)[:4]
    if year.isdigit():
        score += max(0, int(year) - cfg["recency_baseline"]) * cfg["recency_per_year"]
    return score


def select_products() -> None:
    db = init_db()
    df = pd.read_sql_query("SELECT * FROM products", db)

    if df.empty:
        raise SystemExit("Catalogue is empty. Run: python -m src.scraper catalogue")

    sel = CFG["selection"]
    df = df.fillna("")
    stages = [("catalogue", len(df))]

    if not sel["include_archived"]:
        df = df[df.archive_status.str.lower().str.contains("non-archived", na=False)]
    stages.append(("non-archived", len(df)))

    # Canonical insurer via UIN code, with alias fallback for legacy UINs.
    df = df.copy()
    df["canonical_insurer"] = [
        resolve_insurer(c, t) for c, t in zip(df.insurer_code, df.insurer)
    ]
    df = df[df.canonical_insurer != ""]
    stages.append(("target insurers", len(df)))

    # Individual/retail policy forms only (UIN chars 6-7).
    accepted_type = {t.upper() for t in sel["uin_type"]["accepted"]}
    df = df[df.uin_type.str.strip().str.upper().isin(accepted_type)]
    stages.append(("individual policy (UIN type)", len(df)))

    # Health indemnity categories only (UIN chars 4-5).
    accepted_cat = {c.upper() for c in sel["uin_category"]["accepted"]}
    df = df[df.uin_category.str.strip().str.upper().isin(accepted_cat)]
    stages.append(("health category (UIN cat)", len(df)))

    # Free-text type used only as a negative signal -- it is blank on 689
    # rows and says "Revision" on 527, so it cannot be used as a whitelist.
    # The product NAME is checked too: some add-on covers are filed with an
    # individual-policy UIN (e.g. RHIHLIP21407V022021 "Domestic Staff
    # Insurance Add-on"), so the UIN alone does not exclude them.
    reject = "|".join(re.escape(p) for p in sel["reject_product_type_patterns"])
    df = df[
        ~df.product_type.str.lower().str.contains(reject, na=False, regex=True)
        & ~df.product_name.str.lower().str.contains(reject, na=False, regex=True)
    ]
    stages.append(("not rider/add-on", len(df)))

    df = df[df.document_url.str.strip() != ""]
    stages.append(("has document URL", len(df)))

    # Cross-check against IRDAI's own document filename, which is an
    # independent signal from the catalogue row and catches rows IRDAI
    # mislabelled (see reject_document_filename_patterns in config.yaml).
    doc_reject = sel.get("reject_document_filename_patterns") or []
    if doc_reject:
        pattern = "|".join(re.escape(p) for p in doc_reject)
        fname = df.document_url.map(document_filename).str.lower()
        dropped = df[fname.str.contains(pattern, na=False, regex=True)]
        for _, r in dropped.iterrows():
            log.warning(
                "dropping %s (%s): IRDAI document filename %r contradicts "
                "its individual-policy UIN",
                r.uin, norm(r.product_name), document_filename(r.document_url),
            )
        df = df[~fname.str.contains(pattern, na=False, regex=True)]
    stages.append(("document filename agrees", len(df)))

    # One UIN = one product record.
    df = df.drop_duplicates(subset=["uin"]).copy()

    # Keep only the newest version of each product line. A line is
    # insurer+category+type+serial; version lives in UIN "Vnn".
    df["line_key"] = (df.insurer_code + "|" + df.uin_category + "|"
                      + df.uin_type + "|" + df.uin_serial)
    df["version_num"] = pd.to_numeric(df.uin_version, errors="coerce").fillna(0)
    newest = df.groupby("line_key")["version_num"].transform("max")
    df["is_latest_version"] = df.version_num >= newest
    if sel["latest_version_only"]:
        df = df[df.is_latest_version].drop_duplicates(subset=["line_key"]).copy()
    stages.append(("latest version only", len(df)))

    # Collapse re-filings of the same product to the most recent approval so
    # the corpus never holds a superseded wording next to the current one.
    if sel.get("dedupe_by_product_name", True):
        df["product_key"] = df.product_name.map(product_key)
        df = (
            df.sort_values(
                ["approval_date_iso", "version_num", "uin"],
                ascending=[False, False, True],
            )
            .drop_duplicates(subset=["canonical_insurer", "product_key"])
            .copy()
        )
    stages.append(("one filing per product", len(df)))

    log.info("Selection funnel:")
    for name, n in stages:
        log.info("    %-32s %s", name, n)

    df["score"] = df.apply(score_product, axis=1)

    # Balanced selection: round-robin across insurers after deterministic
    # within-insurer ranking. This prevents one insurer from consuming all 100.
    buckets = {}
    for insurer, g in df.groupby("canonical_insurer"):
        buckets[insurer] = g.sort_values(
            ["score", "approval_date_iso", "product_name"],
            ascending=[False, False, True],
        ).to_dict("records")

    selected = []
    while len(selected) < sel["max_products"] and buckets:
        changed = False
        for insurer in list(buckets):
            if buckets[insurer]:
                selected.append(buckets[insurer].pop(0))
                changed = True
                if len(selected) >= sel["max_products"]:
                    break
            else:
                buckets.pop(insurer)
        if not changed:
            break

    out = pd.DataFrame(selected)
    out.to_csv(DATA_DIR / "selected_100.csv", index=False)
    log.info("Selected %s products (target %s)", len(out), sel["max_products"])
    if len(out) < sel["max_products"]:
        log.warning(
            "Only %s of %s requested products are available under the current "
            "filters. Widen selection.uin_type / uin_category or add insurers.",
            len(out), sel["max_products"],
        )
    for insurer, n in sorted(out.canonical_insurer.value_counts().items()):
        log.info("    %-52s %s", insurer[:52], n)
    db.close()


@retry(stop=stop_after_attempt(CFG["runtime"]["retries"]),
       wait=wait_exponential(multiplier=1, min=2, max=20))
async def download_one(page, row: dict) -> tuple[str, bytes]:
    url = row["document_url"]
    if not url:
        raise RuntimeError("No document URL in IRDAI catalogue row")

    response = await page.request.get(
        url,
        timeout=CFG["source"]["download_timeout_ms"],
        fail_on_status_code=True,
    )
    content = await response.body()

    # Direct PDF.
    if content.startswith(b"%PDF"):
        return url, content

    # Document-detail page: find the first PDF link.
    html = content.decode("utf-8", errors="ignore")
    matches = re.findall(r'href=["\']([^"\']+\.pdf[^"\']*)["\']', html, flags=re.I)
    if not matches:
        # Some portals expose a document viewer endpoint. Try anchors with
        # pdf-like text as a second path.
        matches = re.findall(
            r'href=["\']([^"\']+)["\'][^>]*>[^<]*pdf[^<]*<',
            html, flags=re.I
        )
    if not matches:
        raise RuntimeError("IRDAI document page did not expose a PDF href")

    pdf_url = urljoin(url, matches[0])
    response = await page.request.get(
        pdf_url,
        timeout=CFG["source"]["download_timeout_ms"],
        fail_on_status_code=True,
    )
    content = await response.body()
    if not content.startswith(b"%PDF"):
        raise RuntimeError("Resolved document is not a valid PDF")
    return pdf_url, content


async def download_selected(retry_only=False) -> None:
    selected_path = DATA_DIR / "selected_100.csv"
    if not selected_path.exists():
        raise SystemExit("Run selection first: python -m src.scraper select")

    # dtype=str keeps UIN serial/version as written instead of letting
    # pandas turn them into ints (and "007" into 7).
    df = pd.read_csv(selected_path, dtype=str).fillna("")
    db = init_db()

    if retry_only:
        failed = pd.read_sql_query(
            "SELECT uin FROM downloads WHERE status='failed'", db
        )
        df = df[df.uin.isin(failed.uin)]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=CFG["runtime"]["headless"])
        context = await browser.new_context(user_agent=CFG["runtime"]["user_agent"])
        page = await context.new_page()

        for _, row in df.iterrows():
            uin = norm(row["uin"])
            insurer = norm(row["insurer"])
            # Group by canonical insurer so one company gets one directory
            # instead of one per IRDAI spelling variant.
            canonical = norm(row.get("canonical_insurer", "")) or insurer
            out_dir = POLICY_DIR / slug(canonical) / slug(uin)
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                pdf_url, content = await download_one(page, row.to_dict())
                sha = hashlib.sha256(content).hexdigest()
                filename = slug(row["product_name"]) or slug(uin)
                final_path = out_dir / f"{filename}_{sha[:12]}.pdf"
                part_path = final_path.with_suffix(".pdf.part")

                if not final_path.exists():
                    part_path.write_bytes(content)
                    part_path.replace(final_path)

                metadata = {
                    "uin": uin,
                    "insurer_as_published": insurer,
                    "insurer": canonical,
                    "insurer_code": norm(row.get("insurer_code", "")),
                    "product_name": norm(row["product_name"]),
                    "financial_year": norm(row["financial_year"]),
                    "approval_date": norm(row["approval_date"]),
                    "approval_date_iso": norm(row.get("approval_date_iso", "")),
                    "product_type_as_published": norm(row["product_type"]),
                    "uin_category": norm(row.get("uin_category", "")),
                    "uin_type": norm(row.get("uin_type", "")),
                    "uin_version": norm(row.get("uin_version", "")),
                    "archive_status": norm(row["archive_status"]),
                    "catalogue_document_url": norm(row["document_url"]),
                    "resolved_pdf_url": pdf_url,
                    "sha256": sha,
                    "bytes": len(content),
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                    "source": "IRDAI Health Insurance Products catalogue",
                    "source_catalogue_url": CFG["source"]["url"],
                }
                (out_dir / "metadata.json").write_text(
                    json.dumps(metadata, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                db.execute("""
                INSERT OR REPLACE INTO downloads
                VALUES (?,?,?,?,?,?,?,?)
                """, (
                    uin, pdf_url, str(final_path.relative_to(ROOT)),
                    sha, len(content), metadata["downloaded_at"], "success", ""
                ))
                db.commit()
                log.info("downloaded %s | %s", uin, final_path)
                await asyncio.sleep(CFG["runtime"]["delay_ms_between_documents"] / 1000)

            except Exception as exc:
                err = repr(exc)
                db.execute("""
                INSERT OR REPLACE INTO downloads
                VALUES (?,?,?,?,?,?,?,?)
                """, (
                    uin, row["document_url"], "", "", 0,
                    datetime.now(timezone.utc).isoformat(), "failed", err
                ))
                db.commit()
                log.exception("failed %s", uin)

        await context.close()
        await browser.close()

    downloads = pd.read_sql_query("SELECT * FROM downloads", db)
    downloads.to_csv(DATA_DIR / "downloads.csv", index=False)
    downloads[downloads.status == "failed"].to_csv(
        DATA_DIR / "failed_downloads.csv", index=False
    )
    db.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["catalogue", "select", "download", "retry"])
    args = parser.parse_args()

    if args.command == "catalogue":
        asyncio.run(catalogue())
    elif args.command == "select":
        select_products()
    elif args.command == "download":
        asyncio.run(download_selected(False))
    elif args.command == "retry":
        asyncio.run(download_selected(True))


if __name__ == "__main__":
    main()
