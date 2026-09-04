"""Normalisation for the known data-quality defects in the Qdrant corpus.

Every function here compensates for something wrong upstream. Each is
documented with the defect it works around so that when the data is fixed the
corresponding shim can be deleted rather than quietly outliving its purpose.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# DEFECT 1: `insurer_name` carries 16 spellings for ~11 companies, and the
# collections disagree with each other ("National" vs "National Insurance").
# `Activ` is not an insurer at all -- it is an Aditya Birla product line that
# leaked into the insurer field.
#
# Keys are matched against a punctuation-stripped, lowercased form of the raw
# value, longest key first, so "aditya birla health insurance" wins over
# "aditya birla".
# ---------------------------------------------------------------------------
_INSURER_CANON: dict[str, str] = {
    "aditya birla health insurance": "Aditya Birla Health Insurance",
    "aditya birla health": "Aditya Birla Health Insurance",
    "aditya birla": "Aditya Birla Health Insurance",
    "activ": "Aditya Birla Health Insurance",
    "star health": "Star Health and Allied Insurance",
    "national insurance": "National Insurance",
    "national": "National Insurance",
    "shriram general insurance": "Shriram General Insurance",
    "shriram general": "Shriram General Insurance",
    "universal sompo": "Universal Sompo General Insurance",
    "acko": "Acko General Insurance",
    "iffco tokio": "IFFCO-Tokio General Insurance",
    "iffcotokio": "IFFCO-Tokio General Insurance",
    "oriental insurance": "The Oriental Insurance Company",
    "oriental": "The Oriental Insurance Company",
    "united india": "United India Insurance",
    "galaxy health": "Galaxy Health Insurance",
    "galaxy": "Galaxy Health Insurance",
    "tata aig": "Tata AIG General Insurance",
    "tata": "Tata AIG General Insurance",
    "care health": "Care Health Insurance",
    "care": "Care Health Insurance",
    "niva bupa": "Niva Bupa Health Insurance",
    "max bupa": "Niva Bupa Health Insurance",
    "hdfc ergo": "HDFC ERGO General Insurance",
    "icici lombard": "ICICI Lombard General Insurance",
    "bajaj allianz": "Bajaj Allianz General Insurance",
    "sbi general": "SBI General Insurance",
    "manipalcigna": "ManipalCigna Health Insurance",
    "manipal cigna": "ManipalCigna Health Insurance",
}

_CANON_KEYS = sorted(_INSURER_CANON, key=len, reverse=True)


def canonical_names() -> set[str]:
    """Every canonical insurer name. Public so other modules need not reach
    into the private map."""
    return set(_INSURER_CANON.values())


def _fold(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def canonical_insurer(raw: str | None) -> str:
    """Map any observed insurer spelling to one canonical name.

    Returns the cleaned original when nothing matches, so a new insurer
    appearing in the corpus degrades to "shown as published" rather than
    vanishing from results.
    """
    folded = _fold(raw or "")
    if not folded:
        return "Unknown insurer"
    for key in _CANON_KEYS:
        if folded == key or folded.startswith(key + " ") or key in folded:
            return _INSURER_CANON[key]
    return (raw or "").strip()


def insurer_query_variants(canonical: str) -> list[str]:
    """All raw spellings that map to a canonical name.

    Needed because `insurer_name` *is* indexed on the cluster, so an insurer
    filter must enumerate every spelling or it silently drops records --
    filtering "Aditya Birla Health Insurance" alone returns 144 of 180.
    """
    target = canonical_insurer(canonical)
    return sorted({k for k, v in _INSURER_CANON.items() if v == target})


# ---------------------------------------------------------------------------
# DEFECT 2: numeric feature fields use -1 as "unknown" rather than null.
# `maternity_waiting_period_months` is -1 on 112 of 156 plans. A naive range
# filter for "under 24 months" matches all of them and reports plans whose
# waiting period is actually unknown as if it were zero.
# ---------------------------------------------------------------------------
SENTINEL = -1


def clean_number(value: object) -> int | float | None:
    """Return None for the -1 sentinel and for non-numeric input."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return None if value < 0 else value
    return None


def describe_number(value: object, unit: str = "") -> str:
    """Render a feature number for display, never inventing a zero.

    Sums insured run to crores, so plain %g would print "2e+07" and a reader
    (or the model) could misread the magnitude. Integers are grouped instead.
    """
    cleaned = clean_number(value)
    if cleaned is None:
        return "not stated in the source data"
    if isinstance(cleaned, float) and cleaned.is_integer():
        cleaned = int(cleaned)
    text = f"{cleaned:,}" if isinstance(cleaned, int) else f"{cleaned:,.2f}"
    return f"{text}{(' ' + unit) if unit else ''}"


# ---------------------------------------------------------------------------
# Page recovery. The corpus has no page field, but the ingest pipeline left
# "### PAGE n" markers inside the section text for 735 of 1,422 sections
# (51.7%). Where present they let a citation resolve to a page, which is what
# makes a citation checkable against the original PDF.
# ---------------------------------------------------------------------------
_PAGE_MARKER = re.compile(r"#+\s*PAGE\s+(\d+)", re.I)


def has_page_markers(text: str | None) -> bool:
    return bool(_PAGE_MARKER.search(text or ""))


def locate_page(section_text: str | None, quoted: str | None) -> int | None:
    """Page number for a quote, or None when it cannot be established.

    Finds the quote in the section text and walks back to the nearest preceding
    page marker. Returns None rather than guessing when the quote cannot be
    located or the section carries no markers -- a wrong page citation is worse
    than an absent one.
    """
    text, quote = section_text or "", (quoted or "").strip()
    if not text or not quote:
        return None

    idx = text.find(quote)
    if idx < 0:
        # The model may normalise whitespace in the quote it echoes back, so
        # retry against a whitespace-collapsed copy of the text while keeping
        # an exact index mapping. An earlier version estimated the offset
        # proportionally, which could name the wrong page -- worse than naming
        # none, since a citation is only useful if it is checkable.
        probe = re.sub(r"\s+", " ", quote)[:120]
        if not probe:
            return None
        flat_chars: list[str] = []
        offsets: list[int] = []
        prev_space = False
        for i, ch in enumerate(text):
            if ch.isspace():
                if prev_space:
                    continue
                flat_chars.append(" ")
                prev_space = True
            else:
                flat_chars.append(ch)
                prev_space = False
            offsets.append(i)
        pos = "".join(flat_chars).find(probe)
        if pos < 0 or pos >= len(offsets):
            return None
        idx = offsets[pos]

    last = None
    for m in _PAGE_MARKER.finditer(text):
        if m.start() > idx:
            break
        last = int(m.group(1))
    return last


# ---------------------------------------------------------------------------
# DEFECT 3: `coverage_scope` has no payload index, so Qdrant (strict mode)
# rejects any server-side filter on it. Group products therefore have to be
# removed after retrieval. 171 of 1,422 sections are group.
# ---------------------------------------------------------------------------
_GROUP_NAME_HINT = re.compile(r"\bgroup\b|\bgrp\b|\bemployer\b", re.I)


def is_group_product(payload: dict) -> bool:
    """True when a record looks like a group rather than retail product."""
    scope = (payload.get("coverage_scope") or "").strip().lower()
    if scope == "group":
        return True
    if scope == "individual":
        return False
    # scope is missing or "uncertain" -- fall back to the product name
    return bool(_GROUP_NAME_HINT.search(payload.get("plan_name") or ""))


# ---------------------------------------------------------------------------
# DEFECT 4: the COVID-era standard products remain in the corpus although IRDAI
# discontinued them. Corona Kavach and Corona Rakshak were withdrawn in 2022.
#
# An earlier version of this filter also excluded Arogya Sanjeevani and Saral
# Suraksha Bima as "withdrawn". That was wrong: both are IRDAI-mandated
# standard products that insurers are still required to offer. Excluding them
# silently hid 12 currently-sold plans, which is the same silent-loss failure
# this module exists to prevent. They are now classified separately.
# ---------------------------------------------------------------------------
_WITHDRAWN = re.compile(r"corona\s*kavach|corona\s*rakshak", re.I)

# Still sold, but standardised across insurers -- identical wording everywhere,
# so they add little to a comparison and can crowd out differentiated products.
_MANDATED_STANDARD = re.compile(
    r"arogya\s*sanjeevani|saral\s*suraksha", re.I
)


def is_withdrawn_product(payload: dict) -> bool:
    """True only for products IRDAI has actually discontinued."""
    return bool(_WITHDRAWN.search(payload.get("plan_name") or ""))


def is_mandated_standard_product(payload: dict) -> bool:
    """True for still-sold IRDAI standard products with identical wording."""
    return bool(_MANDATED_STANDARD.search(payload.get("plan_name") or ""))


# ---------------------------------------------------------------------------
# DEFECT 5: some payload text was decoded from cp1252 as UTF-8, leaving
# U+FFFD where bullet and dash characters belong (visible in faq_collection).
# ---------------------------------------------------------------------------
def clean_text(s: str | None) -> str:
    if not s:
        return ""
    s = s.replace("�", " ").replace("\xa0", " ")
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("‘", "'").replace("’", "'")
    s = s.replace("“", '"').replace("”", '"')
    return re.sub(r"[ \t]{2,}", " ", s).strip()


def plan_key(payload: dict) -> tuple[str, str]:
    """Stable identity for a plan across the corpus's naming inconsistencies."""
    return (
        canonical_insurer(payload.get("insurer_name")),
        clean_text(payload.get("plan_name")) or "Unknown plan",
    )
