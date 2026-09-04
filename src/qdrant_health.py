"""Qdrant cluster health + filterability check.

Read-only. Writes nothing to the cluster.

Credentials come from the environment so no key is ever committed:

    export QDRANT_URL=https://<cluster>.cloud.qdrant.io
    export QDRANT_API_KEY=<key>
    python -m src.qdrant_health

Why this exists: Qdrant Cloud runs with strict mode
(`unindexed_filtering_retrieve=False`), which means a filter on a payload
field that has no index does not fall back to a scan -- it returns HTTP 400.
So a collection can pass a naive "can I connect / can I search" check and
still fail every real query that filters by insurer, plan or scope. This
script probes each field the application actually filters on.
"""
from __future__ import annotations

import os
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# Fields the application filters on, per collection, with a representative
# value type. Type matters: a keyword index will not serve a bool match, and
# probing a bool field with a string produces a misleading failure.
PROBES: dict[str, list[tuple[str, object]]] = {
    "policy_units_v2": [
        ("insurer_name", "x"), ("plan_name", "x"), ("section_id", "x"),
        ("coverage_scope", "x"), ("plan_type_key", "x"), ("plan_type", "x"),
        ("coverage_scope_confidence", "x"), ("source_pdf", "x"),
    ],
    "policy_section_updated": [
        ("insurer_name", "x"), ("plan_name", "x"), ("section_id", "x"),
        ("plan_type", "x"),
    ],
    "policy_sections": [
        ("plan_name", "x"), ("section_id", "x"), ("plan_type_key", "x"),
    ],
    "migration_payload": [
        ("insurer_name", "x"), ("plan_name", "x"), ("coverage_scope", "x"),
        ("has_maternity_cover", True), ("has_copay", True),
        ("cover_type_individual", True), ("ped_waiting_period_months", 0),
    ],
    "faq_collection": [("type", "x"), ("source", "x")],
}


def probe(client: QdrantClient, coll: str, field: str, value: object) -> str:
    """Return OK or the reason a filter on `field` cannot be served."""
    cond = (
        FieldCondition(key=field, range=Range(gte=value))
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else FieldCondition(key=field, match=MatchValue(value=value))
    )
    try:
        client.count(collection_name=coll, exact=True,
                     count_filter=Filter(must=[cond]))
        return "OK"
    except Exception as exc:
        msg = str(exc)
        if "Index required but not found" in msg:
            return "NO INDEX -- filter returns 400"
        if '"error":"' in msg:
            return msg.split('"error":"')[-1].split('"')[0][:70]
        return type(exc).__name__


def main() -> int:
    url, key = os.environ.get("QDRANT_URL"), os.environ.get("QDRANT_API_KEY")
    if not url or not key:
        print("Set QDRANT_URL and QDRANT_API_KEY in the environment.")
        return 2

    print(f"endpoint: {url}")
    # check_compatibility is off deliberately: the installed client may be a
    # minor version ahead of the managed server, which is only a warning.
    client = QdrantClient(url=url, api_key=key, timeout=60,
                          check_compatibility=False)

    try:
        present = {c.name for c in client.get_collections().collections}
    except Exception as exc:
        print(f"CONNECT FAILED: {type(exc).__name__}: {exc}")
        return 1
    print(f"connected -- {len(present)} collection(s)\n")

    failures = 0
    for coll, fields in PROBES.items():
        if coll not in present:
            print(f"{coll}: ABSENT from cluster\n")
            failures += 1
            continue

        info = client.get_collection(coll)
        indexed = sorted((info.payload_schema or {}).keys())
        print(f"{coll}  points={info.points_count} status={info.status}")
        print(f"  indexed fields ({len(indexed)}): "
              f"{', '.join(indexed) if indexed else 'NONE'}")
        for field, value in fields:
            verdict = probe(client, coll, field, value)
            mark = "  " if verdict == "OK" else "!!"
            if verdict != "OK":
                failures += 1
            print(f"  {mark} {field:<28} {verdict}")
        print()

    print(f"{'FAILURES: ' + str(failures) if failures else 'ALL FILTER PATHS OK'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
