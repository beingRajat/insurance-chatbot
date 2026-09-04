"""Operational CLI for the policy answer service.

    python -m src.rag.cli health
    python -m src.rag.cli verify-embeddings
    python -m src.rag.cli plans
    python -m src.rag.cli plan-ask "What is the maternity waiting period?" --plan "Activ One Max+"
    python -m src.rag.cli ask "Which plans cover bariatric surgery?"
    python -m src.rag.cli features has_maternity_cover has_opd_cover
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from .config import ConfigError, get_settings, load_settings
from .service import RagService


def _print_result(result) -> int:
    if result.answer is None:
        print(f"\nNO ANSWER: {result.message}\n")
        print(json.dumps(result.trace.__dict__, indent=2, default=str))
        return 1
    a, t = result.answer, result.trace
    print("\n" + "=" * 72)
    print(a.text)
    print("=" * 72)
    if a.refusal:
        print(f"\nREFUSED: {a.refusal}")
        return 1
    print(f"\ngrounded: {a.is_grounded}  ({len(a.citations)} citations)")
    for c in a.citations[:12]:
        quote = c.text.strip().replace("\n", " ")
        if len(quote) > 110:
            quote = quote[:110] + "..."
        print(f"  - [{c.document_title}]\n      \"{quote}\"")
    if len(a.citations) > 12:
        print(f"  ... and {len(a.citations) - 12} more")
    print(f"\nplans considered : {', '.join(t.plans_considered) or '-'}")
    print(f"sections used    : {len(t.sections_used)} ({', '.join(t.sections_used)})")
    print(f"dropped          : group={t.dropped_group} withdrawn={t.dropped_withdrawn} "
          f"low_score={t.dropped_low_score}")
    if t.document_read:
        print(f"document read    : {t.document_read}")
    if t.alternative_documents:
        print(f"OTHER DOCS IGNORED: {', '.join(t.alternative_documents)}")
    print(f"latency          : retrieval {t.retrieval_ms}ms + answer {t.answer_ms}ms")
    print(f"tokens           : in={a.input_tokens} out={a.output_tokens} "
          f"cache_read={a.cache_read_tokens} cache_write={a.cache_write_tokens}")
    print(f"model            : {a.model} ({a.provider})")
    if a.unverified_quotes:
        print(f"UNVERIFIED QUOTES: {a.unverified_quotes} discarded — the model "
              f"cited text that is not in the source")
    print()
    return 0


async def run(args) -> int:
    settings = load_settings()
    service = RagService(settings)
    try:
        if args.command == "health":
            info = await service.store.health()
            print(json.dumps(info, indent=2, default=str))
            print(f"\nanswer_provider    : {service.answerer.provider}")
            print(f"answers_enabled    : {service.answerer.available}")
            print(f"embeddings_enabled : {service.embedder.available}")
            return 0

        if args.command == "verify-embeddings":
            # The single highest-risk unknown: the collections were built by
            # another pipeline, and a different 1536-dim model would degrade
            # retrieval silently rather than erroring. Re-embed text taken from
            # a stored point and compare with that point's own vector.
            if not service.embedder.available:
                print("OPENAI_API_KEY is not set — cannot verify.")
                return 2
            sample = await service.store.sample_point_with_vector()
            if sample is None:
                print("No sampleable point with a plain vector in the collection.")
                return 2
            text, vector = sample
            cos = await service.embedder.verify_against_stored(text, vector)
            print(f"configured model : {service.embedder.model}")
            print(f"stored dims      : {len(vector)}")
            print(f"cosine(fresh, stored) = {cos:.6f}\n")
            if cos > 0.99:
                print("MATCH — the configured model is the one that built the index.")
                return 0
            if cos > 0.90:
                print("SUSPECT — close but not identical. Possible model version "
                      "drift or text normalisation differences at ingest.")
                return 1
            print("MISMATCH — a different embedding model built this index. "
                  "Retrieval quality is degraded. Try text-embedding-ada-002 "
                  "or ask the pipeline owner which model was used.")
            return 1

        if args.command == "verify-answer-model":
            # The configured model string cannot be validated offline -- OpenAI
            # model names change and availability is per-account. Ask the API.
            if service.answerer.provider != "openai":
                print(f"answer provider is {service.answerer.provider!r}; "
                      "this check applies to the openai provider.")
                return 2
            if not service.answerer.available:
                print("OPENAI_API_KEY is not set — cannot verify.")
                return 2
            client = service.answerer._client
            want = get_settings().openai_answer_model
            try:
                info = await client.models.retrieve(want)
                print(f"model {want!r} is available (id={info.id})")
            except Exception as exc:
                print(f"model {want!r} NOT available: {type(exc).__name__}: "
                      f"{str(exc)[:160]}\n")
                try:
                    listed = [m.id async for m in client.models.list()]
                except Exception as exc2:
                    print(f"could not list models either: {exc2}")
                    return 1
                chat = sorted(m for m in listed
                              if m.startswith(("gpt", "o1", "o3", "o4")))
                print(f"{len(listed)} models on this account; chat-capable "
                      f"candidates:")
                for m in chat[:40]:
                    print(f"   {m}")
                print("\nSet OPENAI_ANSWER_MODEL to one of these.")
                return 1

            # Structured output is how grounding is enforced on this backend,
            # so a model that cannot do json_schema is unusable here.
            try:
                probe = await client.chat.completions.create(
                    model=want,
                    max_completion_tokens=64,
                    messages=[{"role": "user", "content": "Reply with {\"ok\":true}"}],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "probe", "strict": True,
                            "schema": {
                                "type": "object", "additionalProperties": False,
                                "required": ["ok"],
                                "properties": {"ok": {"type": "boolean"}},
                            },
                        },
                    },
                )
                print("strict structured output: supported "
                      f"({probe.choices[0].message.content})")
                print("\nReady — grounding via verified quotes will work.")
                return 0
            except Exception as exc:
                print(f"strict structured output NOT supported: "
                      f"{type(exc).__name__}: {str(exc)[:200]}")
                print("\nThis backend enforces grounding through a strict JSON "
                      "schema. Without it, pick another model or use "
                      "ANSWER_PROVIDER=anthropic.")
                return 1

        if args.command == "plans":
            rows = await service.store.list_plans()
            for insurer, plan in rows:
                print(f"  {insurer[:34]:34s} | {plan}")
            print(f"\n{len(rows)} retail plans")
            return 0

        if args.command == "features":
            rows = await service.plans_with_features(args.flags)
            for r in rows:
                print(f"  {r['insurer'][:30]:30s} | {r['plan_name']}")
            print(f"\n{len(rows)} plans matching all of: {', '.join(args.flags)}")
            return 0

        if args.command == "retrieve":
            sections, trace, extra = await service.retrieve_only(
                args.question, plan_name=args.plan
            )
            total = sum(len(s.text) for s in sections)
            print(f"\nmode      : {trace.mode}")
            print(f"question  : {args.question}")
            print(f"plans     : {', '.join(trace.plans_considered) or '-'}")
            print(f"top score : {trace.top_score:.4f}")
            print(f"dropped   : group={trace.dropped_group} "
                  f"withdrawn={trace.dropped_withdrawn} "
                  f"low_score={trace.dropped_low_score}")
            print(f"latency   : {trace.retrieval_ms}ms")
            print(f"\n{len(sections)} sections, {total:,} chars "
                  f"(~{total // 4:,} tokens) would be sent as documents:\n")
            for s in sections:
                head = s.text[:88].replace("\n", " ")
                print(f"  [{s.score:.3f}] {s.label[:62]:62s} {len(s.text):>7,}ch")
                print(f"          {head}...")
            if extra:
                print("\nstructured attributes appended:\n")
                for line in extra.splitlines():
                    print("  " + line)
            return 0

        if args.command == "plan-ask":
            return _print_result(await service.ask_plan(
                args.question, args.plan, insurer=args.insurer))

        if args.command == "ask":
            return _print_result(await service.ask_broad(args.question))

        return 2
    finally:
        await service.aclose()


def main() -> int:
    p = argparse.ArgumentParser(prog="python -m src.rag.cli")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("health", help="cluster reachability and index coverage")
    sub.add_parser("verify-embeddings", help="prove the query model matches the index")
    sub.add_parser("verify-answer-model",
                   help="check OPENAI_ANSWER_MODEL exists and does structured output")
    sub.add_parser("plans", help="list retail plans")

    f = sub.add_parser("features", help="exact structured filter, no LLM")
    f.add_argument("flags", nargs="+")

    r = sub.add_parser("retrieve", help="dry run: show context, call no model")
    r.add_argument("question")
    r.add_argument("--plan", default=None,
                   help="named plan (no embedding needed); omit for broad search")

    pa = sub.add_parser("plan-ask", help="ask about one named plan")
    pa.add_argument("question")
    pa.add_argument("--plan", required=True)
    pa.add_argument("--insurer", default=None,
                    help="disambiguates plan names shared across insurers")

    a = sub.add_parser("ask", help="ask across the corpus")
    a.add_argument("question")

    args = p.parse_args()
    try:
        level = load_settings().log_level
    except ConfigError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        return asyncio.run(run(args))
    except ConfigError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
