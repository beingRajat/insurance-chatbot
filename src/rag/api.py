"""FastAPI surface.

Deliberately plain: `anthropic` and `qdrant-client` are called directly rather
than through an orchestration framework. The house rule against deep
abstraction layers in the latency path (see CLAUDE.md) applies here for the
same reason it applies to the finance agent -- every hop costs milliseconds and
obscures failures.

    uvicorn src.rag.api:app --reload
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from qdrant_client.http.exceptions import UnexpectedResponse

from .answer_base import AnswerUnavailable
from .config import load_settings
from .embeddings import EmbeddingUnavailable
from .service import RagService

log = logging.getLogger("rag.api")

_service: RagService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    settings = load_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _service = RagService(settings)
    log.info(
        "started | provider=%s answers=%s embeddings=%s",
        _service.answerer.provider,
        _service.answerer.available, _service.embedder.available,
    )
    if not _service.answerer.available:
        needed = ("OPENAI_API_KEY" if _service.answerer.provider == "openai"
                  else "ANTHROPIC_API_KEY")
        log.warning("%s unset — /ask and /plan/ask will return 503", needed)
    if not _service.embedder.available:
        log.warning("OPENAI_API_KEY unset — /ask will return 503 (/plan/ask still works)")
    try:
        yield
    finally:
        await _service.aclose()
        _service = None


app = FastAPI(
    title="Policy Answer Service",
    version="0.1.0",
    summary="Grounded question answering over the health-policy corpus in Qdrant.",
    lifespan=lifespan,
)


STATIC = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
async def chat_ui():
    """Serve the chat interface.

    Same-origin on purpose: the page calls /plans, /plan/ask and /ask by
    relative path, so there is no CORS configuration to maintain and no host to
    configure. It cannot be hosted elsewhere without opening CORS.
    """
    return FileResponse(STATIC / "chat.html", media_type="text/html")


def svc() -> RagService:
    if _service is None:  # pragma: no cover - lifespan guarantees this
        raise HTTPException(503, "service not initialised")
    return _service


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    started = time.perf_counter()
    response = await call_next(request)
    elapsed = int((time.perf_counter() - started) * 1000)
    response.headers["x-request-id"] = rid
    log.info("%s %s -> %s in %dms | rid=%s",
             request.method, request.url.path, response.status_code, elapsed, rid)
    return response


# ---------------------------------------------------------------- schemas
class CitationOut(BaseModel):
    quoted_text: str
    source: str = Field(description="Insurer — plan — section the quote came from")
    page: int | None = Field(
        None, description="Page in the source PDF, when recoverable from the text"
    )


class TraceOut(BaseModel):
    mode: str
    plans_considered: list[str]
    sections_used: list[str]
    top_score: float
    dropped_group: int
    dropped_withdrawn: int
    dropped_low_score: int
    dropped_standard: int
    document_read: str
    alternative_documents: list[str]
    retrieval_ms: int
    answer_ms: int


class AnswerOut(BaseModel):
    answer: str
    grounded: bool = Field(description="False when no claim carried a citation")
    citations: list[CitationOut]
    trace: TraceOut
    model: str
    provider: str
    unverified_quotes: int = Field(
        0,
        description="Quotes the model attributed to a document they do not "
                    "appear in. Discarded, but counted: non-zero means the "
                    "model fabricated a citation. Always 0 on the Anthropic "
                    "backend, where citations come from the platform.",
    )
    usage: dict
    caveat: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    insurers: list[str] | None = Field(
        None, description="Optional canonical insurer names to restrict the search to"
    )


class PlanAskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    plan_name: str = Field(min_length=2, max_length=200)
    insurer: str | None = Field(
        None,
        description="Required when a plan name is shared by several insurers, "
                    "e.g. the mandated 'Arogya Sanjeevani Policy'.",
    )


class FeatureRequest(BaseModel):
    flags: list[str] = Field(
        min_length=1,
        description="Indexed boolean fields, e.g. ['has_maternity_cover', 'has_opd_cover']",
    )


CAVEAT = (
    "Answers are drawn only from the policy wordings held in this corpus and "
    "describe what those documents say. They are not advice on which policy to "
    "buy, and the corpus may lag the currently issued wording — verify against "
    "the insurer's current policy document and CIS before acting."
)


def _to_out(result) -> AnswerOut:
    a = result.answer
    if a is None:
        raise HTTPException(404, result.message or "nothing retrieved")
    if a.refusal:
        raise HTTPException(422, f"model declined to answer: {a.refusal}")
    t = result.trace
    return AnswerOut(
        answer=a.text,
        grounded=a.is_grounded,
        citations=[
            CitationOut(quoted_text=c.text, source=c.document_title, page=c.page)
            for c in a.citations
        ],
        trace=TraceOut(
            mode=t.mode, plans_considered=t.plans_considered,
            sections_used=t.sections_used, top_score=round(t.top_score, 4),
            dropped_group=t.dropped_group, dropped_withdrawn=t.dropped_withdrawn,
            dropped_low_score=t.dropped_low_score,
            dropped_standard=t.dropped_standard,
            document_read=t.document_read,
            alternative_documents=t.alternative_documents,
            retrieval_ms=t.retrieval_ms, answer_ms=t.answer_ms,
        ),
        model=a.model,
        provider=a.provider,
        unverified_quotes=a.unverified_quotes,
        usage={
            "input_tokens": a.input_tokens,
            "output_tokens": a.output_tokens,
            "cache_read_tokens": a.cache_read_tokens,
            "cache_write_tokens": a.cache_write_tokens,
            "documents_supplied": a.documents_supplied,
        },
        caveat=CAVEAT,
    )


# ---------------------------------------------------------------- routes
@app.get("/health", summary="Cluster reachability and index coverage")
async def health():
    s = svc()
    store = await s.store.health()
    return {
        "status": "ok",
        "qdrant": store,
        "answer_provider": s.answerer.provider,
        "answers_enabled": s.answerer.available,
        "embeddings_enabled": s.embedder.available,
    }


@app.get("/plans", summary="List retail plans available to query")
async def plans():
    rows = await svc().store.list_plans()
    return {
        "count": len(rows),
        "plans": [{"insurer": i, "plan_name": p} for i, p in rows],
    }


@app.post("/plan/ask", response_model=AnswerOut,
          summary="Ask about one named plan — reads the whole policy")
async def plan_ask(req: PlanAskRequest):
    try:
        result = await svc().ask_plan(
            req.question, req.plan_name, insurer=req.insurer
        )
    except AnswerUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    return _to_out(result)


@app.post("/ask", response_model=AnswerOut,
          summary="Ask across the corpus — retrieval narrows, full reads decide")
async def ask(req: AskRequest):
    try:
        result = await svc().ask_broad(req.question, insurers=req.insurers)
    except EmbeddingUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except AnswerUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    return _to_out(result)


@app.post("/plans/by-feature", summary="Exact structured filter — no LLM involved")
async def plans_by_feature(req: FeatureRequest):
    try:
        rows = await svc().plans_with_features(req.flags)
    except UnexpectedResponse as exc:
        # Distinguish "that flag is not indexed" (caller error) from a cluster
        # fault, which must not be reported to the caller as a 400.
        body = str(exc)
        if "Index required but not found" in body or exc.status_code == 400:
            raise HTTPException(
                400,
                "Every flag must be an indexed field on the feature collection; "
                "the cluster rejects filters on unindexed fields. "
                f"Cluster said: {body[:200]}",
            ) from exc
        raise HTTPException(502, f"vector store error: {body[:200]}") from exc
    return {"count": len(rows), "plans": rows}
