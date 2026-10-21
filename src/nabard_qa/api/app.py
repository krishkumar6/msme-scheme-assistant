"""FastAPI application: JSON + server-sent-events API and the web UI."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..config import Settings, get_settings
from ..feedback.store import UnknownAnswerError
from ..schemas import AnswerResult, AskRequest, FeedbackRequest
from ..service import QAService

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"


class RateLimiter:
    """Sliding-window limit per client IP (in-process; put a gateway in front for multi-replica)."""

    def __init__(self, limit: int, window_s: float = 60.0) -> None:
        self.limit, self.window = limit, window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def __call__(self, request: Request) -> None:
        key = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                raise HTTPException(status_code=429, detail="Too many requests, please slow down.")
            q.append(now)


def create_app(settings: Settings | None = None, service: QAService | None = None) -> FastAPI:
    settings = settings or get_settings()
    state: dict[str, QAService] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        state["svc"] = service or QAService.from_settings(settings)
        log.info("service ready: provider=%s chunks=%d", state["svc"].generator.name, state["svc"].store.count())
        yield

    app = FastAPI(title="MSME & NABARD Scheme Assistant", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])
    ask_limit = RateLimiter(limit=30)

    def svc() -> QAService:
        return state["svc"]

    def require_admin(authorization: str | None = Header(default=None)) -> None:
        if not settings.admin_token:
            raise HTTPException(status_code=403, detail="admin endpoints are disabled (set NABARD_QA_ADMIN_TOKEN)")
        if authorization != f"Bearer {settings.admin_token}":
            raise HTTPException(status_code=401, detail="invalid admin token")

    @app.middleware("http")
    async def request_log(request: Request, call_next):  # type: ignore[no-untyped-def]
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled error rid=%s %s %s", rid, request.method, request.url.path)
            return JSONResponse(status_code=500, content={"detail": "internal error", "request_id": rid})
        response.headers["x-request-id"] = rid
        log.info("rid=%s %s %s -> %s %.0fms", rid, request.method, request.url.path,
                 response.status_code, (time.perf_counter() - t0) * 1000)
        return response

    def _validate(req: AskRequest) -> None:
        if len(req.question) > settings.max_question_chars:
            raise HTTPException(status_code=422, detail=f"question longer than {settings.max_question_chars} characters")

    @app.get("/health")
    def health() -> dict:
        s = svc()
        return {
            "status": "ok",
            "provider": s.generator.name,
            "deep_provider": s.deep_generator.name,
            "chunks": s.store.count(),
            "reranker": s.retriever.reranker.model_name if s.retriever.reranker else None,
            "adapter_active": s.retriever.adapter.active,
            "years": sorted(s.retriever.known_years),
        }

    @app.post("/api/ask", response_model=AnswerResult, dependencies=[Depends(ask_limit)])
    def ask(req: AskRequest) -> AnswerResult:
        _validate(req)
        return svc().answer(req)

    @app.post("/api/ask/stream", dependencies=[Depends(ask_limit)])
    def ask_stream(req: AskRequest) -> StreamingResponse:
        _validate(req)

        def events() -> Iterator[str]:
            try:
                for event in svc().stream(req):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception:  # noqa: BLE001 - surface as an SSE error event, not a broken stream
                log.exception("stream failed")
                yield f"data: {json.dumps({'type': 'error', 'detail': 'Something went wrong. Please retry.'})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/feedback")
    def feedback(req: FeedbackRequest) -> dict:
        try:
            signals = svc().feedback.add_feedback(
                req.answer_id, req.rating, req.comment, req.relevant_chunk_ids, req.irrelevant_chunk_ids
            )
        except UnknownAnswerError:
            raise HTTPException(status_code=404, detail="unknown answer_id") from None
        return {"status": "recorded", "signals": len(signals)}

    @app.get("/api/stats")
    def stats() -> dict:
        return {**svc().feedback.stats(), "adapter_active": svc().retriever.adapter.active}

    @app.post("/api/admin/train-adapter", dependencies=[Depends(require_admin)])
    def train() -> dict:
        from ..feedback.learner import train_adapter

        s = svc()
        report = train_adapter(settings, s.retriever.embedder, s.store, s.feedback)
        active = s.reload_adapter() if report and report.accepted else s.retriever.adapter.active
        return {"report": report.__dict__ if report else None, "adapter_active": active}

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app
