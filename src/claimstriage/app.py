"""FastAPI service: FNOL intake, claim detail, queues, policy evaluation, notes, audit, health, and metrics."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Header, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .logs import get_logger, log_event
from .metrics import Metrics
from .notes import get_provider, write_notes
from .pipeline import Paths
from .service import BadRequest, NotFound, Triage, sample_fnol
from .store import Store

STATIC = Path(__file__).parent / "static"


class EstimateLine(BaseModel):
    code: str
    kind: str = Field(pattern="^(part|labor)$")
    units: int = 1
    qty: float = Field(gt=0)
    unit_price: float = Field(ge=0)
    desc: str | None = None


class Photos(BaseModel):
    count: int = 0
    exif_min: str | None = None
    gps_km: float | None = None


class FNOL(BaseModel):
    policy_id: str
    peril: str
    loss_cause: str
    loss_date: str
    report_date: str | None = None
    description: str = ""
    loss_lat: float | None = None
    loss_lon: float | None = None
    third_party: bool = False
    third_party_phone: str | None = None
    passengers: int = 0
    injury_claimed: bool = False
    police_report: bool = False
    photos: Photos = Photos()
    shop_id: str | None = None
    estimate_lines: list[EstimateLine]


def create_app(paths: Paths | None = None, provider=None, store: Store | None = None) -> FastAPI:
    paths = paths or Paths.from_env()
    triage = Triage(paths, store=store)
    provider = provider or get_provider()
    metrics = Metrics()
    log = get_logger()
    ev = triage.evaluation
    metrics.set_gauge(
        "ct_model_test_roc_auc", "Fraud model ROC AUC on held-out months.", ev["model"]["test"].get("roc_auc", 0)
    )
    metrics.set_gauge(
        "ct_policy_stp_share",
        "Straight-through share on held-out months, model policy.",
        ev["policies"]["model"]["stp_share"],
    )
    metrics.set_gauge(
        "ct_gate_passed", "1 when the model policy beats the rules-only baseline.", int(ev["gate"]["passed"])
    )
    app = FastAPI(title="Claims Triage", version="1.0.0")
    app.state.triage = triage

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t = time.perf_counter()
        response = await call_next(request)
        dt = time.perf_counter() - t
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        metrics.observe_request(path, response.status_code, dt)
        response.headers["x-request-id"] = rid
        if path not in ("/healthz", "/metrics"):
            log_event(
                log,
                "request",
                request_id=rid,
                method=request.method,
                path=request.url.path,
                route=path,
                status=response.status_code,
                ms=round(dt * 1000, 1),
            )
        return response

    @app.exception_handler(NotFound)
    async def not_found(_request, exc):
        return JSONResponse({"error": f"not found: {exc}"}, status_code=404)

    @app.exception_handler(BadRequest)
    async def bad_request(_request, exc):
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "claims": len(triage.claims), "config_version": triage.cfg["version"]}

    @app.get("/metrics")
    def prom():
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/summary")
    def summary():
        return {
            "queues": triage.queue_counts(),
            "dataset": ev["dataset"],
            "model": ev["model"],
            "gate": ev["gate"],
            "config_version": triage.cfg["version"],
            "thresholds": triage.cfg["routing"],
        }

    @app.get("/api/queues")
    def queues():
        return triage.queue_counts()

    @app.get("/api/queues/{route}")
    def queue(route: str, limit: int = 50, offset: int = 0):
        return triage.queue(route, min(max(limit, 1), 200), max(offset, 0))

    @app.get("/api/claims/{claim_id}")
    def claim(claim_id: str):
        return triage.detail(claim_id)

    @app.post("/api/claims/{claim_id}/notes")
    def notes(claim_id: str, x_adjuster: Annotated[str, Header()] = "adjuster"):
        out = write_notes(provider, triage.detail(claim_id))
        metrics.note(provider.name)
        triage.store.audit(x_adjuster, "notes_written", claim_id, {"provider": provider.name, "model": provider.model})
        return out

    @app.post("/api/fnol")
    def fnol(body: FNOL, x_adjuster: Annotated[str, Header()] = "fnol-api"):
        out = triage.submit(body.model_dump(), actor=x_adjuster)
        metrics.fnol_routed(out["route"]["queue"])
        log_event(
            log, "fnol_routed", claim_id=out["claim_id"], route=out["route"]["queue"], score=out["fraud"]["score"]
        )
        return out

    @app.get("/api/fnol/sample")
    def fnol_sample():
        return sample_fnol(triage)

    @app.get("/api/evaluation")
    def evaluation():
        return ev

    @app.get("/api/audit")
    def audit(claim_id: str | None = None, limit: int = 100):
        return {"items": triage.store.audit_rows(claim_id, min(max(limit, 1), 500))}

    return app
