"""FastAPI app implementing the Section 6 contract.

Security: optional API key (set API_KEY; send X-API-Key), per-client rate limit, input
size limits, strict Pydantic request models, fixed table allow-list, generic error
bodies (details go to the local log only).
"""

import json
import logging
import os
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import date
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src import service, store
from src.llm import get_llm

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "120"))



@asynccontextmanager
async def lifespan(_: FastAPI):
    store.init_db()
    threading.Thread(target=service.warm_up, daemon=True).start()
    yield


app = FastAPI(title="Multi-Plant Order Allocation", version="1.0.0", lifespan=lifespan)


# ---------- request / response models ----------

class Weights(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cost: float | None = Field(None, ge=0, le=100)
    carbon: float | None = Field(None, ge=0, le=100)
    lateness: float | None = Field(None, ge=0, le=100)


class CapacityChangeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plant: str = Field(min_length=1, max_length=20)
    loss_pct: float = Field(gt=0, le=100, description="0-1 or 0-100")
    from_date: date | None = None
    to_date: date | None = None


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    capacity_changes: list[CapacityChangeIn] | None = Field(None, max_length=20)
    event_ids: list[str] | None = Field(None, max_length=50)
    replan_events: bool | None = None
    fixed_assignments: dict[str, str] | None = Field(None, max_length=500)
    carbon_cut_pct: float | None = Field(None, gt=0, lt=100)
    apply_loaded_events: bool = True
    stability: bool = True


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request: str = Field(min_length=1, max_length=2000)
    weights: Weights | None = None
    scenario: Scenario | None = None


class Recommendation(BaseModel):
    action: str
    rationale: str
    evidence_refs: list[str]
    confidence: str
    estimated_impact: str
    requires_approval: bool


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    run_id: str
    answer_type: Literal["allocation", "scenario", "brief", "infeasible",
                         "clarification_needed", "refused"]
    summary: str
    allocation: list[dict]
    kpis: dict
    tradeoff: list[dict]
    weights_used: dict
    validator: dict
    recommendations: list[Recommendation]
    assumptions: list[str]
    limitations: list[str]


class IngestMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    table: Literal["orders", "plants", "transport_duties", "capacity_events"]
    mode: Literal["replace", "append"] = "replace"


# ---------- security dependencies ----------

def require_key(request: Request) -> None:
    expected = os.getenv("API_KEY")
    if not expected:
        return
    given = request.headers.get("x-api-key", "")
    if not secrets.compare_digest(given.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")


_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def rate_limit(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _hits_lock:
        q = _hits[client]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            raise HTTPException(status_code=429, detail="rate limit exceeded, retry shortly")
        q.append(now)


guarded = [Depends(require_key), Depends(rate_limit)]


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal error; see server log"})


# ---------- endpoints ----------

@app.post("/analyze", response_model=AnalyzeResponse, dependencies=guarded)
def analyze(body: AnalyzeRequest) -> dict:
    weights = body.weights.model_dump(exclude_none=True) if body.weights else None
    if weights and weights.get("cost", 1) == 0 and weights.get("carbon", 1) == 0:
        raise HTTPException(status_code=422, detail="cost and carbon weights cannot both be 0")
    scenario = body.scenario.model_dump(exclude_none=True, mode="json") if body.scenario else None
    return service.analyze(body.request, weights, scenario)


@app.post("/ingest", dependencies=guarded)
async def ingest(file: UploadFile = File(...), meta: str | None = Form(None),
                 table: str | None = Form(None), mode: str | None = Form(None)) -> dict:
    """Multipart: file=<csv> plus meta='{"table": "orders"}' (or form fields table/mode)."""
    try:
        raw = json.loads(meta) if meta else {"table": table, "mode": mode or "replace"}
        info = IngestMeta(**{k: v for k, v in raw.items() if v is not None})
    except (json.JSONDecodeError, ValidationError, TypeError, AttributeError) as e:
        raise HTTPException(status_code=422, detail=f"invalid meta: {str(e)[:300]}") from None
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file too large")
    if file.filename and not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="upload a .csv file")
    return service.ingest(info.table, content, info.mode)


@app.get("/health")
def health() -> dict:
    ok = store.ping()
    data = {"ok": ok}
    if ok:
        engine, checks = service.get_engine()
        data.update(data_version=store.data_version(), data_valid=engine is not None,
                    failed_checks=[c.name for c in checks if not c.ok and c.level == "error"])
    return {"status": "ok" if ok else "degraded", "api": "ok", "data_store": data,
            "llm": get_llm().status()}


@app.get("/trace/{run_id}", dependencies=guarded)
def trace(run_id: str) -> dict:
    if not re.fullmatch(r"al-[0-9a-f]{6}", run_id):
        raise HTTPException(status_code=422, detail="run_id looks like al-1a2b3c")
    record = store.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="run not found")
    return record


@app.get("/validation", dependencies=guarded)
def validation() -> dict:
    """Validation report for the data currently loaded (shown in the demo)."""
    return service.validation_report()
