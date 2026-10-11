"""Application service: engine cache, analyze, ingest. FastAPI and the eval harness both
call these functions, so the evaluation exercises exactly what the API serves."""

import io
import logging
import threading
import time
import uuid

import pandas as pd

from src import store
from src.analysis import Engine
from src.graph import ASSUMPTIONS_BASE, LIMITATIONS, build_graph
from src.loader import SchemaError, normalise
from src.solver import normalise_weights
from src.validate import clean_pack, format_report, is_valid, validate_all, validate_rows

log = logging.getLogger(__name__)

_graph = build_graph()
_engine: Engine | None = None
_engine_checks: list = []
_engine_lock = threading.Lock()


def get_engine() -> tuple[Engine | None, list]:
    """Engine for the current data version (rebuilt after any ingest). Returns
    (None, checks) if the stored data fails validation."""
    global _engine, _engine_checks
    version = store.data_version()
    with _engine_lock:
        if _engine is not None and _engine.version == version:
            return _engine, _engine_checks
        pack, rejected = clean_pack(store.load_pack())
        checks = validate_all(pack, rejected)
        _engine_checks = checks
        _engine = Engine(pack, version) if is_valid(checks) else None
        if _engine is None:
            return None, checks
        return _engine, checks


def warm_up() -> None:
    """Pre-solve the default plan and frontier so the first live question is fast."""
    import os

    from src.analysis import frontier
    if os.getenv("WARM_UP", "true").lower() != "true":
        return
    try:
        engine, _ = get_engine()
        if engine is not None:
            frontier(engine, engine.pack.events)
            frontier(engine, engine.no_events)
    except Exception as e:  # best effort; e.g. interpreter shutting down mid warm-up
        log.warning("warm-up skipped: %s", e)


def analyze(request: str, weights: dict | None = None, scenario: dict | None = None) -> dict:
    t0 = time.perf_counter()
    run_id = f"al-{uuid.uuid4().hex[:6]}"
    w = normalise_weights(weights)
    engine, checks = get_engine()

    if engine is None:
        failed = [f"{c.name}: {c.detail}" for c in checks if not c.ok and c.level == "error"]
        response = base_response(run_id, "clarification_needed",
                                 "The loaded data fails validation, so no plan can be made. "
                                 "Fix and re-upload: " + "; ".join(failed), w)
        trace = {"run_id": run_id, "request": request, "steps": [
            {"agent": "checker", "tool": "validate_data", "output": {"failed": failed}}],
            "llm_calls": 0, "tokens": 0}
        store.save_run(run_id, request, "clarification_needed", 0, response, trace)
        return response

    state = _graph.invoke({"run_id": run_id, "request": request, "weights": w,
                           "scenario": scenario or {}, "engine": engine,
                           "steps": [], "llm": [], "assumptions": []})
    out = state.get("out", {})
    answer = state["answer_type"]
    response = base_response(run_id, answer, out.get("summary", ""), w)
    for key in ("allocation", "kpis", "tradeoff", "validator", "narrative", "explanation",
                "scenario_comparison", "infeasible_orders", "recommendations", "evidence"):
        if key in out:
            response[key] = out[key]
    from src.analysis import implied_carbon_price
    price = implied_carbon_price(engine, w)
    response["assumptions"] = (
        [f"Weights {w['cost'] * 100:.0f}% cost, {w['carbon'] * 100:.0f}% carbon "
         f"(implied carbon price INR {price:,.0f}/t); lateness weight {w['lateness']:g} "
         "puts on-time delivery first" if price is not None else
         f"Weights 0% cost, 100% carbon; lateness weight {w['lateness']:g}"]
        + ASSUMPTIONS_BASE + state.get("assumptions", []) + engine.pack.notes)
    response["limitations"] = LIMITATIONS
    latency = round((time.perf_counter() - t0) * 1000, 1)
    llm = state.get("llm", [])
    trace = {
        "run_id": run_id, "request": request, "weights": w, "scenario": scenario or {},
        "data_version": engine.version, "answer_type": answer,
        "intent_source": state["decision"].source,
        "steps": state.get("steps", []),
        "llm_calls": sum(u.get("calls", 0) for u in llm),
        "tokens": sum(u.get("total_tokens", 0) for u in llm),
        "llm_detail": llm, "latency_ms": latency,
    }
    response["trace_summary"] = {"llm_calls": trace["llm_calls"], "tokens": trace["tokens"],
                                 "latency_ms": latency}
    store.save_run(run_id, request, answer, latency, response, trace)
    return response


def base_response(run_id: str, answer: str, summary: str, w: dict) -> dict:
    return {"run_id": run_id, "answer_type": answer, "summary": summary,
            "allocation": [], "kpis": {}, "tradeoff": [],
            "weights_used": {"cost": round(w["cost"], 4), "carbon": round(w["carbon"], 4),
                             "lateness": w["lateness"]},
            "validator": {}, "recommendations": [], "assumptions": [], "limitations": []}


MAX_ROWS = 100_000


def ingest(table: str, content: bytes, mode: str = "replace") -> dict:
    """Validate one CSV and store its valid rows. Returns the contract fields plus the
    rejected rows (first 20) and the cross-table validation report after loading."""
    try:
        df = pd.read_csv(io.BytesIO(content), nrows=MAX_ROWS + 1)
    except Exception as e:
        return {"rows_loaded": 0, "rows_rejected": 0, "status": "rejected",
                "error": f"Not a readable CSV: {type(e).__name__}"}
    if len(df) > MAX_ROWS:
        return {"rows_loaded": 0, "rows_rejected": len(df), "status": "rejected",
                "error": f"More than {MAX_ROWS} rows"}
    notes: list[str] = []
    try:
        df = normalise(table, df, notes)
    except SchemaError as e:
        return {"rows_loaded": 0, "rows_rejected": len(df), "status": "rejected",
                "error": str(e)}
    valid, rejected = validate_rows(table, df)
    if valid.empty:
        status = "rejected"
    else:
        store.save_table(table, valid, mode)
        status = "partial" if len(rejected) else "loaded"
    store.log_ingest(table, mode, len(valid), len(rejected), status)

    engine, checks = get_engine()
    return {
        "rows_loaded": int(len(valid)), "rows_rejected": int(len(rejected)), "status": status,
        "rejected_rows": [{"line": int(i) + 2, "reason": r}
                          for i, r in zip(rejected.index[:20], rejected["reason"][:20])],
        "notes": notes,
        "data_valid": engine is not None,
        "validation": [{"check": c.name, "ok": c.ok, "level": c.level, "detail": c.detail}
                       for c in checks],
    }


def validation_report() -> dict:
    pack, rejected = clean_pack(store.load_pack())
    checks = validate_all(pack, rejected)
    return {"valid": is_valid(checks), "report": format_report(checks, pack.notes),
            "checks": [{"check": c.name, "ok": c.ok, "level": c.level, "detail": c.detail}
                       for c in checks]}
