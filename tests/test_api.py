"""Contract tests against the real FastAPI app with the tiny pack in a temp store."""

import importlib

import pytest
from fastapi.testclient import TestClient

from tests.conftest import EVENTS, write_pack


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "app.db"))
    monkeypatch.delenv("API_KEY", raising=False)
    from src import service, store
    importlib.reload(store)
    importlib.reload(service)
    store.init_db(seed_dir=write_pack(tmp_path / "pack"))
    import src.api as api
    importlib.reload(api)
    with TestClient(api.app) as c:
        yield c


CONTRACT_KEYS = {"run_id", "answer_type", "summary", "allocation", "kpis", "tradeoff",
                 "weights_used", "validator", "recommendations", "assumptions", "limitations"}


def test_health(client):
    h = client.get("/health").json()
    assert h["api"] == "ok" and h["data_store"]["data_valid"]
    assert h["llm"]["mode"] == "mock"


def test_allocation_contract_and_trace(client):
    r = client.post("/analyze", json={"request": "Allocate the orders",
                                      "weights": {"cost": 1, "carbon": 0}}).json()
    assert CONTRACT_KEYS <= set(r)
    assert r["answer_type"] == "allocation"
    assert r["kpis"]["total_cost"] == 1450 and r["validator"]["valid"]
    for rec in r["recommendations"]:
        assert {"action", "rationale", "evidence_refs", "confidence", "estimated_impact",
                "requires_approval"} <= set(rec)
    t = client.get(f"/trace/{r['run_id']}").json()
    assert [s["tool"] for s in t["steps"]] == ["parse_intent", "solve_allocation",
                                               "validate_plan", "write_answer"]
    assert t["llm_calls"] == 0


def test_scenario_via_json(client):
    r = client.post("/analyze", json={
        "request": "re-plan", "weights": {"cost": 1, "carbon": 0},
        "scenario": {"capacity_changes": [{"plant": "B", "loss_pct": 100,
                                           "from_date": "2026-10-12",
                                           "to_date": "2026-10-18"}]}}).json()
    assert r["answer_type"] == "scenario"
    assert r["scenario_comparison"]["extra_late_orders"] == 1
    assert r["validator"]["capacity_ok"]


def test_refused_and_clarification(client):
    assert client.post("/analyze", json={"request": "Move order O3 to plant A"}) \
        .json()["answer_type"] == "refused"
    assert client.post("/analyze", json={"request": "Plant Z loses 10% capacity"}) \
        .json()["answer_type"] == "clarification_needed"


def test_infeasible_on_time(client):
    r = client.post("/analyze", json={"request": "cheapest plan with everything on time"}).json()
    assert r["answer_type"] == "infeasible"
    assert r["infeasible_orders"][0]["order_id"] == "O4"


def test_ingest_events_then_replan(client):
    r = client.post("/ingest", files={"file": ("e.csv", EVENTS, "text/csv")},
                    data={"meta": '{"table": "capacity_events"}'}).json()
    assert (r["rows_loaded"], r["rows_rejected"], r["status"]) == (1, 0, "loaded")
    out = client.post("/analyze", json={"request": "Re-plan for E9",
                                        "weights": {"cost": 1, "carbon": 0}}).json()
    assert out["answer_type"] == "scenario"
    assert out["scenario_comparison"]["extra_late_orders"] == 1


def test_ingest_rejects_bad_input(client):
    bad = "order_id,product,qty,destination,due\nO9,P1,-3,X,2026-11-01\n"
    r = client.post("/ingest", files={"file": ("o.csv", bad, "text/csv")},
                    data={"meta": '{"table": "orders"}'}).json()
    assert r["status"] == "rejected" and r["rows_rejected"] == 1
    assert client.post("/ingest", files={"file": ("o.csv", bad, "text/csv")},
                       data={"meta": '{"table": "users"}'}).status_code == 422
    assert client.post("/analyze", json={"request": ""}).status_code == 422
    assert client.post("/analyze", json={"request": "x", "weights": {"cost": -1}}) \
        .status_code == 422
    assert client.get("/trace/not-a-run").status_code == 422


def test_api_key_enforced(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    assert client.post("/analyze", json={"request": "allocate"}).status_code == 401
    assert client.get("/health").status_code == 200
    ok = client.post("/analyze", json={"request": "allocate"}, headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200
