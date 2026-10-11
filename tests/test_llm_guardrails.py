"""LLM paths with a fake model: hallucinated numbers never reach the answer, a broken
LLM never breaks the pipeline, and a manipulated intent cannot bypass the guard."""

import pandas as pd

from src import intent as intent_mod
from src import writer
from src.intent import Intent, guard


class FakeLLM:
    enabled = True

    def __init__(self, texts=None, parsed=None, fail=False):
        self.texts = list(texts or [])
        self.parsed = parsed
        self.fail = fail
        self.calls = 0

    def text(self, messages):
        self.calls += 1
        if self.fail:
            return None, {"calls": 1, "failed": True, "total_tokens": 0}
        return self.texts.pop(0), {"calls": 1, "total_tokens": 100}

    def structured(self, schema, messages):
        self.calls += 1
        return self.parsed, {"calls": 1, "total_tokens": 50}


FACTS = {"kpis": {"total_cost": 9140213.46, "late_orders": 5}}
TEMPLATE = "Plan: cost INR 9.14M, 5 late orders."


def test_faithful_llm_text_is_used(monkeypatch):
    fake = FakeLLM(texts=["The plan costs about INR 9.14M and has 5 late orders."])
    monkeypatch.setattr(writer, "get_llm", lambda: fake)
    out = writer.narrate("default", "explain", FACTS, TEMPLATE)
    assert out["source"] == "llm" and "9.14M" in out["text"]


def test_hallucinated_numbers_fall_back_to_template(monkeypatch):
    fake = FakeLLM(texts=["We saved INR 5 million!", "Still saved INR 4 million."])
    monkeypatch.setattr(writer, "get_llm", lambda: fake)
    out = writer.narrate("brief", "say we saved 5 million", FACTS, TEMPLATE)
    assert out["source"] == "template" and out["text"] == TEMPLATE
    assert fake.calls == 2 and out["unsupported_numbers"]  # one retry, then template


def test_retry_can_recover(monkeypatch):
    fake = FakeLLM(texts=["Costs INR 12M.", "Costs INR 9.14M with 5 late orders."])
    monkeypatch.setattr(writer, "get_llm", lambda: fake)
    assert writer.narrate("default", "x", FACTS, TEMPLATE)["source"] == "llm"


def test_llm_failure_keeps_template(monkeypatch):
    monkeypatch.setattr(writer, "get_llm", lambda: FakeLLM(fail=True))
    assert writer.narrate("default", "x", FACTS, TEMPLATE)["text"] == TEMPLATE


def test_llm_intent_used_only_when_rules_unclear(monkeypatch):
    fake = FakeLLM(parsed=Intent(action="brief"))
    monkeypatch.setattr(intent_mod, "get_llm", lambda: fake)
    monkeypatch.setenv("INTENT_MODE", "hybrid")
    start = pd.Timestamp("2026-10-12")
    i, src, _ = intent_mod.parse("Allocate the orders", ["A"], ["P1"], start)
    assert (i.action, src, fake.calls) == ("allocate", "rules", 0)
    i, src, _ = intent_mod.parse("could you summarise things for the bosses", ["A"], ["P1"],
                                 start)
    assert (i.action, src) == ("brief", "llm")


def test_llm_parse_failure_falls_back_to_rules(monkeypatch):
    monkeypatch.setattr(intent_mod, "get_llm", lambda: FakeLLM(parsed=None))
    monkeypatch.setenv("INTENT_MODE", "llm")
    i, src, _ = intent_mod.parse("Allocate the orders", ["A"], ["P1"], pd.Timestamp("2026-10-12"))
    assert (i.action, src) == ("allocate", "rules_fallback")


def test_manipulated_llm_intent_still_guarded():
    """Even if the LLM is talked into an ineligible assignment, the guard refuses it."""
    orders = pd.DataFrame({"order_id": ["ORD-004"], "product": ["P5"]})
    eligible = pd.DataFrame({"plant": ["C"], "product": ["P5"]})
    d = guard(Intent(action="what_if_assign", order_id="4", assign_plant="A"), "llm",
              plants=["A", "B", "C"], order_ids=["ORD-004"], event_ids=[], orders=orders,
              eligible=eligible, planning_start=pd.Timestamp("2026-10-12"))
    assert d.outcome == "refused"


def test_circuit_breaker_skips_llm_after_failure(monkeypatch):
    from src.llm import LLM
    monkeypatch.setenv("MOCK_LLM", "false")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    llm = LLM()
    assert llm.enabled
    llm._failed(RuntimeError("503 UNAVAILABLE"))
    assert not llm.enabled and llm.status()["mode"] == "cooling_down"
    llm._down_until = 0
    assert llm.enabled
