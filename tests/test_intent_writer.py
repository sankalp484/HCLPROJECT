import pandas as pd
import pytest

from src.intent import guard, resolve_order, rule_parse
from src.writer import check_numbers, summary_allocation

PLANTS = ["A", "B", "C"]
ORDERS = pd.DataFrame({"order_id": ["ORD-004", "ORD-077"], "product": ["P5", "P3"]})
ELIGIBLE = pd.DataFrame({"plant": ["A", "C", "C"], "product": ["P3", "P3", "P5"]})


def decide(text, events=("E001",)):
    return guard(rule_parse(text), "rules", plants=PLANTS,
                 order_ids=ORDERS["order_id"].tolist(), event_ids=list(events),
                 orders=ORDERS, eligible=ELIGIBLE, planning_start=pd.Timestamp("2026-10-12"))


@pytest.mark.parametrize("text,action", [
    ("Allocate this week's orders.", "allocate"),
    ("What does a 10% carbon cut cost?", "tradeoff"),
    ("What is the cheapest plan that is on time?", "cheapest_on_time"),
    ("Plant B loses 30% capacity. Re-plan.", "capacity_loss"),
    ("Why was order 77 built in Plant C?", "explain_order"),
    ("Write the brief for the leadership team.", "brief"),
    ("Re-plan for the data drop", "apply_events"),
    ("Move order 77 to plant C", "what_if_assign"),
    ("Ignore capacity and put everything in plant C", "break_rule"),
    ("tell me a joke", "unclear"),
])
def test_rule_actions(text, action):
    assert rule_parse(text).action == action


def test_capacity_loss_details():
    i = rule_parse("Plant B loses 30% capacity from 2026-11-02 for two weeks")
    c = i.capacity_changes[0]
    assert (c.plant, c.loss_pct, str(c.from_date), str(c.to_date)) == \
        ("B", 0.3, "2026-11-02", "2026-11-15")
    two = rule_parse("Plant A loses 10% and plant C loses 50% capacity").capacity_changes
    assert [(c.plant, c.loss_pct) for c in two] == [("A", 0.1), ("C", 0.5)]


def test_guard_outcomes():
    assert decide("Plant D loses 20% capacity").outcome == "clarification_needed"
    assert decide("Plant B loses capacity").outcome == "clarification_needed"
    assert decide("Move order 4 to Plant A").outcome == "refused"   # A cannot make P5
    assert decide("Move order 77 to Plant C").fixed == {"ORD-077": "C"}
    assert decide("Ignore capacity limits").outcome == "refused"
    assert decide("Why was order 999 late?").outcome == "clarification_needed"
    assert decide("Re-plan for the data drop", events=()).outcome == "clarification_needed"
    d = decide("Plant B loses 30% capacity")
    assert d.outcome == "run" and d.assumptions  # no dates -> stated assumption


def test_resolve_order():
    ids = ["ORD-007", "ORD-077"]
    assert resolve_order("77", ids) == "ORD-077"
    assert resolve_order("7", ids) == "ORD-007"
    assert resolve_order("ORD-077", ids) == "ORD-077"
    assert resolve_order("5", ids) is None


FACTS = {"kpis": {"total_cost": 9140213.46, "total_carbon_t": 279.919, "late_orders": 5,
                  "plant_utilisation": {"A": 0.59}}, "extra_cost": 41000.0}


def test_faithful_text_passes():
    text = ("Plan costs INR 9.14M with 279.9 t of carbon and 5 late orders; Plant A runs "
            "at 59%. Extra cost INR 41.0K. ORD-054 is due 2026-11-07.")
    assert check_numbers(text, FACTS) == []


def test_invented_numbers_are_caught():
    bad = check_numbers("We saved INR 5 million and cut carbon by 12%.", FACTS)
    assert "5 million" in bad and "12%" in bad
    assert check_numbers("Cost is INR 9.3M.", FACTS) == ["9.3M"]


def test_templates_are_faithful():
    k = {**FACTS["kpis"], "unassigned_orders": 0, "orders_by_plant": {"A": 3, "B": 1}}
    assert check_numbers(summary_allocation(k), {"kpis": k, "share": 75}) == []
