import pandas as pd
import pytest

from src.models import weekly_capacity
from src.plan_check import validate_plan
from src.solver import cheapest_plant_baseline, normalise_weights, solve
from tests.conftest import plant_of


@pytest.fixture
def setup(tiny_engine):
    return tiny_engine, tiny_engine.candidates, tiny_engine.capacity(None)


def test_cost_first_plan_is_hand_optimum(setup):
    e, cand, cap = setup
    plan = solve(cand, cap, {"cost": 1, "carbon": 0})
    a = plan.allocation
    assert plan.status == "OPTIMAL"
    assert {o: plant_of(a, o) for o in ["O1", "O2", "O3", "O4"]} == \
        {"O1": "B", "O2": "A", "O3": "C", "O4": "A"}
    assert plan.kpis["total_cost"] == 1450
    assert plan.kpis["late_orders"] == 1  # O4 cannot be on time from anywhere
    assert validate_plan(e.pack, a, cap, plan.unassigned)["valid"]


def test_carbon_first_flips_o1_and_o2(setup):
    _, cand, cap = setup
    a = solve(cand, cap, {"cost": 0, "carbon": 1}).allocation
    assert plant_of(a, "O1") == "A" and plant_of(a, "O2") == "B"


def test_weight_settings_trade_cost_for_carbon(setup):
    _, cand, cap = setup
    k = [solve(cand, cap, {"cost": 1 - w, "carbon": w}).kpis for w in (0.0, 0.5, 1.0)]
    assert k[0]["total_cost"] <= k[1]["total_cost"] <= k[2]["total_cost"]
    assert k[0]["total_carbon_t"] >= k[1]["total_carbon_t"] >= k[2]["total_carbon_t"]


def test_capacity_loss_forces_a_late_deferral(setup, tiny_pack):
    e, cand, _ = setup
    events = pd.DataFrame({"event_id": ["E"], "plant": ["B"],
                           "from_date": [pd.Timestamp("2026-10-12")],
                           "to_date": [pd.Timestamp("2026-10-18")],
                           "capacity_loss_pct": [1.0]})
    cap = weekly_capacity(tiny_pack, events)
    plan = solve(cand, cap, {"cost": 1, "carbon": 0})
    check = validate_plan(e.pack, plan.allocation, cap, plan.unassigned)
    assert check["valid"]
    assert plan.kpis["late_orders"] == 2  # O4, plus O1 or O2 pushed a week
    assert plan.kpis["deferred_orders"] == 1


def test_on_time_only_leaves_impossible_order_unassigned(setup):
    _, cand, cap = setup
    assert solve(cand, cap, {"cost": 1, "carbon": 0}, on_time_only=True).unassigned == ["O4"]


def test_fixed_assignment_is_respected(setup):
    _, cand, cap = setup
    plan = solve(cand, cap, {"cost": 1, "carbon": 0}, fixed={"O1": "C"})
    assert plant_of(plan.allocation, "O1") == "C"


def test_stability_keeps_unaffected_orders(setup, tiny_pack):
    _, cand, cap = setup
    base = solve(cand, cap, {"cost": 1, "carbon": 0})
    again = solve(cand, cap, {"cost": 1, "carbon": 0}, base_plan=base.allocation)
    assert again.allocation["plant"].tolist() == base.allocation["plant"].tolist()


def test_cheapest_plant_baseline_breaks_capacity(setup):
    e, cand, cap = setup
    base = cheapest_plant_baseline(cand, cap)
    assert set(base.loc[base["order_id"].isin(["O1", "O2"]), "plant"]) == {"A"}
    assert not validate_plan(e.pack, base, cap)["capacity_ok"]  # 120 > 100 in A


def test_weights_normalised_and_checked():
    assert normalise_weights({"cost": 7, "carbon": 3})["cost"] == pytest.approx(0.7)
    with pytest.raises(ValueError):
        normalise_weights({"cost": 0, "carbon": 0})
    with pytest.raises(ValueError):
        normalise_weights({"cost": -1})
