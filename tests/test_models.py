from pathlib import Path

import pandas as pd

from src.loader import load_all
from src.models import build_candidates, carbon, edge_case_profile, landed_cost, weekly_capacity
from src.validate import clean_pack


def test_landed_cost_hand_computed():
    c = landed_cost(qty=10, unit_cost=100, energy_kwh_per_unit=2, energy_price_per_kwh=5,
                    transport_cost_per_unit=3, duty_pct=0.1)
    # duty on goods value (production 1000 + energy 100), not on transport
    assert c == {"production": 1000, "energy": 100, "transport": 30, "duty": 110.0,
                 "cost": 1240.0}


def test_carbon_hand_computed():
    c = carbon(qty=10, energy_kwh_per_unit=2, carbon_kg_per_kwh=0.5,
               transport_carbon_kg_per_unit=1)
    assert c == {"carbon_plant_kg": 10.0, "carbon_transport_kg": 10, "carbon_kg": 20.0}


def test_formulas_work_on_series():
    assert landed_cost(pd.Series([1, 2]), 10, 0, 0, 0, 0)["cost"].tolist() == [10, 20]


def test_candidates_eligibility_arrival_lateness(tiny_pack):
    c = build_candidates(tiny_pack)
    assert set(c.loc[c["order_id"] == "O3", "plant"]) == {"C"}  # only C makes P3
    assert set(c.loc[c["order_id"] == "O4", "plant"]) == {"A"}
    o1a = c[(c["order_id"] == "O1") & (c["plant"] == "A") & (c["deferred_weeks"] == 0)].iloc[0]
    assert o1a["arrival"] == pd.Timestamp("2026-10-20")  # 10-12 + 6 + 2
    assert o1a["on_time"] and o1a["cost"] == 500 and o1a["carbon_kg"] == 50
    o4 = c[c["order_id"] == "O4"]
    assert not o4["on_time"].any()
    assert o4["days_late"].min() == 13  # 10-12 + 6 + 20 = 11-07 vs due 10-25


def test_no_early_builds_and_deferral_pruning(tiny_pack):
    c = build_candidates(tiny_pack)
    assert (c["week_start"] >= pd.Timestamp("2026-10-12")).all()
    # O1 deferred to 10-19 arrives 10-27 > due 10-26: kept as the first late week only
    deferred = c[(c["order_id"] == "O1") & (c["deferred_weeks"] == 1)]
    assert len(deferred) == 3 and not deferred["on_time"].any()


def test_capacity_shared_and_events_pro_rata(tiny_pack):
    cap = weekly_capacity(tiny_pack, tiny_pack.events.iloc[0:0])
    a = cap[(cap["plant"] == "A") & (cap["week_start"] == "2026-10-12")]
    assert a["capacity"].iloc[0] == 100  # shared across P1/P2, not 100 each
    events = pd.DataFrame({"event_id": ["E1", "E2"], "plant": ["A", "A"],
                           "from_date": pd.to_datetime(["2026-10-12", "2026-10-19"]),
                           "to_date": pd.to_datetime(["2026-10-18", "2026-10-21"]),
                           "capacity_loss_pct": [0.3, 0.7]})
    cap = weekly_capacity(tiny_pack, events).set_index(["plant", "week_start"])["capacity"]
    assert cap[("A", pd.Timestamp("2026-10-12"))] == 70   # full week at 30%
    assert cap[("A", pd.Timestamp("2026-10-19"))] == 70   # 3 of 7 days at 70% = 30%
    assert cap[("B", pd.Timestamp("2026-10-12"))] == 100


def test_edge_cases_found_in_shipped_data():
    pack = clean_pack(load_all(Path(__file__).resolve().parent.parent / "data"))[0]
    p = edge_case_profile(pack, build_candidates(pack))
    assert "ORD-007" in p["cheapest_plant_late"]
    assert p["no_on_time_option"] == ["ORD-054", "ORD-099", "ORD-102", "ORD-110", "ORD-113"]
    assert len(p["single_plant"]) == 18
