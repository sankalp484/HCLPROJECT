"""A tiny pack whose optimal plans can be worked out by hand.

Plants (week 2026-10-12, plus 2026-10-19 for deferral):
  A makes P1, P2: capacity 100, unit cost 10, carbon 1.0 kg/kWh
  B makes P1:     capacity 100, unit cost 12, carbon 0.5 kg/kWh
  C makes P1, P3: capacity 50,  unit cost 15, carbon 0.1 kg/kWh
  1 kWh per unit, energy price 0, no duty, no transport cost or carbon.
Destination X: 2 transit days from every plant. Destination Y: 20 days.
Orders (production week 2026-10-12, ready 10-18):
  O1 P1 50 -> X due 10-26     O2 P1 60 -> X due 10-26
  O3 P3 10 -> X due 10-26 (only C)     O4 P2 10 -> Y due 10-25 (only A; always late)

Cost first:   O4 -> A, O3 -> C; A has 90 left: O2 (60) -> A, O1 (50) -> B. Cost 1450.
Carbon first: C has 40 left (fits neither), B is next greenest: O2 -> B, O1 -> A.
"""

import os

import pandas as pd
import pytest

os.environ["MOCK_LLM"] = "true"
os.environ["WARM_UP"] = "false"

ORDERS = """order_id,product,qty,destination,due,production_week
O1,P1,50,X,2026-10-26,2026-10-12
O2,P1,60,X,2026-10-26,2026-10-12
O3,P3,10,X,2026-10-26,2026-10-12
O4,P2,10,Y,2026-10-25,2026-10-12
"""

TRANSPORT = """plant,destination,transport_cost_per_unit,transit_days,duty_pct,carbon_kg_per_unit
A,X,0,2,0,0
B,X,0,2,0,0
C,X,0,2,0,0
A,Y,0,20,0,0
B,Y,0,20,0,0
C,Y,0,20,0,0
"""

EVENTS = """event_id,plant,from_date,to_date,capacity_loss_pct
E9,B,2026-10-12,2026-10-18,1.0
"""


def plants_csv() -> str:
    rows = ["plant,product,week_start,weekly_capacity,unit_cost,energy_kwh_per_unit,"
            "carbon_kg_per_kwh,energy_price_per_kwh,capacity_unit,carbon_unit"]
    spec = {"A": (["P1", "P2"], 100, 10, 1.0), "B": (["P1"], 100, 12, 0.5),
            "C": (["P1", "P3"], 50, 15, 0.1)}
    for week in ("2026-10-12", "2026-10-19"):
        for plant, (products, cap, cost, ci) in spec.items():
            for p in products:
                rows.append(f"{plant},{p},{week},{cap},{cost},1,{ci},0,units_per_week,"
                            "kgCO2e_per_kWh")
    return "\n".join(rows) + "\n"


def write_pack(path, events: bool = False):
    path.mkdir(parents=True, exist_ok=True)
    (path / "orders.csv").write_text(ORDERS)
    (path / "plants.csv").write_text(plants_csv())
    (path / "transport_duties.csv").write_text(TRANSPORT)
    if events:
        (path / "capacity_events.csv").write_text(EVENTS)
    return path


@pytest.fixture
def tiny_dir(tmp_path):
    return write_pack(tmp_path)


@pytest.fixture
def tiny_pack(tiny_dir):
    from src.loader import load_all
    from src.validate import clean_pack
    return clean_pack(load_all(tiny_dir))[0]


@pytest.fixture
def tiny_engine(tiny_pack):
    from src.analysis import Engine
    return Engine(tiny_pack)


def plant_of(allocation: pd.DataFrame, order_id: str) -> str:
    return allocation.loc[allocation["order_id"] == order_id, "plant"].iloc[0]
