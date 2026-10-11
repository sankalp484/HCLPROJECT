"""Cost, carbon, delivery and capacity models.

The formulas are written once and work on scalars or pandas Series, so the
candidate table the solver uses and any per-order explanation give the same numbers.

Modelling assumptions (also listed in answers):
- Landed cost = production + energy + transport + duty (R2).
- Duty is charged on the goods value leaving the plant: production + energy.
  Transport is not dutiable.
- Carbon = plant energy x plant carbon intensity for that week + transport carbon (R3).
- An order produced in week W is ready at the end of that week (week_start + 6 days)
  and arrives transit_days later.
- An order with a production_week is built that week, or deferred to a later planning
  week if capacity forces it (deferred_weeks > 0). Orders are never built early.
- weekly_capacity is shared by all products a plant makes in that week.
- A capacity event removes capacity_loss_pct of capacity for the days it overlaps
  a week (pro rata by days); overlapping events compound.
"""

import numpy as np
import pandas as pd

from src.loader import DataPack

COMPLETION_DAYS = 6


def landed_cost(qty, unit_cost, energy_kwh_per_unit, energy_price_per_kwh,
                transport_cost_per_unit, duty_pct) -> dict:
    """R2: landed cost components. Works on scalars or Series."""
    production = unit_cost * qty
    energy = energy_kwh_per_unit * energy_price_per_kwh * qty
    transport = transport_cost_per_unit * qty
    duty = duty_pct * (production + energy)
    return {"production": production, "energy": energy, "transport": transport,
            "duty": duty, "cost": production + energy + transport + duty}


def carbon(qty, energy_kwh_per_unit, carbon_kg_per_kwh, transport_carbon_kg_per_unit) -> dict:
    """R3: carbon components in kg CO2e. Works on scalars or Series."""
    plant = energy_kwh_per_unit * carbon_kg_per_kwh * qty
    transport = transport_carbon_kg_per_unit * qty
    return {"carbon_plant_kg": plant, "carbon_transport_kg": transport,
            "carbon_kg": plant + transport}


def planning_weeks(pack: DataPack) -> list[pd.Timestamp]:
    """Weeks the plan covers: the plant weeks if given, else weekly from the
    planning start to the latest due date."""
    if "week_start" in pack.plants.columns:
        return sorted(pack.plants["week_start"].unique())
    end = pack.orders["due"].max()
    return list(pd.date_range(pack.planning_start, end, freq="7D"))


def plants_by_week(pack: DataPack) -> pd.DataFrame:
    """Plant rows with a week_start. Rows without one apply to every planning week."""
    if "week_start" in pack.plants.columns:
        return pack.plants
    weeks = pd.DataFrame({"week_start": planning_weeks(pack)})
    return pack.plants.merge(weeks, how="cross")


def weekly_capacity(pack: DataPack, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Capacity per plant per week after capacity events.

    Returns plant, week_start, base_capacity, capacity_loss_pct, capacity.
    events defaults to pack.events; pass an empty frame for the no-event baseline.
    """
    events = pack.events if events is None else events
    cap = (plants_by_week(pack)
           .groupby(["plant", "week_start"], as_index=False)["weekly_capacity"].min()
           .rename(columns={"weekly_capacity": "base_capacity"}))

    cap["factor"] = 1.0
    if not events.empty:
        hits = cap.merge(events[["plant", "from_date", "to_date", "capacity_loss_pct"]],
                         on="plant")
        hits["week_end"] = hits["week_start"] + pd.Timedelta(days=6)
        first = hits[["from_date", "week_start"]].max(axis=1)
        last = hits[["to_date", "week_end"]].min(axis=1)
        overlap_days = ((last - first).dt.days + 1).clip(lower=0)
        hits["keep"] = 1 - hits["capacity_loss_pct"] * overlap_days / 7
        factor = hits.groupby(["plant", "week_start"])["keep"].prod()
        cap = cap.drop(columns="factor").merge(
            factor.rename("factor").reset_index(), on=["plant", "week_start"], how="left")
        cap["factor"] = cap["factor"].fillna(1.0)

    cap["capacity_loss_pct"] = 1 - cap["factor"]
    # whole units only; round down so a plan never relies on a fractional unit
    cap["capacity"] = np.floor(cap["base_capacity"] * cap["factor"] + 1e-9).astype("int64")
    return cap.drop(columns="factor")


def build_candidates(pack: DataPack) -> pd.DataFrame:
    """Every eligible (order, plant, production week) with its cost, carbon and arrival.

    Eligibility: the plant makes the product that week. Week choice: the order's
    production_week or a later planning week (deferral); orders without a
    production_week may use any planning week from the planning start. This table is
    the solver's input; lateness is a column, not a filter, so impossible dates show up
    as late candidates instead of disappearing.
    """
    orders = pack.orders
    if "production_week" not in orders.columns:
        orders = orders.assign(production_week=pd.NaT)

    plant_cols = ["plant", "product", "week_start", "unit_cost", "energy_kwh_per_unit",
                  "energy_price_per_kwh", "carbon_kg_per_kwh"]
    c = orders[["order_id", "product", "qty", "destination", "due", "production_week"]].merge(
        plants_by_week(pack)[plant_cols], on="product")

    earliest = c["production_week"].fillna(pack.planning_start)
    c = c[c["week_start"] >= earliest]
    c["deferred_weeks"] = np.where(
        c["production_week"].notna(),
        (c["week_start"] - c["production_week"]).dt.days // 7, 0)

    c = c.merge(pack.transport[["plant", "destination", "transport_cost_per_unit",
                                "transit_days", "duty_pct", "carbon_kg_per_unit"]],
                on=["plant", "destination"])

    cost = landed_cost(c["qty"], c["unit_cost"], c["energy_kwh_per_unit"],
                       c["energy_price_per_kwh"], c["transport_cost_per_unit"], c["duty_pct"])
    co2 = carbon(c["qty"], c["energy_kwh_per_unit"], c["carbon_kg_per_kwh"],
                 c["carbon_kg_per_unit"])
    c = c.assign(**cost, **co2)

    c["arrival"] = (c["week_start"] + pd.Timedelta(days=COMPLETION_DAYS)
                    + pd.to_timedelta(c["transit_days"], unit="D"))
    c["days_late"] = (c["arrival"] - c["due"]).dt.days.clip(lower=0)
    c["on_time"] = c["days_late"] == 0

    # Prune dominated deferrals: per order and plant keep the planned week, every
    # on-time week, and only the first late week. Later late weeks add lateness and
    # nothing else, and they make the MIP much slower to prove optimal.
    late_rank = c[~c["on_time"]].groupby(["order_id", "plant"])["week_start"].rank()
    c = c[(c["deferred_weeks"] == 0) | c["on_time"] | (late_rank.reindex(c.index) == 1)]

    keep = ["order_id", "product", "qty", "destination", "due", "plant", "week_start",
            "deferred_weeks", "production", "energy", "transport", "duty", "cost",
            "carbon_plant_kg", "carbon_transport_kg", "carbon_kg",
            "transit_days", "arrival", "days_late", "on_time"]
    return c[keep].sort_values(["order_id", "plant", "week_start"]).reset_index(drop=True)


def edge_case_profile(pack: DataPack, candidates: pd.DataFrame) -> dict[str, list[str]]:
    """Orders that hit the known traps, found from the data rather than hard-coded."""
    planned = candidates[candidates["deferred_weeks"] == 0]
    cheapest = planned.loc[planned.groupby("order_id")["cost"].idxmin()]
    on_time_ids = set(candidates.loc[candidates["on_time"], "order_id"])
    plants_per_order = planned.groupby("order_id")["plant"].nunique()
    return {
        "no_candidate": sorted(set(pack.orders["order_id"]) - set(candidates["order_id"])),
        "cheapest_plant_late": sorted(cheapest.loc[~cheapest["on_time"], "order_id"]),
        "no_on_time_option": sorted(set(candidates["order_id"]) - on_time_ids),
        "single_plant": sorted(plants_per_order[plants_per_order == 1].index),
    }
