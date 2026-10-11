"""Independent plan validator (Annex A). It re-derives every rule from the raw tables
instead of trusting the solver's candidate table, so a solver or model bug shows up here."""

import pandas as pd

from src.loader import DataPack
from src.models import COMPLETION_DAYS, plants_by_week


def validate_plan(pack: DataPack, allocation: pd.DataFrame, capacity: pd.DataFrame,
                  unassigned: list[str] | None = None) -> dict:
    unassigned = unassigned or []
    result = {}

    # Every order appears exactly once: planned or flagged unassigned
    counts = allocation["order_id"].value_counts()
    planned = set(counts.index)
    all_orders = set(pack.orders["order_id"])
    result["assignment_ok"] = bool(
        (counts == 1).all() and planned | set(unassigned) == all_orders
        and not planned & set(unassigned))

    # Eligibility: the plant makes the product in that week
    eligible = plants_by_week(pack)[["plant", "product", "week_start"]].drop_duplicates()
    merged = allocation.merge(eligible, on=["plant", "product", "week_start"], how="left",
                              indicator=True)
    ineligible = merged.loc[merged["_merge"] == "left_only", "order_id"].tolist()
    result["eligibility_ok"] = not ineligible

    # Capacity: units per plant-week within the (event-adjusted) capacity
    used = allocation.groupby(["plant", "week_start"])["qty"].sum().rename("used").reset_index()
    load = used.merge(capacity, on=["plant", "week_start"], how="left").fillna({"capacity": 0})
    over = load[load["used"] > load["capacity"]]
    result["capacity_ok"] = over.empty

    # Never built before its production week
    if "production_week" in pack.orders.columns:
        pw = allocation.merge(pack.orders[["order_id", "production_week"]], on="order_id")
        early = pw[pw["production_week"].notna() & (pw["week_start"] < pw["production_week"])]
        result["no_early_build_ok"] = early.empty

    # Delivery: recompute arrival from transit days; every late order must be flagged
    t = allocation.merge(pack.transport[["plant", "destination", "transit_days"]],
                         on=["plant", "destination"], suffixes=("", "_raw"))
    arrival = (t["week_start"] + pd.Timedelta(days=COMPLETION_DAYS)
               + pd.to_timedelta(t["transit_days_raw"], unit="D"))
    actually_late = arrival > t["due"]
    result["delivery_flags_ok"] = bool((actually_late == ~t["on_time"]).all())
    result["late_orders"] = sorted(t.loc[actually_late, "order_id"])
    result["unassigned_orders"] = sorted(unassigned)

    result["details"] = {
        "ineligible": ineligible,
        "over_capacity": [f"{r.plant} {r.week_start:%Y-%m-%d}: {int(r.used)} > {int(r.capacity)}"
                          for r in over.itertuples()],
    }
    result["valid"] = all(v for k, v in result.items() if k.endswith("_ok"))
    return result
