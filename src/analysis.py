"""Deterministic tools the orchestration graph calls. Every number an answer shows comes
from here (models + solver), never from the LLM.

Engine wraps one data version: the candidate table is built once and plans are cached
by (weights, capacity events, options), so repeated questions do not re-solve.
"""

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from src.loader import DataPack
from src.models import build_candidates, weekly_capacity
from src.plan_check import validate_plan
from src.solver import (
    LATE_DAY,
    LATE_ORDER,
    Plan,
    cheapest_plant_baseline,
    normalise_weights,
    plan_kpis,
    reference_totals,
    solve,
)

# denser at low carbon weights, where most of the carbon reduction happens; 0.99 not 1.0
# so the greenest plan still breaks ties on cost
FRONTIER_CARBON_WEIGHTS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.99]
CACHE_SIZE = 64


def events_key(events: pd.DataFrame) -> tuple:
    if events is None or events.empty:
        return ()
    cols = ["plant", "from_date", "to_date", "capacity_loss_pct"]
    return tuple(sorted(tuple(str(v) for v in row) for row in events[cols].itertuples(index=False)))


class Engine:
    def __init__(self, pack: DataPack, version: int = 0):
        self.pack = pack
        self.version = version
        self.candidates = build_candidates(pack)
        self.references = reference_totals(self.candidates)
        self.no_events = pack.events.iloc[0:0]
        self._plans: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    @property
    def plants(self) -> list[str]:
        return sorted(self.pack.plants["plant"].unique())

    def capacity(self, events: pd.DataFrame | None) -> pd.DataFrame:
        return weekly_capacity(self.pack, self.no_events if events is None else events)

    def plan(self, weights: dict | None, events: pd.DataFrame | None = None, *,
             carbon_cap_kg: float | None = None, on_time_only: bool = False,
             fixed: dict | None = None, base_plan: Plan | None = None) -> Plan:
        w = normalise_weights(weights)
        key = (tuple(sorted(w.items())), events_key(events), carbon_cap_kg, on_time_only,
               tuple(sorted((fixed or {}).items())),
               None if base_plan is None else id(base_plan))
        with self._lock:
            if key in self._plans:
                self._plans.move_to_end(key)
                return self._plans[key]
        plan = solve(self.candidates, self.capacity(events), w, carbon_cap_kg=carbon_cap_kg,
                     on_time_only=on_time_only, fixed=fixed,
                     base_plan=None if base_plan is None else base_plan.allocation,
                     references=self.references)
        if base_plan is None:  # stability-anchored plans depend on another object; skip
            with self._lock:
                self._plans[key] = plan
                while len(self._plans) > CACHE_SIZE:
                    self._plans.popitem(last=False)
        return plan

    def check(self, plan: Plan) -> dict:
        return validate_plan(self.pack, plan.allocation, plan.capacity, plan.unassigned)


# ---------- helpers that turn plans into response-ready facts ----------

def allocation_records(allocation: pd.DataFrame) -> list[dict]:
    out = allocation[["order_id", "plant", "week_start", "qty", "cost", "carbon_kg", "arrival",
                      "due", "on_time", "days_late", "deferred_weeks"]].copy()
    for col in ("week_start", "arrival", "due"):
        out[col] = out[col].dt.strftime("%Y-%m-%d")
    out["cost"] = out["cost"].round(2)
    out["carbon_kg"] = out["carbon_kg"].round(1)
    return out.rename(columns={"week_start": "week"}).to_dict("records")


def implied_carbon_price(engine: Engine, weights: dict) -> float | None:
    """INR per tonne CO2e the weights are willing to pay: (w_carbon / w_cost) x
    (reference_cost / reference_carbon_t). Makes the abstract weight concrete."""
    w = normalise_weights(weights)
    if w["cost"] == 0:
        return None
    ref_cost, ref_carbon_kg = engine.references
    return round(w["carbon"] / w["cost"] * ref_cost / (ref_carbon_kg / 1000), 0)


def compare_plans(base: Plan, new: Plan) -> dict:
    """What changed between two plans. Deltas are like-for-like (orders placed in both);
    orders the new plan cannot place are listed separately with the cost excluded."""
    kb, kn = base.kpis, new.kpis
    a = base.allocation.set_index("order_id")
    b = new.allocation.set_index("order_id")
    common = a.index.intersection(b.index)
    moved_mask = (a.loc[common, "plant"] != b.loc[common, "plant"]) | \
                 (a.loc[common, "week_start"] != b.loc[common, "week_start"])
    moved = []
    for oid in common[moved_mask.to_numpy()]:
        moved.append({
            "order_id": oid,
            "from": f"{a.at[oid, 'plant']} wk {a.at[oid, 'week_start']:%Y-%m-%d}",
            "to": f"{b.at[oid, 'plant']} wk {b.at[oid, 'week_start']:%Y-%m-%d}",
            "cost_change": round(float(b.at[oid, "cost"] - a.at[oid, "cost"]), 2),
            "carbon_change_kg": round(float(b.at[oid, "carbon_kg"] - a.at[oid, "carbon_kg"]), 1),
            "now_late": bool(not b.at[oid, "on_time"] and a.at[oid, "on_time"]),
        })
    newly_unassigned = sorted(set(new.unassigned) - set(base.unassigned))

    # Like-for-like: compare only orders placed in BOTH plans. Otherwise a plan that
    # drops an order looks cheaper and greener, which is the opposite of the truth.
    ca, cb = a.loc[common], b.loc[common]
    dropped = a.loc[a.index.difference(b.index)]  # placed before, not placed now
    extra_cost = float(cb["cost"].sum() - ca["cost"].sum())
    return {
        "extra_cost": round(extra_cost, 2),
        "extra_cost_pct": round(100 * extra_cost / float(ca["cost"].sum()), 2)
        if len(ca) and ca["cost"].sum() else None,
        "extra_carbon_t": round(float(cb["carbon_kg"].sum() - ca["carbon_kg"].sum()) / 1000, 3),
        "extra_late_orders": int((~cb["on_time"]).sum() - (~ca["on_time"]).sum()),
        "newly_late": sorted(set(b.index[~b["on_time"]]) - set(a.index[~a["on_time"]])),
        "newly_unassigned": newly_unassigned,
        "unplaced_cost_excluded": round(float(dropped["cost"].sum()), 2),
        "unplaced_carbon_t_excluded": round(float(dropped["carbon_kg"].sum()) / 1000, 3),
        "orders_moved": len(moved),
        "plan_stability_pct": round(100 * (1 - len(moved) / max(len(common), 1)), 1),
        "moved": moved,
        "base_kpis": kb,
        "new_kpis": kn,
    }


# ---------- tools ----------

def frontier(engine: Engine, events: pd.DataFrame | None, lateness: float = 1.0) -> list[dict]:
    """R6/R11: optimal plans across carbon weights (the supported Pareto points).
    carbon_cut_pct is measured against the cheapest plan (carbon weight 0)."""
    def point(wc: float) -> dict:
        w = {"cost": 1 - wc, "carbon": wc, "lateness": lateness}
        plan = engine.plan(w, events)
        k = plan.kpis
        return {"carbon_weight": wc, "cost": k["total_cost"], "carbon_t": k["total_carbon_t"],
                "late_orders": k["late_orders"], "status": plan.status,
                "implied_carbon_price_inr_per_t": implied_carbon_price(engine, w)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        points = list(pool.map(point, FRONTIER_CARBON_WEIGHTS))
    base = points[0]
    for p in points:
        p["carbon_cut_pct"] = round(100 * (1 - p["carbon_t"] / base["carbon_t"]), 2)
        p["extra_cost"] = round(p["cost"] - base["cost"], 2)
    return points


def tradeoff(engine: Engine, events: pd.DataFrame | None, requested_cut_pct: float | None,
             lateness: float = 1.0) -> dict:
    """R6: what carbon cuts cost. The table is the weight frontier (every row a proven
    optimal plan). A specific requested cut is solved exactly: cheapest plan with carbon
    at least X% below the cheapest plan. A cut deeper than the greenest plan is
    infeasible and the greenest plan is returned as the closest option."""
    points = frontier(engine, events, lateness)
    cheapest, greenest = points[0], points[-1]
    max_cut = greenest["carbon_cut_pct"]
    result = {"rows": points, "max_cut_pct": max_cut, "requested_cut_pct": requested_cut_pct,
              "feasible": requested_cut_pct is None or requested_cut_pct <= max_cut,
              "cheapest": cheapest, "greenest": greenest}
    if requested_cut_pct is None:
        return result
    if requested_cut_pct > max_cut:
        result["requested"] = None
        result["closest"] = greenest
        return result

    # Smallest carbon weight whose optimal plan reaches the cut (bisection between the
    # bracketing frontier points). Every plan is a proven-optimal weighted solve; a hard
    # carbon cap was tried first but CBC often hits the time limit on it.
    target_t = cheapest["carbon_t"] * (1 - requested_cut_pct / 100)
    hi = next(p for p in points if p["carbon_t"] <= target_t + 1e-9)
    lo = max((p for p in points if p["carbon_weight"] < hi["carbon_weight"]),
             key=lambda p: p["carbon_weight"])
    lo_w, hi_w = lo["carbon_weight"], hi["carbon_weight"]
    for _ in range(6):
        mid = round((lo_w + hi_w) / 2, 4)
        k = engine.plan({"cost": 1 - mid, "carbon": mid, "lateness": lateness}, events).kpis
        if k["total_carbon_t"] <= target_t + 1e-9:
            hi_w = mid
        else:
            lo_w = mid
    plan = engine.plan({"cost": 1 - hi_w, "carbon": hi_w, "lateness": lateness}, events)
    k = plan.kpis
    result["requested"] = {
        "carbon_cut_pct": requested_cut_pct,
        "achieved_cut_pct": round(100 * (1 - k["total_carbon_t"] / cheapest["carbon_t"]), 2),
        "carbon_weight": hi_w, "cost": k["total_cost"],
        "carbon_t": k["total_carbon_t"], "late_orders": k["late_orders"],
        "extra_cost": round(k["total_cost"] - cheapest["cost"], 2),
        "extra_late_orders": k["late_orders"] - cheapest["late_orders"],
        "status": plan.status}
    result["requested_plan"] = plan
    return result


def cheapest_on_time(engine: Engine, events: pd.DataFrame | None) -> dict:
    """Cheapest plan with every order on time. If that is impossible, return the
    cheapest delivery-first plan as the closest option and say which orders fail."""
    strict = engine.plan({"cost": 1, "carbon": 0}, events, on_time_only=True)
    closest = engine.plan({"cost": 1, "carbon": 0}, events)
    cannot = sorted(set(strict.unassigned))
    late_rows = closest.allocation[~closest.allocation["on_time"]]
    return {
        "feasible": not cannot,
        "plan": strict if not cannot else closest,
        "orders_cannot_be_on_time": cannot,
        "closest_option_for_failing_orders": [
            {"order_id": r.order_id, "plant": r.plant, "arrival": f"{r.arrival:%Y-%m-%d}",
             "due": f"{r.due:%Y-%m-%d}", "days_late": int(r.days_late)}
            for r in late_rows.itertuples() if r.order_id in cannot],
    }


def capacity_scenario(engine: Engine, weights: dict, base_events: pd.DataFrame | None,
                      new_events: pd.DataFrame, stability: bool = True) -> dict:
    """R8/R9: re-plan with changed capacity and compare with the base plan.
    stability=True penalises moving orders that do not need to move."""
    base = engine.plan(weights, base_events)
    new = engine.plan(weights, new_events, base_plan=base if stability else None)
    cap_before = engine.capacity(base_events)
    cap_after = engine.capacity(new_events)
    changed = cap_before.merge(cap_after, on=["plant", "week_start"], suffixes=("_before", ""))
    changed = changed[changed["capacity"] != changed["capacity_before"]]
    return {
        "base": base, "new": new, "comparison": compare_plans(base, new),
        "capacity_changes": [
            {"plant": r.plant, "week": f"{r.week_start:%Y-%m-%d}",
             "capacity_before": int(r.capacity_before), "capacity_after": int(r.capacity)}
            for r in changed.itertuples()],
        "validation": engine.check(new),
    }


def what_if_assign(engine: Engine, weights: dict, events: pd.DataFrame | None,
                   fixed: dict[str, str]) -> dict:
    """Force orders to plants (eligibility already checked by the caller) and compare."""
    base = engine.plan(weights, events)
    new = engine.plan(weights, events, fixed=fixed)
    return {"base": base, "new": new, "comparison": compare_plans(base, new),
            "validation": engine.check(new)}


def ineligible_assignments(engine: Engine, fixed: dict[str, str]) -> list[str]:
    """Which forced placements break eligibility (plant cannot make the product)."""
    cand = engine.candidates
    problems = []
    for oid, plant in fixed.items():
        if cand[(cand["order_id"] == oid) & (cand["plant"] == plant)].empty:
            product = engine.pack.orders.loc[engine.pack.orders["order_id"] == oid, "product"]
            makes = sorted(engine.pack.plants.loc[
                engine.pack.plants["product"].isin(product), "plant"].unique())
            problems.append(f"{oid} ({', '.join(product)}) cannot be made in Plant {plant}; "
                            f"eligible: {', '.join(makes) or 'none'}")
    return problems


def explain_order(engine: Engine, plan: Plan, order_id: str) -> dict:
    """Why the plan put this order where it did: every eligible option with its cost,
    carbon, delivery and whether that plant-week had room."""
    alloc = plan.allocation
    chosen = alloc[alloc["order_id"] == order_id]
    order = engine.pack.orders[engine.pack.orders["order_id"] == order_id].iloc[0]
    used = alloc.groupby(["plant", "week_start"])["qty"].sum()
    cap = plan.capacity.set_index(["plant", "week_start"])["capacity"]

    options = engine.candidates[engine.candidates["order_id"] == order_id]
    # one row per plant: its best option by the plan's weights
    w = plan.weights
    score = (w["cost"] * options["cost"] / engine.references[0]
             + w["carbon"] * options["carbon_kg"] / engine.references[1]
             + w["lateness"] * (LATE_ORDER * 100 * (~options["on_time"]).astype(float)
                               + LATE_DAY * options["days_late"]))
    best = options.loc[score.groupby(options["plant"]).idxmin()]
    c = chosen.iloc[0] if not chosen.empty else None
    if c is not None:  # for the chosen plant show the chosen week, not its best-scoring one
        best = pd.concat([best[best["plant"] != c["plant"]], chosen]).sort_values("plant")

    rows = []
    for r in best.itertuples():
        key = (r.plant, r.week_start)
        in_use = int(used.get(key, 0)) - (int(c["qty"]) if c is not None and
                                          (c["plant"], c["week_start"]) == key else 0)
        free = int(cap.get(key, 0)) - in_use
        reasons = []
        if c is not None and r.plant != c["plant"]:
            if r.cost < c["cost"]:
                reasons.append("cheaper")
            if r.carbon_kg < c["carbon_kg"]:
                reasons.append("lower carbon")
            if not r.on_time:
                reasons.append(f"{int(r.days_late)} days late")
            if free < int(order["qty"]):
                reasons.append(f"plant-week full ({free} units free, needs {int(order['qty'])})")
        rows.append({"plant": r.plant, "week": f"{r.week_start:%Y-%m-%d}",
                     "cost": round(float(r.cost), 2), "carbon_kg": round(float(r.carbon_kg), 1),
                     "cost_vs_chosen": None if c is None else round(float(r.cost - c["cost"]), 2),
                     "carbon_vs_chosen_kg": None if c is None
                     else round(float(r.carbon_kg - c["carbon_kg"]), 1),
                     "arrival": f"{r.arrival:%Y-%m-%d}", "on_time": bool(r.on_time),
                     "days_late": int(r.days_late), "free_capacity_units": free,
                     "chosen": c is not None and r.plant == c["plant"]
                     and r.week_start == c["week_start"], "notes": reasons})

    eligible = set(options["plant"])
    return {
        "order_id": order_id, "product": order["product"], "qty": int(order["qty"]),
        "destination": order["destination"], "due": f"{order['due']:%Y-%m-%d}",
        "assigned": None if c is None else {
            "plant": c["plant"], "week": f"{c['week_start']:%Y-%m-%d}",
            "cost": round(float(c["cost"]), 2), "carbon_kg": round(float(c["carbon_kg"]), 1),
            "arrival": f"{c['arrival']:%Y-%m-%d}", "on_time": bool(c["on_time"]),
            "cost_breakdown": {k: round(float(c[k]), 2)
                               for k in ("production", "energy", "transport", "duty")}},
        "options": rows,
        "ineligible_plants": [p for p in engine.plants if p not in eligible],
        "weights": plan.weights,
    }


def baseline_comparison(engine: Engine, plan: Plan) -> dict:
    """Cheapest-plant rule (no solver) vs the optimised plan: shows what the solver fixes."""
    base = cheapest_plant_baseline(engine.candidates, plan.capacity)
    check = validate_plan(engine.pack, base, plan.capacity)
    return {"baseline_kpis": plan_kpis(base, plan.capacity),
            "baseline_capacity_violations": check["details"]["over_capacity"],
            "baseline_overloaded_plant_weeks": len(check["details"]["over_capacity"]),
            "baseline_valid": check["valid"]}
