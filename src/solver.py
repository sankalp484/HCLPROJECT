"""R4/R5/R7: allocate orders to (plant, week) with a mixed-integer program (PuLP + CBC).

Decision: x[i] = 1 if candidate row i (order, plant, week) is chosen; u[o] = 1 if order o
cannot be placed at all. Each order is built whole in one plant (no splitting).

Hard constraints (Annex A): every order is chosen once or marked unassigned; plant-week
capacity; eligibility (only eligible candidates exist).

Objective (all terms are dimensionless so weights mean the same thing on any data):
    w_cost   * plan_cost   / reference_cost
  + w_carbon * plan_carbon / reference_carbon
  + w_late   * (late_order_penalty * late orders + LATE_DAY * days late)
  + UNASSIGNED * late_order_penalty * unassigned orders
  + DEFER * deferred weeks + MOVE * orders moved

reference_cost / reference_carbon are the sums of each order's cheapest / greenest
option, so 1.0 = a whole plan's worth. late_order_penalty = LATE_ORDER + the largest
change in the weighted cost+carbon terms any plan could make (every order from its best
to its worst option). So with w_late = 1 one extra late order always outweighs any cost
or carbon gain: late orders are minimised first (lexicographic), then cost vs carbon.
w_late < 1 lets cost and carbon trade against lateness. Delivery is soft so impossible
dates are flagged, not hidden; unassigned is 10x worse than late.
"""

import time
from dataclasses import dataclass, field

import pandas as pd
import pulp

LATE_ORDER = 1.0
LATE_DAY = 0.01
UNASSIGNED = 10.0
DEFER = 1e-6  # tie-breaker only: prefer the planned week when all else is equal
MOVE = 2e-4  # re-plan stability: ~0.02% of plan cost per moved order
TIME_LIMIT_S = 20
# Stop when the solution is proven within this objective distance of optimal:
# 1e-4 of a whole plan = 0.01% of plan cost (~900 INR on the data pack).
GAP_ABS = 1e-4

# Default = knee of the frontier on the data pack: carbon weight 0.2 gets 91% of the
# achievable carbon cut for +0.33% cost; going further costs >2x more per tonne.
# eval/run_eval.py recomputes this table on any data so the choice can be re-checked.
DEFAULT_WEIGHTS = {"cost": 0.8, "carbon": 0.2, "lateness": 1.0}


@dataclass
class Plan:
    status: str
    allocation: pd.DataFrame  # chosen candidate rows
    unassigned: list[str]
    weights: dict
    capacity: pd.DataFrame  # plant/week capacity the plan was solved against
    solve_ms: float
    options: dict = field(default_factory=dict)

    @property
    def kpis(self) -> dict:
        return plan_kpis(self.allocation, self.capacity, self.unassigned)


def normalise_weights(weights: dict | None) -> dict:
    """Cost and carbon weights are rescaled to sum to 1; lateness is a separate multiplier."""
    w = {**DEFAULT_WEIGHTS, **{k: v for k, v in (weights or {}).items() if v is not None}}
    if min(w.values()) < 0:
        raise ValueError("weights must be non-negative")
    total = w["cost"] + w["carbon"]
    if total == 0:
        raise ValueError("cost and carbon weights cannot both be 0")
    return {"cost": w["cost"] / total, "carbon": w["carbon"] / total,
            "lateness": float(w["lateness"])}


def reference_totals(candidates: pd.DataFrame) -> tuple[float, float]:
    by_order = candidates.groupby("order_id")
    return float(by_order["cost"].min().sum()), float(by_order["carbon_kg"].min().sum())


def plan_kpis(allocation: pd.DataFrame, capacity: pd.DataFrame,
              unassigned: list[str] | None = None) -> dict:
    used = allocation.groupby("plant")["qty"].sum()
    cap = capacity.groupby("plant")["capacity"].sum()
    util = {p: round(float(used.get(p, 0) / cap[p]), 3) if cap[p] else 0.0 for p in cap.index}
    return {
        "total_cost": round(float(allocation["cost"].sum()), 2),
        "total_carbon_t": round(float(allocation["carbon_kg"].sum()) / 1000, 3),
        "late_orders": int((~allocation["on_time"]).sum()),
        "total_days_late": int(allocation["days_late"].sum()),
        "unassigned_orders": len(unassigned or []),
        "orders_planned": int(len(allocation)),
        "deferred_orders": int((allocation["deferred_weeks"] > 0).sum()),
        "plant_utilisation": util,
        "orders_by_plant": {p: int(n) for p, n in allocation["plant"].value_counts().items()},
        "orders_share_pct": {p: round(100 * n / len(allocation), 1)
                             for p, n in allocation["plant"].value_counts().items()}
        if len(allocation) else {},
    }


def solve(candidates: pd.DataFrame, capacity: pd.DataFrame, weights: dict | None = None, *,
          carbon_cap_kg: float | None = None, on_time_only: bool = False,
          fixed: dict[str, str] | None = None, base_plan: pd.DataFrame | None = None,
          references: tuple[float, float] | None = None) -> Plan:
    """Solve one allocation.

    carbon_cap_kg: upper bound on total carbon (used for the carbon-cut trade-off).
    on_time_only:  drop late candidates; orders with no on-time option become unassigned.
    fixed:         {order_id: plant} forced placements (what-if). Caller checks eligibility.
    base_plan:     previous allocation; moving an order away from it costs MOVE (stability).
    references:    (reference_cost, reference_carbon); pass the base case's values when
                   comparing scenarios so objectives are on the same scale.
    """
    start = time.perf_counter()
    w = normalise_weights(weights)
    ref_cost, ref_carbon = references or reference_totals(candidates)

    cand = candidates
    if on_time_only:
        cand = cand[cand["on_time"]]
    if fixed:
        forced = cand["order_id"].isin(fixed)
        cand = cand[~forced | (cand["plant"] == cand["order_id"].map(fixed))]
    cand = cand.reset_index(drop=True)

    order_ids = sorted(candidates["order_id"].unique())
    prob = pulp.LpProblem("allocation", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(len(cand))]
    u = {o: pulp.LpVariable(f"u_{n}", cat="Binary") for n, o in enumerate(order_ids)}

    by_order = candidates.groupby("order_id")
    spread = (w["cost"] * (by_order["cost"].max() - by_order["cost"].min()).sum() / ref_cost
              + w["carbon"] * (by_order["carbon_kg"].max() - by_order["carbon_kg"].min()).sum()
              / ref_carbon)
    late_penalty = LATE_ORDER + float(spread)

    late = (~cand["on_time"]).astype(float)
    coef = (w["cost"] * cand["cost"] / ref_cost
            + w["carbon"] * cand["carbon_kg"] / ref_carbon
            + w["lateness"] * (late_penalty * late + LATE_DAY * cand["days_late"])
            + DEFER * cand["deferred_weeks"])
    if base_plan is not None and not base_plan.empty:
        before = dict(zip(base_plan["order_id"], base_plan["plant"]))
        before_week = dict(zip(base_plan["order_id"], base_plan["week_start"]))
        moved = ((cand["plant"] != cand["order_id"].map(before).fillna(cand["plant"]))
                 | (cand["week_start"] != cand["order_id"].map(before_week)
                    .fillna(cand["week_start"])))
        coef = coef + MOVE * moved.astype(float)

    prob += (pulp.lpSum(c * v for c, v in zip(coef, x))
             + pulp.lpSum(UNASSIGNED * late_penalty * v for v in u.values()))

    for o, rows in cand.groupby("order_id").groups.items():
        prob += pulp.lpSum(x[i] for i in rows) + u[o] == 1, f"assign_{o}"
    for o in set(order_ids) - set(cand["order_id"]):
        prob += u[o] == 1, f"no_option_{o}"

    cap = capacity.set_index(["plant", "week_start"])["capacity"]
    for (plant, week), rows in cand.groupby(["plant", "week_start"]).groups.items():
        limit = int(cap.get((plant, week), 0))
        prob += pulp.lpSum(cand.at[i, "qty"] * x[i] for i in rows) <= limit, \
            f"cap_{plant}_{week:%Y%m%d}"

    if carbon_cap_kg is not None:
        prob += pulp.lpSum(cand.at[i, "carbon_kg"] * x[i] for i in range(len(cand))) \
            <= carbon_cap_kg, "carbon_cap"

    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=TIME_LIMIT_S, gapAbs=GAP_ABS))
    # CBC reports "Optimal" even when stopped by the time limit with a feasible
    # solution; sol_status tells them apart (1 = proven optimal, 2 = feasible only).
    if prob.status == 1:
        status = "OPTIMAL" if prob.sol_status == 1 else "FEASIBLE_TIME_LIMIT"
    else:
        status = pulp.LpStatus[prob.status].upper()

    if status not in ("OPTIMAL", "FEASIBLE_TIME_LIMIT"):
        empty = cand.iloc[0:0]
        return Plan(status, empty, order_ids, w, capacity,
                    (time.perf_counter() - start) * 1000)

    chosen = [i for i, v in enumerate(x) if v.value() is not None and v.value() > 0.5]
    allocation = cand.loc[chosen].sort_values("order_id").reset_index(drop=True)
    unassigned = sorted(o for o, v in u.items() if v.value() is not None and v.value() > 0.5)
    options = {"carbon_cap_kg": carbon_cap_kg, "on_time_only": on_time_only,
               "fixed": fixed or {}, "stability": base_plan is not None}
    return Plan(status, allocation, unassigned, w, capacity,
                (time.perf_counter() - start) * 1000, options)


def cheapest_plant_baseline(candidates: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    """Level-2 rule: each order to its cheapest plant in its planned week, ignoring
    capacity and due dates. Used only to show what the solver fixes."""
    planned = candidates[candidates["deferred_weeks"] == 0]
    return planned.loc[planned.groupby("order_id")["cost"].idxmin()].reset_index(drop=True)
