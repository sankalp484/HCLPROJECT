"""LangGraph orchestration for POST /analyze.

    parse (rules -> LLM fallback) -> guard -> one tool node -> check -> write -> END

Why this shape: every planning step is deterministic (solver, comparisons, validation),
so it lives in plain tool nodes, not agents. The LLM is used in only two places: as a
fallback parser for requests the rules do not understand, and to word explanations,
where the faithfulness check guards every number. A second "agent" would add cost and
failure modes without adding capability.
"""

import operator
import time
from typing import Annotated, Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from src import analysis as A
from src import writer as W
from src.intent import Decision, Intent, guard, parse
from src.models import edge_case_profile, weekly_capacity
from src.solver import plan_kpis

LIMITATIONS = [
    "Duty rates, transport costs and transit times are constant for the period",
    "Orders are built whole in one plant and one week (no splitting)",
    "Orders are never built before their production week; deferral stays inside the "
    "planning horizon",
    "Solver stops within 0.01% of the optimal objective or after 20 s (status reported)",
]
ASSUMPTIONS_BASE = [
    "Arrival = production week start + 6 days + transit days",
    "Duty charged on goods value (production + energy); transport not dutiable",
    "Weekly capacity is shared by all products of a plant",
    "Capacity events remove capacity pro rata by the days they overlap each week",
]


class State(TypedDict, total=False):
    run_id: str
    request: str
    weights: dict
    scenario: dict
    engine: Any
    decision: Decision
    current_events: Any
    answer_type: str
    plan: Any
    facts: dict
    out: dict  # response fields produced by tool nodes
    steps: Annotated[list, operator.add]
    llm: Annotated[list, operator.add]
    assumptions: Annotated[list, operator.add]


def step(agent: str, tool: str, t0: float, inp: dict | None = None,
         out: dict | None = None) -> dict:
    return {"agent": agent, "tool": tool, "input": inp or {}, "output": out or {},
            "ms": round((time.perf_counter() - t0) * 1000, 1)}


def label_weights(w: dict) -> str:
    return f"weights {w['cost'] * 100:.0f}% cost / {w['carbon'] * 100:.0f}% carbon"


# ---------- parse + guard ----------

def custom_events(changes, engine) -> pd.DataFrame:
    weeks = weekly_capacity(engine.pack, engine.no_events)["week_start"]
    first, last = weeks.min(), weeks.max() + pd.Timedelta(days=6)
    rows = [{"event_id": f"REQ-{i + 1}", "plant": c.plant,
             "from_date": pd.Timestamp(c.from_date) if c.from_date else first,
             "to_date": pd.Timestamp(c.to_date) if c.to_date else last,
             "capacity_loss_pct": c.loss_pct} for i, c in enumerate(changes)]
    return pd.DataFrame(rows)


def intent_from_scenario(sc: dict) -> Intent | None:
    if sc.get("capacity_changes"):
        return Intent(action="capacity_loss", capacity_changes=sc["capacity_changes"])
    if sc.get("event_ids") or sc.get("replan_events"):
        return Intent(action="apply_events", event_ids=sc.get("event_ids") or [])
    if sc.get("fixed_assignments"):
        return Intent(action="what_if_assign")
    if sc.get("carbon_cut_pct") is not None:
        return Intent(action="tradeoff", carbon_cut_pct=sc["carbon_cut_pct"])
    return None


def parse_node(state: State) -> dict:
    t0 = time.perf_counter()
    e = state["engine"]
    pack = e.pack
    sc = state.get("scenario") or {}
    llm_usage = []

    intent = intent_from_scenario(sc)
    source = "scenario"
    if intent is None:
        intent, source, usage = parse(state["request"], e.plants,
                                      sorted(pack.orders["product"].unique()),
                                      pack.planning_start)
        if usage.get("calls"):
            llm_usage.append({"step": "parse_intent", **usage})

    eligible = pack.plants[["plant", "product"]].drop_duplicates()
    decision = guard(intent, source, plants=e.plants,
                     order_ids=sorted(pack.orders["order_id"]),
                     event_ids=sorted(pack.events["event_id"]), orders=pack.orders,
                     eligible=eligible, planning_start=pack.planning_start)

    if sc.get("fixed_assignments") and decision.outcome == "run":
        fixed = {str(k): str(v).upper() for k, v in sc["fixed_assignments"].items()}
        problems = A.ineligible_assignments(e, fixed)
        unknown = [o for o in fixed if o not in set(pack.orders["order_id"])]
        if unknown:
            decision.outcome, decision.message = "clarification_needed", \
                f"Unknown order(s): {', '.join(unknown)}"
        elif problems:
            decision.outcome, decision.message = "refused", \
                "That breaks the eligibility rule: " + "; ".join(problems)
        decision.fixed = fixed

    apply_loaded = sc.get("apply_loaded_events", True)
    current = pack.events if apply_loaded else e.no_events
    assumptions = list(decision.assumptions)
    if apply_loaded and not pack.events.empty:
        assumptions.append("Loaded capacity events applied: "
                           + ", ".join(sorted(pack.events["event_id"])))
    return {"decision": decision, "current_events": current, "llm": llm_usage,
            "assumptions": assumptions,
            "steps": [step("intent", "parse_intent", t0, {"request": state["request"][:300]},
                           {"action": intent.action, "source": source,
                            "outcome": decision.outcome,
                            "intent": intent.model_dump(mode="json", exclude_defaults=True)})]}


def route(state: State) -> str:
    d = state["decision"]
    if d.outcome != "run":
        return d.outcome
    return d.intent.action


# ---------- tool nodes ----------

def clarify_node(state: State) -> dict:
    d = state["decision"]
    return {"answer_type": "clarification_needed", "out": {"summary": d.message}}


def refuse_node(state: State) -> dict:
    d = state["decision"]
    return {"answer_type": "refused", "out": {"summary": d.message}}


def allocate_node(state: State) -> dict:
    t0 = time.perf_counter()
    e, w = state["engine"], state["weights"]
    plan = e.plan(w, state["current_events"])
    alloc = plan.allocation
    label = "Plan"
    if state["decision"].intent.this_week_only:
        week = e.pack.planning_start
        orders = e.pack.orders
        if "production_week" in orders.columns and orders["production_week"].notna().any():
            ids = orders.loc[orders["production_week"] == week, "order_id"]
        else:
            ids = alloc.loc[alloc["week_start"] == week, "order_id"]
        alloc = alloc[alloc["order_id"].isin(ids)]
        label = f"This week's plan (orders for week {week:%Y-%m-%d})"
    kpis = plan_kpis(alloc, plan.capacity, [o for o in plan.unassigned])
    answer = "infeasible" if plan.unassigned else "allocation"
    facts = {"kpis": kpis, "status": plan.status, "late": alloc.loc[~alloc["on_time"],
             ["order_id", "plant", "days_late"]].to_dict("records")}
    return {"plan": plan, "answer_type": answer, "facts": facts,
            "out": {"allocation": A.allocation_records(alloc), "kpis": kpis,
                    "summary": W.summary_allocation(kpis, label)},
            "steps": [step("optimiser", "solve_allocation", t0, {"weights": w},
                           {"status": plan.status, "solve_ms": round(plan.solve_ms),
                            "late_orders": kpis["late_orders"],
                            "unassigned": plan.unassigned})]}


def tradeoff_node(state: State) -> dict:
    t0 = time.perf_counter()
    e, w = state["engine"], state["weights"]
    cut = state["decision"].intent.carbon_cut_pct
    t = A.tradeoff(e, state["current_events"], cut, w["lateness"])
    plan = t.get("requested_plan") or e.plan(w, state["current_events"])
    table = [{"carbon_cut_pct": r["carbon_cut_pct"], "carbon_weight": r["carbon_weight"],
              "cost": r["cost"], "carbon_t": r["carbon_t"], "late_orders": r["late_orders"],
              "extra_cost": r["extra_cost"],
              "implied_carbon_price_inr_per_t": r["implied_carbon_price_inr_per_t"]}
             for r in t["rows"]]
    if t.get("requested"):
        table.append({**t["requested"], "carbon_weight": None, "requested": True})
    facts = {k: v for k, v in t.items() if k != "requested_plan"}
    answer = "scenario" if t["feasible"] else "infeasible"
    return {"plan": plan, "answer_type": answer, "facts": facts,
            "out": {"tradeoff": table, "kpis": plan.kpis,
                    "allocation": A.allocation_records(plan.allocation),
                    "summary": W.summary_tradeoff(t)},
            "steps": [step("analyst", "tradeoff_frontier", t0, {"carbon_cut_pct": cut},
                           {"max_cut_pct": t["max_cut_pct"], "feasible": t["feasible"],
                            "points": len(t["rows"])})]}


def cheapest_node(state: State) -> dict:
    t0 = time.perf_counter()
    e = state["engine"]
    r = A.cheapest_on_time(e, state["current_events"])
    plan = r["plan"]
    k = plan.kpis
    if r["feasible"]:
        summary = W.summary_allocation(k, "Cheapest plan with every order on time")
    else:
        summary = (f"No plan gets every order on time: {len(r['orders_cannot_be_on_time'])} "
                   f"orders ({', '.join(r['orders_cannot_be_on_time'])}) are late from every "
                   "plant with the current capacity. Closest option: "
                   + W.summary_allocation(k, "cheapest delivery-first plan"))
    facts = {"kpis": k, **{x: r[x] for x in ("feasible", "orders_cannot_be_on_time",
                                              "closest_option_for_failing_orders")}}
    return {"plan": plan, "answer_type": "allocation" if r["feasible"] else "infeasible",
            "facts": facts,
            "out": {"allocation": A.allocation_records(plan.allocation), "kpis": k,
                    "summary": summary,
                    "infeasible_orders": r["closest_option_for_failing_orders"]},
            "steps": [step("optimiser", "solve_allocation", t0,
                           {"weights": {"cost": 1, "carbon": 0}, "on_time_only": True},
                           {"feasible": r["feasible"],
                            "cannot_be_on_time": r["orders_cannot_be_on_time"]})]}


def _scenario_out(state: State, s: dict, title: str, t0: float, tool_in: dict) -> dict:
    c = s["comparison"]
    new = s["new"]
    answer = "infeasible" if c["newly_unassigned"] else "scenario"
    facts = {"comparison": {k: v for k, v in c.items() if k != "moved"},
             "moved_orders": c["moved"][:40], "capacity_changes": s.get("capacity_changes", [])}
    out = {"allocation": A.allocation_records(new.allocation), "kpis": new.kpis,
           "summary": W.summary_scenario(title, c),
           "scenario_comparison": {**{k: v for k, v in c.items()
                                      if k not in ("base_kpis", "new_kpis")},
                                   "base_kpis": c["base_kpis"],
                                   "capacity_changes": s.get("capacity_changes", [])}}
    return {"plan": new, "answer_type": answer, "facts": facts, "out": out,
            "steps": [step("optimiser", "solve_allocation", t0, tool_in,
                           {"base_status": s["base"].status, "new_status": new.status}),
                      step("scenario", "compare_plans", t0, {},
                           {"extra_cost": c["extra_cost"], "extra_carbon_t": c["extra_carbon_t"],
                            "extra_late_orders": c["extra_late_orders"],
                            "orders_moved": c["orders_moved"]})]}


def capacity_node(state: State) -> dict:
    t0 = time.perf_counter()
    e, w = state["engine"], state["weights"]
    changes = state["decision"].intent.capacity_changes
    extra = custom_events(changes, e)
    current = state["current_events"]
    new_events = pd.concat([current, extra], ignore_index=True) if not current.empty else extra
    stable = (state.get("scenario") or {}).get("stability", True)
    s = A.capacity_scenario(e, w, current, new_events, stability=stable)
    title = "Re-plan for " + ", ".join(
        f"Plant {c.plant} -{c.loss_pct * 100:.0f}%" for c in changes)
    out = _scenario_out(state, s, title, t0,
                        {"weights": w, "capacity_changes": [c.model_dump(mode="json")
                                                            for c in changes],
                         "stability": stable})
    out["facts"]["requested_capacity_changes"] = [c.model_dump(mode="json") for c in changes]
    return out


def events_node(state: State) -> dict:
    t0 = time.perf_counter()
    e, w = state["engine"], state["weights"]
    loaded = e.pack.events
    ids = state["decision"].intent.event_ids or sorted(loaded["event_id"])
    base_events = loaded[~loaded["event_id"].isin(ids)]
    new_events = loaded
    stable = (state.get("scenario") or {}).get("stability", True)
    s = A.capacity_scenario(e, w, base_events, new_events, stability=stable)
    return _scenario_out(state, s, "Re-plan for capacity events " + ", ".join(ids), t0,
                         {"weights": w, "event_ids": ids, "stability": stable})


def assign_node(state: State) -> dict:
    t0 = time.perf_counter()
    e, w = state["engine"], state["weights"]
    fixed = state["decision"].fixed
    s = A.what_if_assign(e, w, state["current_events"], fixed)
    title = "What-if " + ", ".join(f"{o} -> Plant {p}" for o, p in list(fixed.items())[:5]) \
        + (" ..." if len(fixed) > 5 else "")
    return _scenario_out(state, s, title, t0, {"weights": w, "fixed": fixed})


def explain_node(state: State) -> dict:
    t0 = time.perf_counter()
    e, w = state["engine"], state["weights"]
    plan = e.plan(w, state["current_events"])
    oid = state["decision"].intent.order_id
    facts = A.explain_order(e, plan, oid)
    asked = state["decision"].intent.assign_plant
    facts["asked_plant"] = asked.upper() if asked else None
    row = plan.allocation[plan.allocation["order_id"] == oid]
    return {"plan": plan, "answer_type": "allocation", "facts": facts,
            "out": {"allocation": A.allocation_records(row), "kpis": plan.kpis,
                    "summary": W.text_explain(facts), "explanation": facts},
            "steps": [step("analyst", "explain_order", t0, {"order_id": oid},
                           {"assigned": facts["assigned"],
                            "options": len(facts["options"])})]}


def brief_node(state: State) -> dict:
    t0 = time.perf_counter()
    e, w = state["engine"], state["weights"]
    current = state["current_events"]
    plan = e.plan(w, current)
    steps = [step("optimiser", "solve_allocation", t0, {"weights": w}, {"status": plan.status})]
    t1 = time.perf_counter()
    t = A.tradeoff(e, current, None, w["lateness"])
    steps.append(step("analyst", "tradeoff_frontier", t1, {}, {"max_cut_pct": t["max_cut_pct"]}))
    facts = {
        "weights_label": label_weights(w),
        "plan_kpis": plan.kpis,
        "baseline": A.baseline_comparison(e, plan),
        "tradeoff": {k: v for k, v in t.items() if k != "requested_plan"},
        "implied_carbon_price": A.implied_carbon_price(e, w) or 0,
        "cannot_be_on_time": edge_case_profile(e.pack, e.candidates)["no_on_time_option"],
        "event_ids": sorted(current["event_id"]) if not current.empty else [],
    }
    if not current.empty:
        t2 = time.perf_counter()
        c = A.compare_plans(e.plan(w, e.no_events), plan)
        facts["events_impact"] = {k: v for k, v in c.items() if k != "moved"}
        steps.append(step("scenario", "compare_plans", t2, {"events": facts["event_ids"]},
                          {"extra_cost": c["extra_cost"], "extra_carbon_t": c["extra_carbon_t"]}))
    return {"plan": plan, "answer_type": "brief", "facts": facts,
            "out": {"kpis": plan.kpis, "allocation": A.allocation_records(plan.allocation),
                    "tradeoff": t["rows"], "summary": W.text_brief(facts)},
            "steps": steps}


# ---------- check + write ----------

def check_node(state: State) -> dict:
    plan = state.get("plan")
    if plan is None:
        return {}
    t0 = time.perf_counter()
    v = state["engine"].check(plan)
    out = dict(state["out"])
    out["validator"] = {k: v[k] for k in v if k.endswith("_ok") or k == "valid"}
    return {"out": out, "steps": [step("checker", "validate_plan", t0, {}, out["validator"])]}


NARRATED = {"brief": "brief", "scenario": "default", "infeasible": "default"}


def write_node(state: State) -> dict:
    t0 = time.perf_counter()
    answer = state["answer_type"]
    out = dict(state["out"])
    facts = state.get("facts") or {}
    action = state["decision"].intent.action
    kind = NARRATED.get(answer) or ("default" if action == "explain_order" else None)
    llm_usage, step_out = [], {"source": "template"}
    if kind and facts:
        n = W.narrate(kind, state["request"], facts, out["summary"])
        out["narrative"] = n["text"]
        llm_usage = [{"step": "write", **u} for u in n["llm_usage"]]
        step_out = {"source": n["source"], "unsupported_numbers": n["unsupported_numbers"],
                    "faithful": n["source"] == "template" or not n["unsupported_numbers"]}
    out["recommendations"] = recommendations(state, out)
    if facts:  # every number in summary/narrative traces to these computed facts
        out["evidence"] = W.jsonable(facts)
    tool = "write_brief" if answer == "brief" else "write_answer"
    return {"out": out, "llm": llm_usage, "steps": [step("writer", tool, t0, {}, step_out)]}


def recommendations(state: State, out: dict) -> list[dict]:
    answer = state["answer_type"]
    k = out.get("kpis") or {}
    v = out.get("validator") or {}
    recs = []
    if answer in ("clarification_needed", "refused"):
        return recs
    if answer in ("allocation", "brief") and k:
        recs.append(W.rec(
            "Approve the allocation plan" if v.get("valid", True) else "Do not release: "
            "the plan failed validation", "Solver plan respects capacity and eligibility; "
            "late orders minimised first, then weighted cost and carbon",
            ["kpis.total_cost", "kpis.total_carbon_t", "validator"],
            "high" if v.get("valid") else "low",
            f"cost {W.money(k['total_cost'])}, carbon {W.tonnes(k['total_carbon_t'])}", True))
    late = [a["order_id"] for a in out.get("allocation", []) if not a["on_time"]]
    if late:
        recs.append(W.rec(f"Tell customers now about {len(late)} late order(s)",
                          "No feasible plant and week delivers these by their due date",
                          [f"allocation[{o}]" for o in late[:10]], "high",
                          f"{len(late)} orders late", False))
    if answer in ("scenario", "infeasible") and out.get("scenario_comparison"):
        c = out["scenario_comparison"]
        recs.append(W.rec("Adopt the re-plan" if not c["newly_unassigned"] else
                          "Adopt the re-plan and escalate orders that cannot be placed",
                          f"{c['orders_moved']} orders move; the rest of the plan is unchanged",
                          ["scenario_comparison.extra_cost", "scenario_comparison.moved"],
                          "high", f"cost {W.signed_money(c['extra_cost'])}, carbon "
                          f"{c['extra_carbon_t']:+,.1f} t", True))
    if out.get("tradeoff") and answer != "brief":
        recs.append(W.rec("Review the carbon trade-off before changing weights",
                          "Each row is an optimal plan; costs rise as carbon falls",
                          ["tradeoff"], "medium", "see tradeoff table", True))
    return recs


# ---------- graph ----------

def build_graph():
    g = StateGraph(State)
    nodes = {"parse": parse_node, "clarification_needed": clarify_node, "refused": refuse_node,
             "allocate": allocate_node, "tradeoff": tradeoff_node,
             "cheapest_on_time": cheapest_node, "capacity_loss": capacity_node,
             "apply_events": events_node, "what_if_assign": assign_node,
             "explain_order": explain_node, "brief": brief_node,
             "check": check_node, "write": write_node}
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.add_edge(START, "parse")
    tools = [n for n in nodes if n not in ("parse", "check", "write")]
    g.add_conditional_edges("parse", route, {n: n for n in tools})
    for n in tools:
        g.add_edge(n, "check")
    g.add_edge("check", "write")
    g.add_edge("write", END)
    return g.compile()
