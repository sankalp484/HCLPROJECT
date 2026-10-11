"""Evaluation harness (guide Section 7).

Usage: python eval/run_eval.py [--live-llm]

Runs every case in eval/cases.json through the same service.analyze the API uses, checks
each expectation (exact match on structured fields; the faithfulness checker on text),
then runs configuration comparisons and writes eval/report.md (+ eval/results/*.json).
Expected results come from how the data was built and from hand-solved datasets, never
from this system's own output.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASETS = {"pack": ROOT / "data", "tiny": ROOT / "eval/datasets/tiny",
            "annexb_minimal": ROOT / "eval/datasets/annexb_minimal"}
RESULTS = ROOT / "eval/results"


def use_dataset(name: str):
    """Point the store at a fresh SQLite file seeded from the dataset."""
    from src import service, store
    RESULTS.mkdir(parents=True, exist_ok=True)
    db = RESULTS / f"{name}.db"
    for suffix in ("", "-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True)
    store.DB_PATH = db
    store.init_db(seed_dir=DATASETS[name])
    service._engine = None


def text_of(resp: dict) -> str:
    return "\n".join(filter(None, [resp.get("summary"), resp.get("narrative")]))


def facts_of(resp: dict) -> dict:
    return {k: v for k, v in resp.items() if k not in ("summary", "narrative",
                                                        "assumptions", "limitations")}


def check(case: dict, resp: dict, orders_csv) -> list[str]:
    """Return failed expectations (empty = pass)."""
    from src.writer import check_numbers
    e, fails = case["expect"], []
    alloc = {a["order_id"]: a for a in resp.get("allocation", [])}
    k = resp.get("kpis") or {}
    sc = resp.get("scenario_comparison") or {}

    want = e.get("answer_type")
    if want and resp["answer_type"] not in (want if isinstance(want, list) else [want]):
        fails.append(f"answer_type {resp['answer_type']} != {want}")
    if e.get("valid") and not (resp.get("validator") or {}).get("valid"):
        fails.append(f"plan invalid: {resp.get('validator')}")
    if "late_orders" in e and k.get("late_orders") != e["late_orders"]:
        fails.append(f"late_orders {k.get('late_orders')} != {e['late_orders']}")
    if "total_cost" in e and abs(k.get("total_cost", -1) - e["total_cost"]) > 0.01:
        fails.append(f"total_cost {k.get('total_cost')} != {e['total_cost']}")
    for oid, plant in e.get("plant", {}).items():
        got = alloc.get(oid, {}).get("plant")
        if got != plant:
            fails.append(f"{oid} in {got}, expected {plant}")
    for oid, plant in e.get("not_plant", {}).items():
        if alloc.get(oid, {}).get("plant") == plant:
            fails.append(f"{oid} should not be in {plant}")
    for oid in e.get("on_time", []):
        if not alloc.get(oid, {}).get("on_time"):
            fails.append(f"{oid} not on time")
    for oid in e.get("late_includes", []):
        if alloc.get(oid, {}).get("on_time", True):
            fails.append(f"{oid} expected late")
    if "infeasible_orders" in e:
        got = sorted(o["order_id"] for o in resp.get("infeasible_orders", []))
        if got != sorted(e["infeasible_orders"]):
            fails.append(f"infeasible_orders {got} != {e['infeasible_orders']}")
    if "extra_late_orders" in e and sc.get("extra_late_orders") != e["extra_late_orders"]:
        fails.append(f"extra_late_orders {sc.get('extra_late_orders')} != "
                     f"{e['extra_late_orders']}")
    if "extra_late_orders_min" in e and sc.get("extra_late_orders", -99) \
            < e["extra_late_orders_min"]:
        fails.append(f"extra_late_orders {sc.get('extra_late_orders')} < "
                     f"{e['extra_late_orders_min']}")
    text = text_of(resp)
    for s in e.get("text_contains", []):
        if s not in text:
            fails.append(f"text missing '{s}'")
    for s in e.get("text_not_contains", []):
        if s in text:
            fails.append(f"text contains '{s}'")
    if e.get("faithful"):
        bad = check_numbers(text, facts_of(resp))
        if bad:
            fails.append(f"unsupported numbers {bad}")
    for dest, plant in e.get("destination_not_in_plant", {}).items():
        ids = set(orders_csv.loc[orders_csv["destination"] == dest, "order_id"])
        hits = [o for o in ids if alloc.get(o, {}).get("plant") == plant]
        if hits:
            fails.append(f"{dest} orders in {plant}: {sorted(hits)}")
    for plant, (lo, hi) in e.get("utilisation_between", {}).items():
        u = k.get("plant_utilisation", {}).get(plant, -1)
        if not lo <= u <= hi:
            fails.append(f"utilisation {plant} {u} not in [{lo}, {hi}]")
    return fails


def run_case(case: dict) -> dict:
    import pandas as pd

    from src import service, store
    orders_csv = pd.read_csv(DATASETS[case["dataset"]] / "orders.csv")
    if case.get("type") == "weights_sweep":
        t0 = time.perf_counter()
        resps = [service.analyze(case["request"], w, case.get("scenario"))
                 for w in case["sweep"]]
        ks = [r["kpis"] for r in resps]
        fails = []
        if not all(ks[i]["total_cost"] <= ks[i + 1]["total_cost"] + 1e-6 and
                   ks[i]["total_carbon_t"] >= ks[i + 1]["total_carbon_t"] - 1e-9
                   for i in range(len(ks) - 1)):
            fails.append(f"not monotonic: {[(k['total_cost'], k['total_carbon_t']) for k in ks]}")
        plans = [tuple(sorted((a["order_id"], a["plant"], a["week"]) for a in r["allocation"]))
                 for r in resps]
        if len(set(plans)) != len(plans):
            fails.append("weight settings gave identical plans")
        fails += [f"plan invalid ({r['run_id']})" for r in resps
                  if not r["validator"].get("valid")]
        latency = (time.perf_counter() - t0) * 1000 / len(resps)
        traces = [store.get_run(r["run_id"]) for r in resps]
        return {"id": case["id"], "category": case["category"], "passed": not fails,
                "fails": fails, "answer_type": "allocation", "latency_ms": latency,
                "llm_calls": sum(t["llm_calls"] for t in traces) / len(traces),
                "tokens": sum(t["tokens"] for t in traces) / len(traces),
                "valid": all(r["validator"].get("valid") for r in resps), "has_plan": True,
                "kpis": ks}
    t0 = time.perf_counter()
    resp = service.analyze(case["request"], case.get("weights"), case.get("scenario"))
    latency = (time.perf_counter() - t0) * 1000
    trace = store.get_run(resp["run_id"]) or {}
    fails = check(case, resp, orders_csv)
    from src.writer import check_numbers
    return {"id": case["id"], "category": case["category"], "passed": not fails,
            "fails": fails, "answer_type": resp["answer_type"], "run_id": resp["run_id"],
            "latency_ms": latency, "llm_calls": trace.get("llm_calls", 0),
            "tokens": trace.get("tokens", 0), "has_plan": bool(resp.get("validator")),
            "valid": (resp.get("validator") or {}).get("valid"),
            "late_orders": (resp.get("kpis") or {}).get("late_orders"),
            "has_evidence": "evidence" in resp,
            "text_numbers_unsupported": check_numbers(text_of(resp), facts_of(resp))
            if "evidence" in resp else [],
            "intent_source": trace.get("intent_source"),
            "summary": resp["summary"][:300]}


def pct(values, q):
    values = sorted(values)
    if not values:
        return 0
    idx = min(len(values) - 1, max(0, round(q / 100 * (len(values) - 1))))
    return values[idx]


def run_suite(cases, label: str) -> list[dict]:
    results = []
    for name in DATASETS:
        subset = [c for c in cases if c["dataset"] == name]
        if not subset:
            continue
        use_dataset(name)
        for c in subset:
            r = run_case(c)
            results.append(r)
            print(f"[{label}] {r['id']:4} {'PASS' if r['passed'] else 'FAIL'} "
                  f"{r['answer_type']:20} {r['latency_ms']:8.0f} ms  {'; '.join(r['fails'])}")
    return results


def comparisons() -> dict:
    """Config comparisons on the data pack: optimiser vs simple rule; re-plan stability."""
    from src.analysis import (
        Engine,
        baseline_comparison,
        capacity_scenario,
        frontier,
        implied_carbon_price,
    )
    from src.loader import load_all
    from src.validate import clean_pack
    pack = clean_pack(load_all(DATASETS["pack"]))[0]
    e = Engine(pack)
    w = {"cost": 0.8, "carbon": 0.2}
    plan = e.plan(w, None)
    base = baseline_comparison(e, plan)
    cost_first = e.plan({"cost": 1, "carbon": 0}, None)
    out = {"optimiser_vs_rule": {
        "rule_cheapest_plant": {**{k: base["baseline_kpis"][k] for k in
                                   ("total_cost", "total_carbon_t", "late_orders")},
                                "valid": base["baseline_valid"],
                                "capacity_violations": len(base["baseline_capacity_violations"])},
        "optimiser_cost_first (reference)": {**{k: cost_first.kpis[k] for k in
                                                ("total_cost", "total_carbon_t", "late_orders")},
                                             "valid": e.check(cost_first)["valid"]},
        "optimiser_default_weights": {**{k: plan.kpis[k] for k in
                                         ("total_cost", "total_carbon_t", "late_orders")},
                                      "valid": e.check(plan)["valid"]}}}

    rows = []
    for ev_id in ("E001", "E002", "E003"):
        ev = pack.events[pack.events["event_id"] == ev_id]
        for stable in (True, False):
            t0 = time.perf_counter()
            s = capacity_scenario(e, w, None, ev, stability=stable)
            c = s["comparison"]
            rows.append({"event": ev_id, "stability": stable, "extra_cost": c["extra_cost"],
                         "extra_carbon_t": c["extra_carbon_t"],
                         "extra_late_orders": c["extra_late_orders"],
                         "orders_moved": c["orders_moved"],
                         "plan_unchanged_pct": c["plan_stability_pct"],
                         "valid": s["validation"]["valid"], "status": s["new"].status,
                         "solve_ms": round((time.perf_counter() - t0) * 1000)})
    out["replan_stability"] = rows

    pts = frontier(e, None)
    max_cut = pts[-1]["carbon_cut_pct"]
    for p in pts:
        p["share_of_max_cut_pct"] = round(100 * p["carbon_cut_pct"] / max_cut, 1) \
            if max_cut else 0.0
    out["weight_frontier"] = pts
    out["default_weights"] = {"weights": w, "implied_carbon_price_inr_per_t":
                              implied_carbon_price(e, w)}
    return out


def intent_comparison(cases) -> dict | None:
    """Rules vs LLM intent parsing on the text requests (needs a live LLM)."""
    from src.intent import llm_parse, rule_parse
    from src.llm import get_llm
    if not get_llm().enabled:
        return None
    import pandas as pd
    rows = []
    for c in cases:
        if c.get("scenario", {}).get("capacity_changes") or c.get("type"):
            continue
        r_t0 = time.perf_counter()
        r = rule_parse(c["request"]).action
        r_ms = (time.perf_counter() - r_t0) * 1000
        l_t0 = time.perf_counter()
        li, usage = llm_parse(c["request"], ["A", "B", "C"], ["P1", "P2", "P3", "P4", "P5"],
                              pd.Timestamp("2026-10-12"))
        rows.append({"id": c["id"], "rules": r, "llm": li.action if li else None,
                     "agree": li is not None and li.action == r,
                     "rules_ms": round(r_ms, 2),
                     "llm_ms": round((time.perf_counter() - l_t0) * 1000),
                     "llm_tokens": usage.get("total_tokens", 0)})
    agree = round(100 * sum(r["agree"] for r in rows) / len(rows), 1)
    return {"rows": rows, "agreement_pct": agree}


def metrics(results: list[dict]) -> dict:
    planned = [r for r in results if r["has_plan"]]
    lat = [r["latency_ms"] for r in results]
    texts = [r for r in results if r.get("has_evidence")]
    faithful = [r for r in texts if not r.get("text_numbers_unsupported")]
    return {
        "cases": len(results), "passed": sum(r["passed"] for r in results),
        "pass_rate_pct": round(100 * sum(r["passed"] for r in results) / len(results), 1),
        "plan_validity_pct": round(100 * sum(bool(r["valid"]) for r in planned)
                                   / max(len(planned), 1), 1),
        "brief_faithfulness_pct": round(100 * len(faithful) / max(len(texts), 1), 1),
        "latency_p50_ms": round(pct(lat, 50)), "latency_p95_ms": round(pct(lat, 95)),
        "llm_calls_per_request": round(statistics.mean(r["llm_calls"] for r in results), 2),
        "tokens_per_request": round(statistics.mean(r["tokens"] for r in results), 1),
        "by_category": {cat: f"{sum(r['passed'] for r in results if r['category'] == cat)}/"
                        f"{sum(1 for r in results if r['category'] == cat)}"
                        for cat in dict.fromkeys(r["category"] for r in results)},
    }


def write_report(m: dict, results: list[dict], comp: dict, intents: dict | None,
                 mode: str) -> str:
    L = [f"# Evaluation report\n\nGenerated by `python eval/run_eval.py` ({mode}). "
         f"{m['cases']} cases from `eval/cases.json`.\n",
         "**Method.** Exact match on structured fields (answer type, plant per order, "
         "late-order counts, infeasible order lists, plan validator). Text is checked by "
         "the faithfulness checker: every number in the summary or narrative must match a "
         "number in the response's computed fields. Expected values come from how the data "
         "was built (data card) or from datasets small enough to solve by hand "
         "(`eval/datasets/tiny`, `eval/datasets/annexb_minimal`), never from this system's "
         "own output.\n",
         "## Headline metrics\n", "| Metric | Value |", "|---|---|"]
    for key in ("pass_rate_pct", "plan_validity_pct", "brief_faithfulness_pct",
                "latency_p50_ms", "latency_p95_ms", "llm_calls_per_request",
                "tokens_per_request"):
        L.append(f"| {key} | {m[key]} |")
    L.append(f"| passed | {m['passed']}/{m['cases']} |\n")
    L += ["By category: " + ", ".join(f"{k} {v}" for k, v in m["by_category"].items()) + "\n",
          "## Cases\n", "| Case | Category | Result | Answer type | Late | ms | Notes |",
          "|---|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r['id']} | {r['category']} | {'PASS' if r['passed'] else 'FAIL'} | "
                 f"{r['answer_type']} | {r.get('late_orders', '')} | {r['latency_ms']:.0f} | "
                 f"{'; '.join(r['fails'])} |")

    o = comp["optimiser_vs_rule"]
    L += ["\n## Total cost and carbon: simple rule vs optimiser (data pack, no events)\n",
          "| Plan | Cost (INR) | Carbon (t) | Late orders | Valid |", "|---|---|---|---|---|"]
    for name, k in o.items():
        extra = f" ({k['capacity_violations']} plant-weeks over capacity)" \
            if "capacity_violations" in k else ""
        L.append(f"| {name} | {k['total_cost']:,.0f} | {k['total_carbon_t']:,.1f} | "
                 f"{k['late_orders']} | {k['valid']}{extra} |")
    L += ["\nThe cheapest-plant rule looks cheaper only because it ignores capacity and due "
          "dates; it is not a valid plan. The cost-first optimiser is the reference solution.",
          "\n## Re-plan quality: stability penalty on vs off\n",
          "| Event | Stability | Extra cost | Extra carbon (t) | Extra late | Moved | "
          "Unchanged % | Valid | Solve ms |", "|---|---|---|---|---|---|---|---|---|"]
    for r in comp["replan_stability"]:
        L.append(f"| {r['event']} | {r['stability']} | {r['extra_cost']:,.0f} | "
                 f"{r['extra_carbon_t']:+.2f} | {r['extra_late_orders']:+d} | "
                 f"{r['orders_moved']} | {r['plan_unchanged_pct']} | {r['valid']} | "
                 f"{r['solve_ms']} |")
    L += ["\n## Weight choice: the frontier\n",
          "| Carbon weight | Cost (INR) | Carbon (t) | Cut vs cheapest | Share of max cut | "
          "Extra cost | Implied INR/t | Late |", "|---|---|---|---|---|---|---|---|"]
    for p in comp["weight_frontier"]:
        L.append(f"| {p['carbon_weight']} | {p['cost']:,.0f} | {p['carbon_t']:,.1f} | "
                 f"{p['carbon_cut_pct']}% | {p['share_of_max_cut_pct']}% | "
                 f"{p['extra_cost']:,.0f} | {p['implied_carbon_price_inr_per_t'] or '-'} | "
                 f"{p['late_orders']} |")
    d = comp["default_weights"]
    L.append(f"\nDefault {d['weights']} implies a carbon price of INR "
             f"{d['implied_carbon_price_inr_per_t']:,.0f}/t.")
    if intents:
        L += ["\n## Intent parsing: rules vs LLM\n",
              f"Agreement {intents['agreement_pct']}%.\n",
              "| Case | Rules | LLM | Agree | Rules ms | LLM ms | LLM tokens |",
              "|---|---|---|---|---|---|---|"]
        for r in intents["rows"]:
            L.append(f"| {r['id']} | {r['rules']} | {r['llm']} | {r['agree']} | "
                     f"{r['rules_ms']} | {r['llm_ms']} | {r['llm_tokens']} |")
    else:
        L.append("\n## Intent parsing: rules vs LLM\n\nNot run: no live LLM configured "
                 "(set GEMINI_API_KEY and run with --live-llm).")
    return "\n".join(L) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-llm", action="store_true",
                        help="use the configured LLM (default: MOCK_LLM=true)")
    args = parser.parse_args()
    os.environ["MOCK_LLM"] = "false" if args.live_llm else "true"
    os.environ["WARM_UP"] = "false"
    from src.llm import get_llm, reset_llm
    reset_llm()
    mode = f"LLM {get_llm().model} live" if get_llm().enabled else "MOCK_LLM, no LLM calls"

    cases = json.loads((ROOT / "eval/cases.json").read_text(encoding="utf-8"))
    results = run_suite(cases, "eval")
    m = metrics(results)
    print(json.dumps(m, indent=2))
    comp = comparisons()
    intents = intent_comparison(cases)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(
        {"metrics": m, "results": results, "comparisons": comp, "intents": intents},
        indent=2, default=str), encoding="utf-8")
    (ROOT / "eval/report.md").write_text(write_report(m, results, comp, intents, mode),
                                         encoding="utf-8")
    print("wrote eval/report.md")
    return 0 if m["passed"] == m["cases"] else 1


if __name__ == "__main__":
    sys.exit(main())
