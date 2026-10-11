import pandas as pd

PLANNING_START = pd.Timestamp("2026-10-12")

def validate_all(orders, plants, transport, events) -> dict:
    results = {}

    # 1. Every order's product can be made in at least one plant
    makeable = set(plants["product"])
    results["all_orders_makeable"] = bool(orders["product"].isin(makeable).all())

    # 2. Every (plant, destination) pair has a transport row
    plants_list = plants["plant"].unique()
    dests = orders["destination"].unique()
    needed = {(p, d) for p in plants_list for d in dests}
    have = set(zip(transport["plant"], transport["destination"]))
    missing = needed - have
    results["transport_coverage"] = len(missing) == 0
    results["_missing_pairs"] = list(missing)[:5]

    # 3. Capacity is positive
    results["capacity_positive"] = bool((plants["weekly_capacity"] > 0).all())

    # 4. Due dates are after planning start
    results["due_dates_valid"] = bool((orders["due_date"] > PLANNING_START).all())  # ← FIXED

    # 5. Carbon intensity is non-negative
    results["carbon_valid"] = bool((plants["carbon_kg_per_kwh"] >= 0).all())

    return results

def print_report(results: dict):
    print("=" * 50)
    print("VALIDATION REPORT")
    print("=" * 50)
    for key, val in results.items():
        if key.startswith("_"):
            continue
        icon = "✅" if val else "❌"
        print(f"{icon}  {key}: {val}")
    if results.get("_missing_pairs"):
        print(f"\n⚠️  Missing (plant, destination) pairs: {results['_missing_pairs']}")
    print("=" * 50)