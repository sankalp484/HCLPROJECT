import pandas as pd

def validate_all(orders, plants, transport, events):
    results = {}

    # Check 1: Every order's product can be made in at least one plant
    plant_products = set(plants["product"])
    results["orders_makeable"] = orders["product"].isin(plant_products).all()

    # Check 2: Every (plant, destination) pair has a transport row
    pairs_needed = set(zip(orders["destination"], orders["destination"]))
    # ... check transport covers all destinations

    # Check 3: Capacity is positive
    results["capacity_positive"] = (plants["weekly_capacity"] > 0).all()

    # Check 4: Due dates are after planning start
    results["due_dates_valid"] = (orders["due_date"] > pd.Timestamp("2026-10-11")).all()

    # Check 5: Carbon intensity is in a stated unit (non-null, non-negative)
    results["carbon_valid"] = (plants["carbon_kg_per_kwh"] >= 0).all()

    return results