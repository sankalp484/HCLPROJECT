"""Show cost, carbon and arrival for one order in every eligible plant.

Usage: python scripts/compare_plants.py [--order ORD-001] [--data-dir DIR]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loader import DATA_DIR, load_all  # noqa: E402
from src.models import build_candidates  # noqa: E402
from src.validate import clean_pack  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", default="ORD-001")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    pack, _ = clean_pack(load_all(args.data_dir))
    rows = build_candidates(pack)
    rows = rows[rows["order_id"] == args.order]
    if rows.empty:
        sys.exit(f"No eligible plant found for {args.order}")

    first = rows.iloc[0]
    print(f"Order {args.order}: {first['qty']} x {first['product']} -> {first['destination']}, "
          f"due {first['due'].date()}")
    print("=" * 70)
    for r in rows.itertuples():
        status = "on time" if r.on_time else f"{r.days_late} days late"
        print(f"\nPlant {r.plant}, week {r.week_start.date()}: "
              f"arrives {r.arrival.date()} ({status})")
        print(f"  cost   = {r.cost:>12,.2f}  (production {r.production:,.2f}, "
              f"energy {r.energy:,.2f}, transport {r.transport:,.2f}, duty {r.duty:,.2f})")
        print(f"  carbon = {r.carbon_kg:>12,.2f} kg  (plant {r.carbon_plant_kg:,.2f}, "
              f"transport {r.carbon_transport_kg:,.2f})")
