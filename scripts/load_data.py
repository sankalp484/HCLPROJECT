"""Load a data pack (Annex B CSVs) into the store, validating it first.

Usage: python scripts/load_data.py --data-dir path/to/csvs [--db data/app.db] [--force]

Reads orders.csv, plants.csv, transport_duties.csv and (optional) capacity_events.csv.
Invalid rows are skipped and listed. If an error-level check fails the store is left
unchanged unless --force is given. The running API picks up the new data on its next
request (the data version changes).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import store  # noqa: E402
from src.loader import SchemaError, load_all  # noqa: E402
from src.validate import clean_pack, format_report, is_valid, validate_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=store.DB_PATH)
    parser.add_argument("--force", action="store_true", help="store even if checks fail")
    args = parser.parse_args()

    try:
        pack = load_all(args.data_dir)
    except (SchemaError, FileNotFoundError) as e:
        print(f"[FAIL] {e}")
        return 1
    pack, rejected = clean_pack(pack)
    checks = validate_all(pack, rejected, data_card=args.data_dir / "data_card.md")
    print(format_report(checks, pack.notes))
    for table, rows in rejected.items():
        for line, reason in zip(rows.index[:10], rows["reason"][:10]):
            print(f"  rejected {table} line {line + 2}: {reason}")

    if not is_valid(checks) and not args.force:
        print("Not loaded: fix the failed checks or pass --force.")
        return 1
    store.init_db(seed_dir=None, path=args.db)
    version = store.replace_all(pack, args.db)
    print(f"Loaded into {args.db}: orders={len(pack.orders)}, plants={len(pack.plants)}, "
          f"transport_duties={len(pack.transport)}, capacity_events={len(pack.events)} "
          f"(data version {version})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
