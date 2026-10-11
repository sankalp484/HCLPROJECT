"""Validate a data pack before any analysis.

Usage: python scripts/validate_data.py [--data-dir DIR]
Exits with code 1 if any error-level check fails.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loader import DATA_DIR, SchemaError, load_all  # noqa: E402
from src.models import build_candidates, edge_case_profile  # noqa: E402
from src.validate import clean_pack, format_report, is_valid, validate_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    try:
        pack = load_all(args.data_dir)
    except (SchemaError, FileNotFoundError) as e:
        print(f"[FAIL] {e}")
        return 1

    pack, rejected = clean_pack(pack)
    checks = validate_all(pack, rejected, data_card=args.data_dir / "data_card.md")
    print(format_report(checks, pack.notes))
    if not is_valid(checks):
        return 1

    profile = edge_case_profile(pack, build_candidates(pack))
    print("\nEDGE CASES FOUND IN THE DATA")
    for name, ids in profile.items():
        shown = ", ".join(ids[:8]) + (f" (+{len(ids) - 8} more)" if len(ids) > 8 else "")
        print(f"  {name}: {len(ids)}" + (f"  [{shown}]" if ids else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
