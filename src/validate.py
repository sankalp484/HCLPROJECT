"""Data validation, run before any analysis (guide Section 4.5).

Two layers:
- validate_rows: per-row checks for one table. Bad rows are split off with a reason,
  which gives POST /ingest its rows_loaded / rows_rejected.
- validate_all: cross-table checks on the cleaned pack, producing the report.
"""

import re
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from src.loader import DataPack

CAPACITY_UNIT = "units_per_week"
CARBON_UNIT = "kgCO2e_per_kWh"

NUMERIC_RULES = {
    # column: (minimum, maximum, minimum is exclusive)
    "orders": {"qty": (0, None, True)},
    "plants": {
        "weekly_capacity": (0, None, True),
        "unit_cost": (0, None, False),
        "energy_kwh_per_unit": (0, None, False),
        "carbon_kg_per_kwh": (0, None, False),
        "energy_price_per_kwh": (0, None, False),
    },
    "transport_duties": {
        "transport_cost_per_unit": (0, None, False),
        "transit_days": (0, None, False),
        "duty_pct": (0, 1, False),
        "carbon_kg_per_unit": (0, None, False),
    },
    "capacity_events": {"capacity_loss_pct": (0, 1, False)},
}

INTEGER_COLUMNS = {"qty", "weekly_capacity", "transit_days"}

NOT_NULL = {
    "orders": ["order_id", "product", "destination", "due"],
    "plants": ["plant", "product"],
    "transport_duties": ["plant", "destination"],
    "capacity_events": ["event_id", "plant", "from_date", "to_date"],
}

UNIQUE_KEYS = {
    "orders": ["order_id"],
    "plants": ["plant", "product", "week_start"],
    "transport_duties": ["plant", "destination"],
    "capacity_events": ["event_id"],
}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    level: str = "error"  # "error" fails validation, "warning" is reported only


def validate_rows(table: str, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (valid_rows, rejected_rows). Rejected rows carry a 'reason' column.
    Numeric columns are coerced; values that do not parse are rejected."""
    df = df.copy().reset_index(drop=True)
    reasons = pd.Series("", index=df.index, dtype="string")

    def flag(mask: pd.Series, message: str) -> None:
        reasons[mask] = reasons[mask] + message + "; "

    for col in NOT_NULL[table]:
        flag(df[col].isna(), f"{col} missing")
    if table == "plants" and "week_start" in df.columns:
        flag(df["week_start"].isna(), "week_start missing or not a date")

    for col, (low, high, exclusive) in NUMERIC_RULES[table].items():
        if col not in df.columns:
            continue
        raw = df[col]
        df[col] = pd.to_numeric(raw, errors="coerce")
        flag(df[col].isna(), f"{col} missing or not a number")
        if exclusive:
            flag(df[col] <= low, f"{col} must be > {low}")
        else:
            flag(df[col] < low, f"{col} must be >= {low}")
        if high is not None:
            flag(df[col] > high, f"{col} must be <= {high}")
        if col in INTEGER_COLUMNS:
            flag(df[col].notna() & (df[col] % 1 != 0), f"{col} must be a whole number")

    if table == "capacity_events":
        flag(df["from_date"] > df["to_date"], "from_date is after to_date")

    key = [c for c in UNIQUE_KEYS[table] if c in df.columns]
    flag(df.duplicated(key, keep="first"), f"duplicate {'/'.join(key)}")

    bad = reasons != ""
    rejected = df[bad].assign(reason=reasons[bad].str.rstrip("; "))
    valid = df[~bad].copy()
    for col in INTEGER_COLUMNS & set(valid.columns):
        valid[col] = valid[col].astype("int64")
    # rejected keeps the original index, so index + 2 is the CSV line number
    return valid.reset_index(drop=True), rejected


def clean_pack(pack: DataPack) -> tuple[DataPack, dict[str, pd.DataFrame]]:
    """Run validate_rows on every table. Returns the cleaned pack and rejects by table."""
    rejected = {}
    cleaned = {}
    for attr, table in (("orders", "orders"), ("plants", "plants"),
                        ("transport", "transport_duties"), ("events", "capacity_events")):
        cleaned[attr], rejected[table] = validate_rows(table, getattr(pack, attr))
    return replace(pack, **cleaned), rejected


def read_data_card_counts(path: Path) -> dict[str, int]:
    """Row counts from the data card's table, e.g. '| orders.csv | 150 | ...'."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    return {m[1]: int(m[2]) for m in re.finditer(r"\|\s*(\w+)\.csv\s*\|\s*(\d+)", text)}


def _ids(values, limit: int = 8) -> str:
    values = sorted(map(str, values))
    more = f" (+{len(values) - limit} more)" if len(values) > limit else ""
    return ", ".join(values[:limit]) + more


def _unit_check(name: str, df: pd.DataFrame, col: str, expected: str) -> Check:
    if col not in df.columns:
        return Check(name, True, f"no {col} column; assumed {expected}", "warning")
    found = set(df[col].dropna().astype(str))
    return Check(name, found == {expected}, f"{col} = {_ids(found)}; expected {expected}")


def validate_all(pack: DataPack, rejected: dict[str, pd.DataFrame] | None = None,
                 data_card: Path | None = None) -> list[Check]:
    """Cross-table checks on a cleaned pack (see clean_pack)."""
    orders, plants, transport, events = pack.orders, pack.plants, pack.transport, pack.events
    rejected = rejected or {}
    checks = []

    for table, rej in rejected.items():
        detail = "; ".join(f"line {i + 2}: {r}" for i, r in zip(rej.index[:3], rej["reason"][:3]))
        checks.append(Check(f"rows_rejected_{table}", rej.empty,
                            f"{len(rej)} rejected" + (f" ({detail})" if detail else ""),
                            "warning"))

    # 1. Every order's product can be made in at least one plant
    unmakeable = orders.loc[~orders["product"].isin(plants["product"]), "order_id"]
    checks.append(Check("all_orders_makeable", unmakeable.empty,
                        f"no plant makes: {_ids(unmakeable)}" if len(unmakeable) else
                        f"{len(orders)} orders, products {_ids(orders['product'].unique())}"))

    # 1b. If an order's production week is fixed, some plant has data for that week
    if "production_week" in orders.columns and "week_start" in plants.columns:
        fixed = orders[orders["production_week"].notna()]
        have = fixed.merge(plants, left_on=["product", "production_week"],
                           right_on=["product", "week_start"])["order_id"]
        no_week = fixed.loc[~fixed["order_id"].isin(have), "order_id"]
        checks.append(Check("orders_have_plant_week", no_week.empty,
                            f"no plant data for production week: {_ids(no_week)}"
                            if len(no_week) else "every fixed production week has plant data"))

    # 2. Every plant and destination pair has a transport row
    needed = {(p, d) for p in plants["plant"].unique() for d in orders["destination"].unique()}
    missing = needed - set(zip(transport["plant"], transport["destination"]))
    checks.append(Check("transport_coverage", not missing,
                        f"missing pairs: {_ids(missing)}" if missing else
                        f"{len(needed)} plant/destination pairs covered"))

    # 3. Capacity is positive (row level) and in a stated unit
    checks.append(_unit_check("capacity_unit_stated", plants, "capacity_unit", CAPACITY_UNIT))

    # 3b. Capacity is shared by all products of a plant in a week, so it should be
    #     the same on every product row. If not, the model uses the minimum.
    week_key = ["plant", "week_start"] if "week_start" in plants.columns else ["plant"]
    spread = plants.groupby(week_key)["weekly_capacity"].nunique()
    uneven = spread[spread > 1]
    checks.append(Check("capacity_shared_per_plant_week", uneven.empty,
                        f"capacity differs across products for {_ids(uneven.index)}; "
                        "using the minimum" if len(uneven) else
                        "one capacity per plant per week", "warning"))

    # 4. Due dates are after the planning start
    early = orders.loc[orders["due"] <= pack.planning_start, "order_id"]
    checks.append(Check("due_after_planning_start", early.empty,
                        f"due on/before {pack.planning_start.date()}: {_ids(early)}"
                        if len(early) else f"planning start {pack.planning_start.date()}"))

    # 5. Carbon intensity is in a stated unit (non-negative is checked per row)
    checks.append(_unit_check("carbon_unit_stated", plants, "carbon_unit", CARBON_UNIT))

    # 6. Capacity events refer to known plants
    unknown = events.loc[~events["plant"].isin(plants["plant"]), "event_id"]
    checks.append(Check("events_known_plants", unknown.empty,
                        f"unknown plant in: {_ids(unknown)}" if len(unknown) else
                        f"{len(events)} events"))

    # 7. Row counts match the data card
    expected = read_data_card_counts(data_card) if data_card else {}
    if expected:
        actual = {
            "orders": len(orders) + len(rejected.get("orders", [])),
            "plants": len(plants) + len(rejected.get("plants", [])),
            "transport_duties": len(transport) + len(rejected.get("transport_duties", [])),
            "capacity_events": len(events) + len(rejected.get("capacity_events", [])),
        }
        diffs = [f"{t}: card {n}, file {actual[t]}" for t, n in expected.items()
                 if t in actual and actual[t] != n]
        checks.append(Check("row_counts_match_data_card", not diffs,
                            "; ".join(diffs) or "all counts match"))
    else:
        checks.append(Check("row_counts_match_data_card", True,
                            "no data card next to the data; skipped", "warning"))
    return checks


def is_valid(checks: list[Check]) -> bool:
    return all(c.ok for c in checks if c.level == "error")


def format_report(checks: list[Check], notes: list[str] | None = None) -> str:
    lines = ["=" * 60, "VALIDATION REPORT", "=" * 60]
    for c in checks:
        tag = "PASS" if c.ok else ("FAIL" if c.level == "error" else "WARN")
        lines.append(f"[{tag}] {c.name}: {c.detail}")
    for note in notes or []:
        lines.append(f"[NOTE] {note}")
    lines.append("=" * 60)
    lines.append("RESULT: " + ("VALID" if is_valid(checks) else "INVALID"))
    return "\n".join(lines)
