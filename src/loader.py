"""Load the four data-pack CSVs (Annex B schema) into normalised DataFrames.

Annex B columns are required. Extra columns we know about are optional and get a
stated default when missing, so judge files in the plain Annex B schema still load:

- orders.production_week   fixed production week; if absent the solver picks the week
- plants.week_start        week the row applies to; if absent the row applies to every week
- plants.energy_price_per_kwh   if absent, energy cost is 0 (unit_cost assumed to include it)

Every default or unit conversion applied is recorded in DataPack.notes, so it can be
shown in the validation report and listed as an assumption in answers.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

REQUIRED = {
    "orders": ["order_id", "product", "qty", "destination", "due"],
    "plants": ["plant", "product", "weekly_capacity", "unit_cost",
               "energy_kwh_per_unit", "carbon_kg_per_kwh"],
    "transport_duties": ["plant", "destination", "transport_cost_per_unit",
                         "transit_days", "duty_pct", "carbon_kg_per_unit"],
    "capacity_events": ["event_id", "plant", "from_date", "to_date", "capacity_loss_pct"],
}

FILES = {table: f"{table}.csv" for table in REQUIRED}

# Older copies of our own pack used due_date; Annex B says due.
COLUMN_ALIASES = {"orders": {"due_date": "due"}}

DATE_COLUMNS = {
    "orders": ["due", "production_week"],
    "plants": ["week_start"],
    "capacity_events": ["from_date", "to_date"],
}

STRING_COLUMNS = ["order_id", "product", "destination", "plant", "event_id"]


class SchemaError(ValueError):
    """A file is missing columns that Annex B requires."""


@dataclass
class DataPack:
    orders: pd.DataFrame
    plants: pd.DataFrame
    transport: pd.DataFrame
    events: pd.DataFrame
    planning_start: pd.Timestamp
    notes: list[str] = field(default_factory=list)


def normalise(table: str, df: pd.DataFrame, notes: list[str] | None = None) -> pd.DataFrame:
    """Rename aliases, check required columns, strip strings and parse dates.

    Values that cannot be parsed become NaN/NaT; validate.validate_rows rejects them.
    """
    notes = notes if notes is not None else []
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=COLUMN_ALIASES.get(table, {}))

    missing = [c for c in REQUIRED[table] if c not in df.columns]
    if missing:
        raise SchemaError(f"{table}: missing required column(s) {missing}")

    for col in STRING_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    for col in DATE_COLUMNS.get(table, []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if table == "plants" and "energy_price_per_kwh" not in df.columns:
        df["energy_price_per_kwh"] = 0.0
        notes.append("plants: no energy_price_per_kwh column; energy cost set to 0 "
                     "(unit_cost assumed to include energy)")

    # Annex B calls these a "percentage" without fixing the scale. Our pack uses
    # fractions (0.25); a file using 25 would otherwise cost 25x the goods value.
    for pct_col in ("duty_pct", "capacity_loss_pct"):
        if pct_col in df.columns:
            df[pct_col] = pd.to_numeric(df[pct_col], errors="coerce")
            if (df[pct_col] > 1).any():
                df[pct_col] = df[pct_col] / 100
                notes.append(f"{table}.{pct_col}: values above 1 found, column read as "
                             "percent and divided by 100")
    return df


def read_table(table: str, path: Path, notes: list[str] | None = None) -> pd.DataFrame:
    return normalise(table, pd.read_csv(path), notes)


def empty_table(table: str) -> pd.DataFrame:
    return normalise(table, pd.DataFrame(columns=REQUIRED[table]))


def planning_start(orders: pd.DataFrame, plants: pd.DataFrame) -> pd.Timestamp:
    """PLANNING_START env var, else the first plant week, else the first production week,
    else the Monday of the week of the earliest due date minus 4 weeks."""
    if os.getenv("PLANNING_START"):
        return pd.Timestamp(os.environ["PLANNING_START"])
    if "week_start" in plants.columns and plants["week_start"].notna().any():
        return plants["week_start"].min()
    if "production_week" in orders.columns and orders["production_week"].notna().any():
        return orders["production_week"].min()
    first_due = orders["due"].min() - pd.Timedelta(weeks=4)
    return (first_due - pd.Timedelta(days=first_due.weekday())).normalize()


def load_all(data_dir: Path | str = DATA_DIR) -> DataPack:
    """Load all four tables from data_dir. capacity_events.csv may be absent
    (it only arrives at the data drop); it then loads as an empty table."""
    data_dir = Path(data_dir)
    notes: list[str] = []
    tables = {}
    for table, filename in FILES.items():
        path = data_dir / filename
        if path.exists():
            tables[table] = read_table(table, path, notes)
        elif table == "capacity_events":
            tables[table] = empty_table(table)
            notes.append("capacity_events.csv not found; no capacity events applied")
        else:
            raise FileNotFoundError(f"{path} not found")

    start = planning_start(tables["orders"], tables["plants"])
    return DataPack(
        orders=tables["orders"],
        plants=tables["plants"],
        transport=tables["transport_duties"],
        events=tables["capacity_events"],
        planning_start=start,
        notes=notes,
    )


if __name__ == "__main__":
    pack = load_all()
    for name in ("orders", "plants", "transport", "events"):
        df = getattr(pack, name)
        print(f"{name}: {len(df)} rows, {len(df.columns)} cols")
    print(f"planning start: {pack.planning_start.date()}")
    for note in pack.notes:
        print(f"note: {note}")
