from pathlib import Path

import pandas as pd

from src.loader import load_all, normalise
from src.validate import (
    clean_pack,
    is_valid,
    read_data_card_counts,
    validate_all,
    validate_rows,
)

DATA = Path(__file__).resolve().parent.parent / "data"


def by_name(checks):
    return {c.name: c for c in checks}


def test_row_level_rejects_with_line_numbers():
    df = normalise("orders", pd.DataFrame({
        "order_id": ["O1", "O2", "O1", "O4"], "product": ["P1"] * 4,
        "qty": [5, -1, 3, "abc"], "destination": ["X"] * 4,
        "due": ["2026-11-01", "2026-11-01", "2026-11-01", "not a date"]}))
    valid, rejected = validate_rows("orders", df)
    assert valid["order_id"].tolist() == ["O1"]
    reasons = dict(zip(rejected.index + 2, rejected["reason"]))
    assert "qty must be > 0" in reasons[3]
    assert "duplicate order_id" in reasons[4]
    assert "qty missing or not a number" in reasons[5] and "due missing" in reasons[5]


def test_event_rules():
    df = normalise("capacity_events", pd.DataFrame({
        "event_id": ["E1", "E2"], "plant": ["A", "A"],
        "from_date": ["2026-11-10", "2026-11-01"], "to_date": ["2026-11-01", "2026-11-05"],
        "capacity_loss_pct": [0.3, 0.5]}))
    valid, rejected = validate_rows("capacity_events", df)
    assert valid["event_id"].tolist() == ["E2"]
    assert "from_date is after to_date" in rejected["reason"].iloc[0]


def test_tiny_pack_is_valid(tiny_dir):
    pack, rejected = clean_pack(load_all(tiny_dir))
    assert is_valid(validate_all(pack, rejected))


def test_cross_table_failures(tiny_dir):
    pack, rejected = clean_pack(load_all(tiny_dir))
    extra = pd.DataFrame({"order_id": ["O9"], "product": ["P9"], "qty": [5],
                          "destination": ["Z"], "due": [pd.Timestamp("2026-10-01")],
                          "production_week": [pd.Timestamp("2026-10-12")]})
    pack.orders = pd.concat([pack.orders, extra], ignore_index=True)
    c = by_name(validate_all(pack, rejected))
    assert not c["all_orders_makeable"].ok and "O9" in c["all_orders_makeable"].detail
    assert not c["transport_coverage"].ok
    assert not c["due_after_planning_start"].ok
    assert not is_valid(list(c.values()))


def test_unit_mismatch_fails(tiny_dir):
    pack, rejected = clean_pack(load_all(tiny_dir))
    pack.plants["capacity_unit"] = "units_per_day"
    assert not by_name(validate_all(pack, rejected))["capacity_unit_stated"].ok


def test_data_card_counts(tmp_path, tiny_dir):
    card = tmp_path / "card.md"
    card.write_text("| orders.csv | 4 | x |\n| plants.csv | 99 | y |\n")
    assert read_data_card_counts(card) == {"orders": 4, "plants": 99}
    pack, rejected = clean_pack(load_all(tiny_dir))
    check = by_name(validate_all(pack, rejected, card))["row_counts_match_data_card"]
    assert not check.ok and "plants: card 99" in check.detail


def test_shipped_data_pack_is_valid():
    """data/ passes every check, including row counts against the data card."""
    pack, rejected = clean_pack(load_all(DATA))
    checks = validate_all(pack, rejected, DATA / "data_card.md")
    assert is_valid(checks), [c for c in checks if not c.ok]
