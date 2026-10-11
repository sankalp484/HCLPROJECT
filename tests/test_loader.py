import pandas as pd
import pytest

from src.loader import SchemaError, load_all, normalise


def test_annex_b_minimal_files_load(tmp_path):
    """Plain Annex B: no week_start, energy price, production_week or events file."""
    (tmp_path / "orders.csv").write_text(
        "order_id,product,qty,destination,due\nO1,P1,5,X,2026-11-30\n")
    (tmp_path / "plants.csv").write_text(
        "plant,product,weekly_capacity,unit_cost,energy_kwh_per_unit,carbon_kg_per_kwh\n"
        "A,P1,100,10,1,0.5\n")
    (tmp_path / "transport_duties.csv").write_text(
        "plant,destination,transport_cost_per_unit,transit_days,duty_pct,carbon_kg_per_unit\n"
        "A,X,1,3,0,0.1\n")
    pack = load_all(tmp_path)
    assert pack.plants["energy_price_per_kwh"].tolist() == [0.0]
    assert pack.events.empty
    assert any("energy_price_per_kwh" in n for n in pack.notes)
    assert any("capacity_events.csv not found" in n for n in pack.notes)
    assert pack.planning_start < pack.orders["due"].min()


def test_due_date_alias_and_whitespace():
    df = normalise("orders", pd.DataFrame({
        " order_id": [" O1 "], "product": ["P1"], "qty": [1], "destination": ["X"],
        "due_date": ["2026-11-01"]}))
    assert df.loc[0, "order_id"] == "O1"
    assert df.loc[0, "due"] == pd.Timestamp("2026-11-01")


def test_percent_columns_given_as_percent_are_scaled():
    notes = []
    df = normalise("transport_duties", pd.DataFrame({
        "plant": ["A", "B"], "destination": ["X", "X"], "transport_cost_per_unit": [1, 1],
        "transit_days": [1, 1], "duty_pct": [25, 0], "carbon_kg_per_unit": [0, 0]}), notes)
    assert df["duty_pct"].tolist() == [0.25, 0.0]
    assert notes


def test_missing_required_column_raises():
    with pytest.raises(SchemaError, match="qty"):
        normalise("orders", pd.DataFrame({"order_id": ["O1"], "product": ["P1"],
                                          "destination": ["X"], "due": ["2026-11-01"]}))


def test_planning_start_env_override(tiny_dir, monkeypatch):
    monkeypatch.setenv("PLANNING_START", "2026-10-01")
    assert load_all(tiny_dir).planning_start == pd.Timestamp("2026-10-01")
