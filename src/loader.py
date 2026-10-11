import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

def load_orders() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "orders.csv",
                     parse_dates=["due_date", "production_week"])   # ← FIXED
    return df

def load_plants() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "plants.csv", parse_dates=["week_start"])
    return df

def load_transport() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "transport_duties.csv")

def load_capacity_events() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "capacity_events.csv",
                     parse_dates=["from_date", "to_date"])
    return df

def load_all() -> dict:
    return {
        "orders": load_orders(),
        "plants": load_plants(),
        "transport": load_transport(),
        "events": load_capacity_events(),
    }

if __name__ == "__main__":
    data = load_all()
    for name, df in data.items():
        print(f"{name}: {len(df)} rows, {len(df.columns)} cols")