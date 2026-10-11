import pandas as pd

def load_orders(path="data/orders.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["due_date"])

def load_plants(path="data/plants.csv") -> pd.DataFrame:
    return pd.read_csv(path)

def load_transport(path="data/transport_duties.csv") -> pd.DataFrame:
    return pd.read_csv(path)

def load_capacity_events(path="data/capacity_events.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["from_date", "to_date"])
    return df