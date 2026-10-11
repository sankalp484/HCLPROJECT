"""SQLite store for the loaded tables, run traces and the data version.

One file, WAL mode so readers do not block the single writer. Every write bumps
data_version, which keys the in-process caches, so a new upload is picked up on the
next request. All SQL is parameterised; table names come from a fixed allow-list.
"""

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from src.loader import DATA_DIR, FILES, DataPack, empty_table, load_all, normalise, planning_start

DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "app.db"))
TABLES = tuple(FILES)  # orders, plants, transport_duties, capacity_events
_write_lock = threading.Lock()


@contextmanager
def connect(path: Path | None = None):
    path = Path(path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        yield con
        con.commit()
    finally:
        con.close()


def init_db(seed_dir: Path | None = DATA_DIR, path: Path | None = None) -> None:
    """Create the bookkeeping tables; seed data tables from CSVs if the store is empty."""
    with connect(path) as con:
        con.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        con.execute("""CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            request TEXT, answer_type TEXT, latency_ms REAL,
            response_json TEXT, trace_json TEXT)""")
        con.execute("""CREATE TABLE IF NOT EXISTS ingest_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            table_name TEXT, mode TEXT, rows_loaded INTEGER, rows_rejected INTEGER,
            status TEXT)""")
        seeded = con.execute("SELECT value FROM meta WHERE key='data_version'").fetchone()
    if not seeded and seed_dir and Path(seed_dir, "orders.csv").exists():
        replace_all(load_all(seed_dir), path)


def _write(con: sqlite3.Connection, table: str, df: pd.DataFrame, mode: str) -> None:
    if table not in TABLES:
        raise ValueError(f"unknown table {table}")
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    out.to_sql(table, con, if_exists="append" if mode == "append" else "replace", index=False)


def _bump_version(con: sqlite3.Connection, notes: list[str] | None = None) -> int:
    row = con.execute("SELECT value FROM meta WHERE key='data_version'").fetchone()
    version = int(row[0]) + 1 if row else 1
    con.execute("INSERT OR REPLACE INTO meta VALUES ('data_version', ?)", (str(version),))
    if notes is not None:
        con.execute("INSERT OR REPLACE INTO meta VALUES ('notes', ?)", (json.dumps(notes),))
    return version


def replace_all(pack: DataPack, path: Path | None = None) -> int:
    """Replace all four tables at once (CLI loader)."""
    with _write_lock, connect(path) as con:
        for table, df in (("orders", pack.orders), ("plants", pack.plants),
                          ("transport_duties", pack.transport),
                          ("capacity_events", pack.events)):
            _write(con, table, df, "replace")
        return _bump_version(con, pack.notes)


def save_table(table: str, df: pd.DataFrame, mode: str = "replace",
               path: Path | None = None) -> int:
    """Store one validated table (POST /ingest). Returns the new data version."""
    with _write_lock, connect(path) as con:
        if mode == "append" and not _table_exists(con, table):
            mode = "replace"
        _write(con, table, df, mode)
        return _bump_version(con)


def log_ingest(table: str, mode: str, loaded: int, rejected: int, status: str,
               path: Path | None = None) -> None:
    with connect(path) as con:
        con.execute("INSERT INTO ingest_log (table_name, mode, rows_loaded, rows_rejected, "
                    "status) VALUES (?, ?, ?, ?, ?)", (table, mode, loaded, rejected, status))


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                       (table,)).fetchone() is not None


def data_version(path: Path | None = None) -> int:
    with connect(path) as con:
        row = con.execute("SELECT value FROM meta WHERE key='data_version'").fetchone()
    return int(row[0]) if row else 0


def load_pack(path: Path | None = None) -> DataPack:
    """Read the stored tables back into a normalised DataPack."""
    with connect(path) as con:
        frames = {}
        for table in TABLES:
            if _table_exists(con, table):
                frames[table] = normalise(table, pd.read_sql(f'SELECT * FROM "{table}"', con))
            else:
                frames[table] = empty_table(table)
        row = con.execute("SELECT value FROM meta WHERE key='notes'").fetchone()
    notes = json.loads(row[0]) if row else []
    return DataPack(orders=frames["orders"], plants=frames["plants"],
                    transport=frames["transport_duties"], events=frames["capacity_events"],
                    planning_start=planning_start(frames["orders"], frames["plants"]),
                    notes=notes)


def save_run(run_id: str, request: str, answer_type: str, latency_ms: float,
             response: dict, trace: dict, path: Path | None = None) -> None:
    with connect(path) as con:
        con.execute("INSERT OR REPLACE INTO runs (run_id, request, answer_type, latency_ms, "
                    "response_json, trace_json) VALUES (?, ?, ?, ?, ?, ?)",
                    (run_id, request, answer_type, latency_ms,
                     json.dumps(response, default=str), json.dumps(trace, default=str)))


def get_run(run_id: str, path: Path | None = None) -> dict | None:
    with connect(path) as con:
        row = con.execute("SELECT trace_json FROM runs WHERE run_id=?", (run_id,)).fetchone()
    return json.loads(row[0]) if row else None


def ping(path: Path | None = None) -> bool:
    try:
        with connect(path) as con:
            con.execute("SELECT 1")
        return True
    except sqlite3.Error:
        return False
