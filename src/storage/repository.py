"""SQLite persistence for prediction runs and audit events (no raw customer data)."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from configs.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_utc TEXT NOT NULL,
    model_version TEXT NOT NULL,
    n_rows INTEGER NOT NULL,
    mean_probability REAL,
    positive_rate REAL,
    schema_json TEXT
);
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_utc TEXT NOT NULL,
    event TEXT NOT NULL,
    detail TEXT
);
"""


def _conn(db_path: Path | None = None) -> sqlite3.Connection:
    p = Path(db_path or DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.executescript(SCHEMA)
    return conn


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def log_prediction_run(model_version: str, n_rows: int, mean_prob: float,
                       positive_rate: float, columns: list[str],
                       db_path: Path | None = None) -> None:
    with closing(_conn(db_path)) as c:
        c.execute(
            "INSERT INTO prediction_runs(created_utc, model_version, n_rows,"
            " mean_probability, positive_rate, schema_json) VALUES (?,?,?,?,?,?)",
            (_now(), model_version, n_rows, mean_prob, positive_rate,
             json.dumps(columns)))
        c.commit()


def log_event(event: str, detail: str = "", db_path: Path | None = None) -> None:
    with closing(_conn(db_path)) as c:
        c.execute("INSERT INTO audit_events(created_utc, event, detail) VALUES (?,?,?)",
                  (_now(), event, detail))
        c.commit()


def list_prediction_runs(limit: int = 50, db_path: Path | None = None) -> list[dict]:
    with closing(_conn(db_path)) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute("SELECT * FROM prediction_runs ORDER BY id DESC LIMIT ?",
                         (limit,)).fetchall()
        return [dict(r) for r in rows]


def list_events(limit: int = 100, db_path: Path | None = None) -> list[dict]:
    with closing(_conn(db_path)) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute("SELECT * FROM audit_events ORDER BY id DESC LIMIT ?",
                         (limit,)).fetchall()
        return [dict(r) for r in rows]
