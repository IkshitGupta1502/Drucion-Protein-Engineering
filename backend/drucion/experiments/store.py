"""SQLite persistence for experiments, rounds and benchmarks.

Experiment state lives server-side so a dashboard reload never loses a run.
The schema is deliberately small: JSON blobs for the structured payloads,
real columns for the things worth querying.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from ..config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id            TEXT PRIMARY KEY,
    assay_id      TEXT NOT NULL,
    acquisition   TEXT NOT NULL,
    status        TEXT NOT NULL,
    config_json   TEXT NOT NULL,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    n_measured    INTEGER DEFAULT 0,
    budget        INTEGER DEFAULT 0,
    round         INTEGER DEFAULT 0,
    n_rounds      INTEGER DEFAULT 0,
    best_measured REAL,
    best_mutant   TEXT,
    error         TEXT
);
CREATE TABLE IF NOT EXISTS rounds (
    experiment_id TEXT NOT NULL,
    round         INTEGER NOT NULL,
    record_json   TEXT NOT NULL,
    created_at    REAL NOT NULL,
    PRIMARY KEY (experiment_id, round)
);
CREATE TABLE IF NOT EXISTS benchmarks (
    id          TEXT PRIMARY KEY,
    assay_id    TEXT NOT NULL,
    status      TEXT NOT NULL,
    config_json TEXT NOT NULL,
    result_json TEXT,
    progress    REAL DEFAULT 0,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    error       TEXT
);
"""

_lock = threading.Lock()


def _connect(path: Path | None = None) -> sqlite3.Connection:
    p = Path(path or DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


class Store:
    """Thin synchronous store. One instance per process is plenty."""

    def __init__(self, path: Path | None = None) -> None:
        self.conn = _connect(path)
        with _lock:
            self.conn.executescript(_SCHEMA)
            self.conn.commit()

    # -- experiments ---------------------------------------------------

    def create_experiment(self, exp_id: str, config: dict, budget: int) -> None:
        now = time.time()
        with _lock:
            self.conn.execute(
                "INSERT INTO experiments (id, assay_id, acquisition, status, "
                "config_json, created_at, updated_at, budget, n_rounds) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    exp_id,
                    config["assay_id"],
                    config["loop"]["acquisition"],
                    "created",
                    json.dumps(config),
                    now,
                    now,
                    budget,
                    config["loop"]["n_rounds"],
                ),
            )
            self.conn.commit()

    def update_experiment(self, exp_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k} = ?" for k in fields)
        with _lock:
            self.conn.execute(
                f"UPDATE experiments SET {cols} WHERE id = ?",
                (*fields.values(), exp_id),
            )
            self.conn.commit()

    def add_round(self, exp_id: str, round_no: int, record: dict) -> None:
        with _lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO rounds (experiment_id, round, "
                "record_json, created_at) VALUES (?,?,?,?)",
                (exp_id, round_no, json.dumps(record), time.time()),
            )
            self.conn.commit()

    def get_experiment(self, exp_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (exp_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["config"] = json.loads(d.pop("config_json"))
        return d

    def get_rounds(self, exp_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT record_json FROM rounds WHERE experiment_id = ? ORDER BY round",
            (exp_id,),
        ).fetchall()
        return [json.loads(r["record_json"]) for r in rows]

    def list_experiments(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, assay_id, acquisition, status, n_measured, budget, "
            "round, n_rounds, best_measured, best_mutant, created_at "
            "FROM experiments ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- benchmarks ----------------------------------------------------

    def create_benchmark(self, bench_id: str, config: dict) -> None:
        now = time.time()
        with _lock:
            self.conn.execute(
                "INSERT INTO benchmarks (id, assay_id, status, config_json, "
                "created_at, updated_at) VALUES (?,?,?,?,?,?)",
                (
                    bench_id,
                    config["assay_id"],
                    "running",
                    json.dumps(config),
                    now,
                    now,
                ),
            )
            self.conn.commit()

    def update_benchmark(self, bench_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k} = ?" for k in fields)
        with _lock:
            self.conn.execute(
                f"UPDATE benchmarks SET {cols} WHERE id = ?",
                (*fields.values(), bench_id),
            )
            self.conn.commit()

    def get_benchmark(self, bench_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM benchmarks WHERE id = ?", (bench_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["config"] = json.loads(d.pop("config_json"))
        rj = d.pop("result_json")
        d["result"] = json.loads(rj) if rj else None
        return d

    def list_benchmarks(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, assay_id, status, progress, created_at FROM benchmarks "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
