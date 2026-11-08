"""SQLite store for submitted FNOLs and an append-only audit trail (triggers block UPDATE and DELETE)."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS fnol (
    claim_id TEXT PRIMARY KEY,
    submitted_at TEXT NOT NULL,
    route TEXT NOT NULL,
    score REAL NOT NULL,
    payload TEXT NOT NULL,
    result TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    claim_id TEXT,
    detail TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit
BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit
BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
"""


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(path), check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.con.executescript(SCHEMA)
            if path != ":memory:":
                self.con.execute("PRAGMA journal_mode=WAL")

    def audit(self, actor: str, action: str, claim_id: str | None, detail: dict) -> None:
        with self.lock, self.con:
            self.con.execute(
                "INSERT INTO audit (ts, actor, action, claim_id, detail) VALUES (?, ?, ?, ?, ?)",
                (
                    datetime.now(UTC).isoformat(timespec="milliseconds"),
                    actor,
                    action,
                    claim_id,
                    json.dumps(detail, default=str),
                ),
            )

    def audit_rows(self, claim_id: str | None = None, limit: int = 200) -> list[dict]:
        q, args = "SELECT * FROM audit", []
        if claim_id:
            q, args = q + " WHERE claim_id = ?", [claim_id]
        rows = self.con.execute(q + " ORDER BY id DESC LIMIT ?", [*args, limit]).fetchall()
        return [{**dict(r), "detail": json.loads(r["detail"])} for r in rows]

    def next_id(self) -> str:
        n = self.con.execute("SELECT COUNT(*) FROM fnol").fetchone()[0]
        return f"FNOL-{n + 1:06d}"

    def save_fnol(self, claim_id: str, route: str, score: float, payload: dict, result: dict) -> None:
        with self.lock, self.con:
            self.con.execute(
                "INSERT INTO fnol (claim_id, submitted_at, route, score, payload, result) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    claim_id,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                    route,
                    score,
                    json.dumps(payload, default=str),
                    json.dumps(result, default=str),
                ),
            )

    def get_fnol(self, claim_id: str) -> dict | None:
        r = self.con.execute("SELECT result FROM fnol WHERE claim_id = ?", (claim_id,)).fetchone()
        return json.loads(r["result"]) if r else None

    def list_fnol(self, route: str | None = None) -> list[dict]:
        q, args = "SELECT claim_id, submitted_at, route, score, result FROM fnol", []
        if route:
            q, args = q + " WHERE route = ?", [route]
        return [
            {**dict(r), "result": json.loads(r["result"])}
            for r in self.con.execute(q + " ORDER BY claim_id DESC", args)
        ]
