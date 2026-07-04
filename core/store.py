"""SQLite persistence layer.

Zero-setup by design: one file, no services. Embeddings live as JSON
arrays; applicant metadata (stated fields, gold labels) as a JSON blob.
Every accessor returns plain dicts so the storage engine can be swapped
for Postgres/pgvector without touching callers.
"""

import json
import sqlite3
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

from core.config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS applicants (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    meta TEXT,
    is_demo INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS segments (
    id TEXT PRIMARY KEY,
    applicant_id TEXT,          -- NULL ⇒ policy corpus segment
    seq INTEGER,
    text TEXT NOT NULL,
    category TEXT,
    entities TEXT,
    embedding TEXT
);
CREATE TABLE IF NOT EXISTS cached_answers (
    id TEXT PRIMARY KEY,
    applicant_id TEXT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources TEXT
);
CREATE TABLE IF NOT EXISTS eval_questions (
    id TEXT PRIMARY KEY,
    applicant_id TEXT,
    question TEXT NOT NULL,
    gold_segments TEXT,
    category TEXT
);
"""


_resolved_path: Path | None = None


def _db_path() -> Path:
    """Prefer a DB next to the app; fall back to temp dir on filesystems
    that don't support SQLite locking (some network/virtual mounts)."""
    global _resolved_path
    if _resolved_path:
        return _resolved_path
    for candidate in (DB_PATH, Path(tempfile.gettempdir()) / "lendlens.db"):
        try:
            con = sqlite3.connect(candidate)
            con.execute("CREATE TABLE IF NOT EXISTS _probe (x INTEGER)")
            con.execute("DROP TABLE _probe")
            con.commit()
            con.close()
            _resolved_path = candidate
            return candidate
        except sqlite3.Error:
            continue
    _resolved_path = Path(tempfile.gettempdir()) / "lendlens.db"
    return _resolved_path


@contextmanager
def _conn():
    con = sqlite3.connect(_db_path())
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with _conn() as con:
        con.executescript(_SCHEMA)


def reset_db() -> None:
    with _conn() as con:
        for t in ("applicants", "segments", "cached_answers", "eval_questions"):
            con.execute(f"DELETE FROM {t}")


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------- applicants ----------

def save_applicant(name: str, meta: dict | None = None, is_demo=False, applicant_id=None) -> str:
    aid = applicant_id or _new_id()
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO applicants (id, name, meta, is_demo) VALUES (?,?,?,?)",
            (aid, name, json.dumps(meta or {}), int(is_demo)),
        )
    return aid


def update_applicant_meta(aid: str, patch: dict) -> None:
    ap = get_applicant(aid)
    if not ap:
        return
    meta = ap["meta"]
    meta.update(patch)
    with _conn() as con:
        con.execute("UPDATE applicants SET meta=? WHERE id=?", (json.dumps(meta), aid))


def list_applicants() -> list[dict]:
    with _conn() as con:
        rows = con.execute("SELECT * FROM applicants ORDER BY created_at").fetchall()
    return [_ap(r) for r in rows]


def get_applicant(aid: str) -> dict | None:
    with _conn() as con:
        row = con.execute("SELECT * FROM applicants WHERE id=?", (aid,)).fetchone()
    return _ap(row) if row else None


def _ap(row) -> dict:
    d = dict(row)
    d["meta"] = json.loads(d["meta"]) if d["meta"] else {}
    return d


# ---------- segments ----------

def save_segments(applicant_id: str | None, segments: list[dict]) -> None:
    with _conn() as con:
        for seg in segments:
            con.execute(
                "INSERT OR REPLACE INTO segments (id, applicant_id, seq, text, category, entities, embedding) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    seg.get("id") or _new_id(),
                    applicant_id,
                    seg.get("seq", 0),
                    seg["text"],
                    seg.get("category"),
                    json.dumps(seg.get("entities", {})),
                    json.dumps(seg["embedding"]) if seg.get("embedding") else None,
                ),
            )


def segments_for(applicant_id: str | None) -> list[dict]:
    """Segments for one applicant, or the policy corpus when None."""
    q = (
        "SELECT * FROM segments WHERE applicant_id IS NULL ORDER BY seq"
        if applicant_id is None
        else "SELECT * FROM segments WHERE applicant_id=? ORDER BY seq"
    )
    with _conn() as con:
        rows = con.execute(q, () if applicant_id is None else (applicant_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["entities"] = json.loads(d["entities"]) if d["entities"] else {}
        d["embedding"] = json.loads(d["embedding"]) if d["embedding"] else None
        out.append(d)
    return out


# ---------- cached answers ----------

def save_answers(applicant_id: str, answers: list[dict]) -> None:
    with _conn() as con:
        for a in answers:
            con.execute(
                "INSERT OR REPLACE INTO cached_answers (id, applicant_id, question, answer, sources) VALUES (?,?,?,?,?)",
                (a.get("id") or _new_id(), applicant_id, a["question"], a["answer"], json.dumps(a.get("sources", []))),
            )


def answers_for(applicant_id: str) -> list[dict]:
    with _conn() as con:
        rows = con.execute("SELECT * FROM cached_answers WHERE applicant_id=?", (applicant_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["sources"] = json.loads(d["sources"]) if d["sources"] else []
        out.append(d)
    return out


# ---------- eval questions ----------

def save_eval_questions(applicant_id: str, questions: list[dict]) -> None:
    with _conn() as con:
        for q in questions:
            con.execute(
                "INSERT OR REPLACE INTO eval_questions (id, applicant_id, question, gold_segments, category) VALUES (?,?,?,?,?)",
                (q.get("id") or _new_id(), applicant_id, q["question"], json.dumps(q.get("gold", [])), q.get("category")),
            )


def eval_questions_for(applicant_id: str | None = None) -> list[dict]:
    with _conn() as con:
        rows = (
            con.execute("SELECT * FROM eval_questions").fetchall()
            if applicant_id is None
            else con.execute("SELECT * FROM eval_questions WHERE applicant_id=?", (applicant_id,)).fetchall()
        )
    out = []
    for r in rows:
        d = dict(r)
        d["gold"] = json.loads(d["gold_segments"]) if d["gold_segments"] else []
        out.append(d)
    return out
