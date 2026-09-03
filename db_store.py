"""SQLite storage for persisting threads and responses across restarts.

Connections are opened per operation and always closed; runtime sqlite
failures degrade to cache-miss / no-op so a storage problem never fails a
request whose upstream answer already succeeded.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Optional, ParamSpec, TypeVar, overload

from pplx_transport import ThreadState

log = logging.getLogger("pplx.store")

DB_PATH = os.environ.get("PPLX_DB_PATH", "state.db")


P = ParamSpec("P")
T = TypeVar("T")
D = TypeVar("D")


@overload
def _safe(
    default: None = None,
) -> Callable[[Callable[P, T]], Callable[P, T | None]]: ...
@overload
def _safe(default: D) -> Callable[[Callable[P, T]], Callable[P, T | D]]: ...
def _safe(default: Any = None) -> Any:
    """Swallow sqlite3.Error: persistence problems must never break requests."""

    def deco(fn: Callable[P, T]) -> Callable[P, T | Any]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            try:
                return fn(*args, **kwargs)
            except sqlite3.Error as e:
                log.error(
                    "store.%s failed: %s", getattr(fn, "__name__", type(fn).__name__), e
                )
                return default

        return wrapper

    return deco


class Store:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        # journal_mode=WAL persists in the database file; set once here.
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS threads (
                    key TEXT PRIMARY KEY,
                    backend_uuid TEXT,
                    read_write_token TEXT,
                    slug TEXT,
                    title TEXT,
                    model TEXT,
                    mode TEXT,
                    account_id INTEGER,
                    last_used REAL
                );

                CREATE TABLE IF NOT EXISTS threads_by_answer (
                    answer_hash TEXT PRIMARY KEY,
                    backend_uuid TEXT,
                    read_write_token TEXT,
                    slug TEXT,
                    title TEXT,
                    model TEXT,
                    mode TEXT,
                    account_id INTEGER,
                    last_used REAL
                );

                CREATE TABLE IF NOT EXISTS responses (
                    id TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    backend_uuid TEXT,
                    read_write_token TEXT,
                    slug TEXT,
                    title TEXT,
                    model TEXT,
                    mode TEXT,
                    account_id INTEGER,
                    last_used REAL,
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_threads_last_used ON threads(last_used);
                CREATE INDEX IF NOT EXISTS idx_threads_by_answer_last_used ON threads_by_answer(last_used);
                CREATE INDEX IF NOT EXISTS idx_responses_created_at ON responses(created_at);
            """)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _row_to_thread(row: Optional[sqlite3.Row]) -> Optional[ThreadState]:
        if row is None:
            return None
        return ThreadState(
            backend_uuid=row["backend_uuid"],
            read_write_token=row["read_write_token"],
            slug=row["slug"],
            title=row["title"],
            model=row["model"],
            mode=row["mode"] or "copilot",
            account_id=row["account_id"],
            last_used=row["last_used"] or time.time(),
        )

    # -- threads (conversation-key -> handle) --------------------------------

    @_safe()
    def get_thread(self, key: str) -> Optional[ThreadState]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM threads WHERE key = ?", (key,)).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE threads SET last_used = ? WHERE key = ?", (time.time(), key)
            )
            return self._row_to_thread(row)

    @_safe()
    def save_thread(self, key: str, thread: ThreadState, cap: int = 1024) -> None:
        now = time.time()
        thread.last_used = now
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO threads (key, backend_uuid, read_write_token, slug, title, model, mode, account_id, last_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    backend_uuid = excluded.backend_uuid,
                    read_write_token = excluded.read_write_token,
                    slug = excluded.slug,
                    title = excluded.title,
                    model = excluded.model,
                    mode = excluded.mode,
                    account_id = excluded.account_id,
                    last_used = excluded.last_used
            """,
                (
                    key,
                    thread.backend_uuid,
                    thread.read_write_token,
                    thread.slug,
                    thread.title,
                    thread.model,
                    thread.mode,
                    thread.account_id,
                    now,
                ),
            )
            self._evict(conn, "threads", "key", "last_used", cap)

    # -- threads_by_answer (echoed-answer hash -> handle) --------------------

    @_safe()
    def get_thread_by_answer(self, answer_hash: str) -> Optional[ThreadState]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM threads_by_answer WHERE answer_hash = ?", (answer_hash,)
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE threads_by_answer SET last_used = ? WHERE answer_hash = ?",
                (time.time(), answer_hash),
            )
            return self._row_to_thread(row)

    @_safe()
    def save_thread_by_answer(
        self, answer_hash: str, thread: ThreadState, cap: int = 1024
    ) -> None:
        now = time.time()
        thread.last_used = now
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO threads_by_answer (answer_hash, backend_uuid, read_write_token, slug, title, model, mode, account_id, last_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(answer_hash) DO UPDATE SET
                    backend_uuid = excluded.backend_uuid,
                    read_write_token = excluded.read_write_token,
                    slug = excluded.slug,
                    title = excluded.title,
                    model = excluded.model,
                    mode = excluded.mode,
                    account_id = excluded.account_id,
                    last_used = excluded.last_used
            """,
                (
                    answer_hash,
                    thread.backend_uuid,
                    thread.read_write_token,
                    thread.slug,
                    thread.title,
                    thread.model,
                    thread.mode,
                    thread.account_id,
                    now,
                ),
            )
            self._evict(conn, "threads_by_answer", "answer_hash", "last_used", cap)

    # -- responses -----------------------------------------------------------

    @_safe()
    def get_response(self, resp_id: str) -> Optional[dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM responses WHERE id = ?", (resp_id,)
            ).fetchone()
            if row is None:
                return None
            loaded: Any = json.loads(row["data_json"])
            if not isinstance(loaded, dict):
                return None
            data: dict[str, Any] = loaded
            thread = self._row_to_thread(row)
            if thread is not None:
                data["_thread"] = thread
            return data

    @_safe()
    def save_response(self, resp_id: str, data: dict[str, Any], cap: int = 512) -> None:
        candidate = data.get("_thread")
        thread: Optional[ThreadState] = (
            candidate if isinstance(candidate, ThreadState) else None
        )
        created_raw = data.get("created_at")
        created_at: float = (
            created_raw if isinstance(created_raw, (int, float)) else time.time()
        )
        clean = {k: v for k, v in data.items() if k != "_thread"}
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO responses (id, data_json, backend_uuid, read_write_token, slug, title, model, mode, account_id, last_used, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    data_json = excluded.data_json,
                    backend_uuid = excluded.backend_uuid,
                    read_write_token = excluded.read_write_token,
                    slug = excluded.slug,
                    title = excluded.title,
                    model = excluded.model,
                    mode = excluded.mode,
                    account_id = excluded.account_id,
                    last_used = excluded.last_used,
                    created_at = excluded.created_at
            """,
                (
                    resp_id,
                    json.dumps(clean),
                    thread.backend_uuid if thread else None,
                    thread.read_write_token if thread else None,
                    thread.slug if thread else None,
                    thread.title if thread else None,
                    thread.model if thread else None,
                    thread.mode if thread else "copilot",
                    thread.account_id if thread else None,
                    thread.last_used if thread else time.time(),
                    created_at,
                ),
            )
            self._evict(conn, "responses", "id", "created_at", cap)

    # -- misc ------------------------------------------------------------------

    @staticmethod
    def _evict(
        conn: sqlite3.Connection, table: str, pk: str, order_col: str, cap: int
    ) -> None:
        """Keep the newest `cap` rows (LRU by order_col), drop the rest."""
        if cap <= 0:
            return
        conn.execute(
            f"""
            DELETE FROM {table} WHERE {pk} IN (
                SELECT {pk} FROM {table} ORDER BY {order_col} DESC LIMIT -1 OFFSET ?
            )
        """,
            (cap,),
        )

    @_safe(default=0)
    def count_threads(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM threads").fetchone()
            if row is None:
                return 0
            value = row["c"]
            return value if isinstance(value, int) else 0

    @_safe(default=0)
    def count_responses(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM responses").fetchone()
            if row is None:
                return 0
            value = row["c"]
            return value if isinstance(value, int) else 0
