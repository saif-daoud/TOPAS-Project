from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import settings


_write_lock = threading.RLock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    db = sqlite3.connect(settings.database_path, timeout=30, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.execute("PRAGMA journal_mode = WAL")
    try:
        yield db
        db.commit()
    finally:
        db.close()


def init_db() -> None:
    with _write_lock, connection() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                domain TEXT NOT NULL,
                system_name TEXT NOT NULL,
                user_name TEXT NOT NULL,
                interaction_unit TEXT NOT NULL,
                model TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'draft',
                progress INTEGER NOT NULL DEFAULT 0,
                current_step TEXT NOT NULL DEFAULT 'Ready for textbooks',
                run_id TEXT,
                output_path TEXT,
                error TEXT,
                is_demo INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS books (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                original_name TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                page_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                kind TEXT NOT NULL,
                message TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_projects_user_updated
                ON projects(user_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_books_project_role
                ON books(project_id, role);
            CREATE INDEX IF NOT EXISTS idx_events_project_id
                ON events(project_id, id);
            """
        )
        existing_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(projects)").fetchall()
        }
        migrations = {
            "refinement_status": "TEXT NOT NULL DEFAULT 'not_started'",
            "refinement_progress": "INTEGER NOT NULL DEFAULT 0",
            "refinement_step": "TEXT NOT NULL DEFAULT 'Ready after extraction'",
            "refined_path": "TEXT",
            "refinement_error": "TEXT",
            "refined_at": "TEXT",
        }
        for column, declaration in migrations.items():
            if column not in existing_columns:
                db.execute(f"ALTER TABLE projects ADD COLUMN {column} {declaration}")


def row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def upsert_user(email: str) -> dict[str, Any]:
    normalized = email.strip().lower()
    now = utc_now()
    with _write_lock, connection() as db:
        row = db.execute("SELECT * FROM users WHERE email = ?", (normalized,)).fetchone()
        if row:
            db.execute("UPDATE users SET last_seen_at = ? WHERE id = ?", (now, row["id"]))
            return {**dict(row), "last_seen_at": now}
        user = {"id": uuid.uuid4().hex, "email": normalized, "created_at": now, "last_seen_at": now}
        db.execute(
            "INSERT INTO users (id, email, created_at, last_seen_at) VALUES (?, ?, ?, ?)",
            (user["id"], user["email"], now, now),
        )
        return user


def get_user(user_id: str) -> dict[str, Any] | None:
    with connection() as db:
        return row_dict(db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone())


def create_project(user_id: str, values: dict[str, Any], *, is_demo: bool = False) -> dict[str, Any]:
    now = utc_now()
    project_id = uuid.uuid4().hex
    with _write_lock, connection() as db:
        db.execute(
            """
            INSERT INTO projects (
                id, user_id, name, domain, system_name, user_name, interaction_unit,
                model, status, progress, current_step, is_demo, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                user_id,
                values["name"],
                values["domain"],
                values["system_name"],
                values["user_name"],
                values["interaction_unit"],
                values["model"],
                values.get("status", "draft"),
                values.get("progress", 0),
                values.get("current_step", "Ready for textbooks"),
                int(is_demo),
                now,
                now,
            ),
        )
    return get_project(project_id, user_id) or {}


def get_project(project_id: str, user_id: str | None = None) -> dict[str, Any] | None:
    sql = "SELECT * FROM projects WHERE id = ?"
    args: tuple[Any, ...] = (project_id,)
    if user_id is not None:
        sql += " AND user_id = ?"
        args += (user_id,)
    with connection() as db:
        project = row_dict(db.execute(sql, args).fetchone())
        if not project:
            return None
        books = [dict(row) for row in db.execute(
            "SELECT * FROM books WHERE project_id = ? ORDER BY role, created_at", (project_id,)
        ).fetchall()]
        project["books"] = books
        project["is_demo"] = bool(project["is_demo"])
        return project


def list_projects(user_id: str) -> list[dict[str, Any]]:
    with connection() as db:
        rows = db.execute(
            """
            SELECT p.*,
                COUNT(b.id) AS book_count,
                COALESCE(SUM(CASE WHEN b.role = 'system' THEN 1 ELSE 0 END), 0) AS system_book_count,
                COALESCE(SUM(CASE WHEN b.role = 'user' THEN 1 ELSE 0 END), 0) AS user_book_count
            FROM projects p
            LEFT JOIN books b ON b.project_id = p.id
            WHERE p.user_id = ?
            GROUP BY p.id
            ORDER BY p.is_demo ASC, p.updated_at DESC
            """,
            (user_id,),
        ).fetchall()
        return [{**dict(row), "is_demo": bool(row["is_demo"])} for row in rows]


def update_project(project_id: str, **values: Any) -> None:
    if not values:
        return
    values["updated_at"] = utc_now()
    allowed = {
        "status", "progress", "current_step", "run_id", "output_path", "error",
        "completed_at", "updated_at", "model", "refinement_status",
        "refinement_progress", "refinement_step", "refined_path",
        "refinement_error", "refined_at",
    }
    clean = {key: value for key, value in values.items() if key in allowed}
    setters = ", ".join(f"{key} = ?" for key in clean)
    with _write_lock, connection() as db:
        db.execute(
            f"UPDATE projects SET {setters} WHERE id = ?",
            (*clean.values(), project_id),
        )


def add_book(project_id: str, role: str, name: str, path: Path, size: int, pages: int) -> dict[str, Any]:
    now = utc_now()
    book = {
        "id": uuid.uuid4().hex,
        "project_id": project_id,
        "role": role,
        "original_name": name,
        "stored_path": str(path),
        "size_bytes": size,
        "page_count": pages,
        "created_at": now,
    }
    with _write_lock, connection() as db:
        db.execute(
            """
            INSERT INTO books (id, project_id, role, original_name, stored_path, size_bytes, page_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(book.values()),
        )
    return book


def delete_book(book_id: str, project_id: str) -> dict[str, Any] | None:
    with _write_lock, connection() as db:
        row = db.execute(
            "SELECT * FROM books WHERE id = ? AND project_id = ?", (book_id, project_id)
        ).fetchone()
        if not row:
            return None
        db.execute("DELETE FROM books WHERE id = ?", (book_id,))
        return dict(row)


def add_event(project_id: str, kind: str, message: str, payload: dict[str, Any] | None = None) -> int:
    now = utc_now()
    with _write_lock, connection() as db:
        cursor = db.execute(
            "INSERT INTO events (project_id, kind, message, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (project_id, kind, message, json.dumps(payload or {}, ensure_ascii=False), now),
        )
        return int(cursor.lastrowid)


def get_events(project_id: str, after: int = 0, limit: int = 250) -> list[dict[str, Any]]:
    with connection() as db:
        rows = db.execute(
            "SELECT * FROM events WHERE project_id = ? AND id > ? ORDER BY id LIMIT ?",
            (project_id, after, limit),
        ).fetchall()
        events = []
        for row in rows:
            event = dict(row)
            try:
                event["payload"] = json.loads(event.pop("payload_json"))
            except json.JSONDecodeError:
                event["payload"] = {}
            events.append(event)
        return events


def ensure_demo_project(user_id: str, demo_root: Path) -> None:
    if not demo_root.exists():
        return
    with connection() as db:
        exists = db.execute(
            "SELECT id FROM projects WHERE user_id = ? AND is_demo = 1", (user_id,)
        ).fetchone()
    if exists:
        project_id = str(exists["id"])
    else:
        project = create_project(
            user_id,
            {
                "name": "CBT agent blueprint",
                "domain": "Cognitive behavioral therapy",
                "system_name": "Therapist",
                "user_name": "Patient",
                "interaction_unit": "session",
                "model": "gpt-4.1",
                "status": "completed",
                "progress": 100,
                "current_step": "Extraction complete",
            },
            is_demo=True,
        )
        project_id = project["id"]
        now = utc_now()
        with _write_lock, connection() as db:
            for role, count in (("system", 7), ("user", 6)):
                for index in range(count):
                    db.execute(
                        """
                        INSERT INTO books (id, project_id, role, original_name, stored_path, size_bytes, page_count, created_at)
                        VALUES (?, ?, ?, ?, ?, 0, 0, ?)
                        """,
                        (
                            uuid.uuid4().hex,
                            project_id,
                            role,
                            f"{role.title()} textbook {index + 1:02d}.pdf",
                            "",
                            now,
                        ),
                    )
        add_event(project_id, "completed", "Late-fusion extraction completed", {"demo": True})

    showcase_root = settings.data_dir / user_id / project_id / "showcase"
    extracted_target = showcase_root / "extracted_components"
    if not extracted_target.exists():
        extracted_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(demo_root, extracted_target)

    refined_source = demo_root.parent / "refined_components"
    refined_target = showcase_root / "refined_components"
    refinement_values: dict[str, Any] = {}
    if refined_source.exists():
        if not refined_target.exists():
            shutil.copytree(refined_source, refined_target)
        refinement_values = {
            "refined_path": str(refined_target),
            "refinement_status": "completed",
            "refinement_progress": 100,
            "refinement_step": "Refinement complete",
            "refined_at": utc_now(),
        }
    update_project(
        project_id,
        output_path=str(extracted_target),
        completed_at=utc_now(),
        **refinement_values,
    )
