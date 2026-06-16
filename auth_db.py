from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path(os.environ.get("FEDPRIVTAB_AUTH_DB", "fedprivtab_auth.sqlite3"))
PBKDF2_ITERATIONS = 200_000

DEFAULT_USERS = [
    ("admin", "admin123", "系统管理员"),
    ("client", "client123", "客户端用户"),
    ("researcher", "research123", "实验研究人员"),
]

CLIENT_ROLE = "客户端用户"
MANAGER_ROLES = {"系统管理员", "实验研究人员"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_db_path(path: str | Path | None = None) -> Path:
    return Path(path) if path is not None else DEFAULT_DB_PATH


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    db_path = get_db_path(path)
    if db_path.parent != Path("."):
        db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return salt.hex(), digest.hex()


def verify_password(password: str, salt_hex: str, password_hash: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    _, candidate = hash_password(password, salt)
    return hmac.compare_digest(candidate, password_hash)


def init_db(path: str | Path | None = None) -> None:
    with connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                role TEXT NOT NULL,
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                last_login_at TEXT
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                logout_at TEXT,
                is_active INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                username TEXT,
                role TEXT,
                session_id TEXT,
                success INTEGER NOT NULL,
                message TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        user_count = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if user_count == 0:
            now = utc_now()
            for username, password, role in DEFAULT_USERS:
                salt, password_hash = hash_password(password)
                connection.execute(
                    """
                    INSERT INTO users (username, role, salt, password_hash, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (username, role, salt, password_hash, now),
                )
            connection.commit()


def record_event(
    event_type: str,
    username: str | None = None,
    role: str | None = None,
    session_id: str | None = None,
    success: bool = True,
    message: str | None = None,
    path: str | Path | None = None,
) -> None:
    init_db(path)
    with connect(path) as connection:
        connection.execute(
            """
            INSERT INTO audit_events (event_type, username, role, session_id, success, message, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (event_type, username, role, session_id, int(success), message, utc_now()),
        )
        connection.commit()


def get_user(username: str, path: str | Path | None = None) -> dict[str, Any] | None:
    init_db(path)
    with connect(path) as connection:
        row = connection.execute(
            "SELECT id, username, role, salt, password_hash, is_active, created_at, last_login_at FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    return dict(row) if row else None


def list_users(role: str | None = None, path: str | Path | None = None) -> list[dict[str, Any]]:
    init_db(path)
    query = "SELECT id, username, role, is_active, created_at, last_login_at FROM users"
    params: tuple[Any, ...] = ()
    if role:
        query += " WHERE role = ?"
        params = (role,)
    query += " ORDER BY id"
    with connect(path) as connection:
        rows = connection.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def create_user(username: str, password: str, role: str = CLIENT_ROLE, path: str | Path | None = None) -> dict[str, Any]:
    username = username.strip()
    if not username:
        raise ValueError("用户名不能为空")
    if not password:
        raise ValueError("密码不能为空")
    if role not in {CLIENT_ROLE, *MANAGER_ROLES}:
        raise ValueError("不支持的角色")
    salt, password_hash = hash_password(password)
    now = utc_now()
    init_db(path)
    with connect(path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO users (username, role, salt, password_hash, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (username, role, salt, password_hash, now),
        )
        connection.commit()
    return {"id": cursor.lastrowid, "username": username, "role": role, "is_active": 1, "created_at": now, "last_login_at": None}


def set_user_active(username: str, active: bool, path: str | Path | None = None) -> bool:
    username = username.strip()
    init_db(path)
    now = utc_now()
    with connect(path) as connection:
        cursor = connection.execute("UPDATE users SET is_active = ? WHERE username = ?", (int(active), username))
        if not active:
            connection.execute(
                """
                UPDATE sessions
                SET is_active = 0, logout_at = COALESCE(logout_at, ?), last_seen_at = ?
                WHERE username = ? AND is_active = 1
                """,
                (now, now, username),
            )
        connection.commit()
    return cursor.rowcount > 0


def authenticate(username: str, password: str, path: str | Path | None = None) -> dict[str, Any] | None:
    username = username.strip()
    user = get_user(username, path)
    if not user or not user["is_active"]:
        record_event("login", username=username, success=False, message="unknown or inactive user", path=path)
        return None
    if not verify_password(password, user["salt"], user["password_hash"]):
        record_event("login", username=username, role=user["role"], success=False, message="invalid password", path=path)
        return None
    return {"id": user["id"], "username": user["username"], "role": user["role"]}


def create_session(user: dict[str, Any], path: str | Path | None = None) -> str:
    init_db(path)
    session_id = secrets.token_urlsafe(32)
    now = utc_now()
    with connect(path) as connection:
        connection.execute(
            """
            INSERT INTO sessions (session_id, user_id, username, role, created_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, user["id"], user["username"], user["role"], now, now),
        )
        connection.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (now, user["id"]))
        connection.execute(
            """
            INSERT INTO audit_events (event_type, username, role, session_id, success, message, created_at)
            VALUES (?, ?, ?, ?, 1, ?, ?)
            """,
            ("login", user["username"], user["role"], session_id, "login successful", now),
        )
        connection.commit()
    return session_id


def login(username: str, password: str, path: str | Path | None = None) -> dict[str, Any] | None:
    user = authenticate(username, password, path)
    if not user:
        return None
    session_id = create_session(user, path)
    return {**user, "session_id": session_id}


def get_session(session_id: str | None, path: str | Path | None = None, touch: bool = True) -> dict[str, Any] | None:
    if not session_id:
        return None
    init_db(path)
    with connect(path) as connection:
        row = connection.execute(
            """
            SELECT session_id, user_id, username, role, created_at, last_seen_at, logout_at, is_active
            FROM sessions
            WHERE session_id = ? AND is_active = 1 AND logout_at IS NULL
            """,
            (session_id,),
        ).fetchone()
        if row and touch:
            connection.execute("UPDATE sessions SET last_seen_at = ? WHERE session_id = ?", (utc_now(), session_id))
            connection.commit()
    return dict(row) if row else None


def logout(session_id: str | None, path: str | Path | None = None) -> bool:
    session = get_session(session_id, path, touch=False)
    if not session:
        return False
    now = utc_now()
    with connect(path) as connection:
        connection.execute(
            "UPDATE sessions SET is_active = 0, logout_at = ?, last_seen_at = ? WHERE session_id = ?",
            (now, now, session_id),
        )
        connection.execute(
            """
            INSERT INTO audit_events (event_type, username, role, session_id, success, message, created_at)
            VALUES (?, ?, ?, ?, 1, ?, ?)
            """,
            ("logout", session["username"], session["role"], session_id, "logout successful", now),
        )
        connection.commit()
    return True


def list_audit_events(path: str | Path | None = None) -> list[dict[str, Any]]:
    init_db(path)
    with connect(path) as connection:
        rows = connection.execute(
            """
            SELECT event_type, username, role, session_id, success, message, created_at
            FROM audit_events
            ORDER BY id
            """
        ).fetchall()
    return [dict(row) for row in rows]
