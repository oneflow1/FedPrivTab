from __future__ import annotations

import hashlib
import hmac
import json
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
]

CLIENT_ROLE = "客户端用户"
MANAGER_ROLES = {"系统管理员"}


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
                demo_password TEXT,
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

            CREATE TABLE IF NOT EXISTS preprocess_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_id TEXT NOT NULL,
                user_id INTEGER NOT NULL REFERENCES users(id),
                username TEXT NOT NULL,
                scope TEXT NOT NULL,
                name TEXT,
                original_filename TEXT,
                processed_at TEXT NOT NULL,
                target_column TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                missing_strategies_json TEXT NOT NULL DEFAULT '{}',
                scaler_strategies_json TEXT NOT NULL DEFAULT '{}',
                message TEXT,
                records_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                statistics_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                UNIQUE(username, version_id)
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_preprocess_versions_username_version_id
            ON preprocess_versions(username, version_id);
            """
        )
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(users)").fetchall()}
        if "demo_password" not in columns:
            connection.execute("ALTER TABLE users ADD COLUMN demo_password TEXT")
        preprocess_columns = {row["name"] for row in connection.execute("PRAGMA table_info(preprocess_versions)").fetchall()}
        preprocess_migrations = {
            "version_id": "ALTER TABLE preprocess_versions ADD COLUMN version_id TEXT",
            "user_id": "ALTER TABLE preprocess_versions ADD COLUMN user_id INTEGER REFERENCES users(id)",
            "username": "ALTER TABLE preprocess_versions ADD COLUMN username TEXT",
            "scope": "ALTER TABLE preprocess_versions ADD COLUMN scope TEXT",
            "name": "ALTER TABLE preprocess_versions ADD COLUMN name TEXT",
            "original_filename": "ALTER TABLE preprocess_versions ADD COLUMN original_filename TEXT",
            "processed_at": "ALTER TABLE preprocess_versions ADD COLUMN processed_at TEXT",
            "target_column": "ALTER TABLE preprocess_versions ADD COLUMN target_column TEXT",
            "row_count": "ALTER TABLE preprocess_versions ADD COLUMN row_count INTEGER",
            "missing_strategies_json": "ALTER TABLE preprocess_versions ADD COLUMN missing_strategies_json TEXT NOT NULL DEFAULT '{}'",
            "scaler_strategies_json": "ALTER TABLE preprocess_versions ADD COLUMN scaler_strategies_json TEXT NOT NULL DEFAULT '{}'",
            "message": "ALTER TABLE preprocess_versions ADD COLUMN message TEXT",
            "records_json": "ALTER TABLE preprocess_versions ADD COLUMN records_json TEXT",
            "metadata_json": "ALTER TABLE preprocess_versions ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'",
            "statistics_json": "ALTER TABLE preprocess_versions ADD COLUMN statistics_json TEXT NOT NULL DEFAULT '{}'",
            "created_at": "ALTER TABLE preprocess_versions ADD COLUMN created_at TEXT",
        }
        for column, statement in preprocess_migrations.items():
            if column not in preprocess_columns:
                connection.execute(statement)
        user_count = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        now = utc_now()
        if user_count == 0:
            for username, password, role in DEFAULT_USERS:
                salt, password_hash = hash_password(password)
                connection.execute(
                    """
                    INSERT INTO users (username, role, salt, password_hash, demo_password, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (username, role, salt, password_hash, password, now),
                )
            connection.commit()
        else:
            for username, password, role in DEFAULT_USERS:
                existing = connection.execute("SELECT demo_password FROM users WHERE username = ?", (username,)).fetchone()
                if existing:
                    if existing["demo_password"] is None:
                        connection.execute("UPDATE users SET demo_password = ? WHERE username = ?", (password, username))
                    continue
                salt, password_hash = hash_password(password)
                connection.execute(
                    """
                    INSERT INTO users (username, role, salt, password_hash, demo_password, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (username, role, salt, password_hash, password, now),
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
    query = "SELECT id, username, role, is_active, created_at, last_login_at, demo_password FROM users"
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
            INSERT INTO users (username, role, salt, password_hash, demo_password, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (username, role, salt, password_hash, password, now),
        )
        connection.commit()
    return {"id": cursor.lastrowid, "username": username, "role": role, "is_active": 1, "created_at": now, "last_login_at": None, "demo_password": password}


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


def delete_user(username: str, path: str | Path | None = None) -> bool:
    username = username.strip()
    init_db(path)
    with connect(path) as connection:
        user = connection.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if not user:
            return False
        connection.execute("DELETE FROM sessions WHERE username = ?", (username,))
        cursor = connection.execute("DELETE FROM users WHERE username = ?", (username,))
        connection.commit()
    return cursor.rowcount > 0


def change_password(username: str, old_password: str | None, new_password: str, path: str | Path | None = None) -> bool:
    username = username.strip()
    if not new_password:
        raise ValueError("新密码不能为空")
    user = get_user(username, path)
    if not user:
        return False
    if old_password is not None and not verify_password(old_password, user["salt"], user["password_hash"]):
        return False
    salt, password_hash = hash_password(new_password)
    with connect(path) as connection:
        connection.execute(
            "UPDATE users SET salt = ?, password_hash = ?, demo_password = ? WHERE username = ?",
            (salt, password_hash, new_password, username),
        )
        connection.commit()
    return True


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


def _json_dumps(value: Any, fallback: Any) -> str:
    return json.dumps(value if value is not None else fallback, ensure_ascii=False, allow_nan=False)


def _json_loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _preprocess_version_from_row(row: sqlite3.Row) -> dict[str, Any]:
    records = _json_loads(row["records_json"], [])
    metadata = _json_loads(row["metadata_json"] if "metadata_json" in row.keys() else None, {})
    statistics = _json_loads(row["statistics_json"] if "statistics_json" in row.keys() else None, {})
    return {
        "db_id": row["id"],
        "id": row["version_id"],
        "version_id": row["version_id"],
        "user_id": row["user_id"],
        "username": row["username"],
        "scope": row["scope"],
        "name": row["name"],
        "original_filename": row["original_filename"],
        "processed_at": row["processed_at"],
        "processedAt": row["processed_at"],
        "target_column": row["target_column"],
        "target": row["target_column"],
        "row_count": row["row_count"],
        "rows": row["row_count"],
        "missing_strategies": _json_loads(row["missing_strategies_json"], {}),
        "scaler_strategies": _json_loads(row["scaler_strategies_json"], {}),
        "message": row["message"] or "",
        "records": records if isinstance(records, list) else [],
        "metadata": metadata if isinstance(metadata, dict) else {},
        "statistics": statistics if isinstance(statistics, dict) else {},
        "created_at": row["created_at"],
        "createdAt": row["created_at"],
    }


def save_preprocess_version(user: dict[str, Any], payload: dict[str, Any], path: str | Path | None = None) -> dict[str, Any]:
    init_db(path)
    username = str(user["username"])
    user_id = int(user["user_id"] if "user_id" in user else user["id"])
    role = str(user["role"])
    allowed_scope = "federated" if role == CLIENT_ROLE else "centralized"
    requested_scope = str(payload.get("scope") or allowed_scope)
    scope = requested_scope if requested_scope == allowed_scope else allowed_scope
    records = payload.get("records")
    if scope == "centralized" and (not isinstance(records, list) or not records):
        raise ValueError("缺少处理后的 records")
    if scope == "federated":
        records = []
    elif not isinstance(records, list):
        records = []
    version_id = str(payload.get("version_id") or payload.get("id") or f"v{secrets.token_urlsafe(8)}").strip()
    if not version_id:
        raise ValueError("版本 ID 不能为空")
    row_count = int(payload.get("row_count") or payload.get("rows") or len(records))
    if row_count <= 0:
        raise ValueError("行数必须大于 0")
    target_column = str(payload.get("target_column") or payload.get("target") or "target").strip() or "target"
    processed_at = str(payload.get("processed_at") or payload.get("processedAt") or utc_now())
    created_at = utc_now()
    values = (
        version_id,
        user_id,
        username,
        scope,
        str(payload.get("name") or ""),
        str(payload.get("original_filename") or payload.get("originalFilename") or ""),
        processed_at,
        target_column,
        row_count,
        _json_dumps(payload.get("missing_strategies") or payload.get("missingStrategies"), {}),
        _json_dumps(payload.get("scaler_strategies") or payload.get("scalerStrategies"), {}),
        str(payload.get("message") or ""),
        _json_dumps(records, []),
        _json_dumps(payload.get("metadata"), {}),
        _json_dumps(payload.get("statistics"), {}),
        created_at,
    )
    with connect(path) as connection:
        connection.execute(
            """
            INSERT INTO preprocess_versions (
                version_id, user_id, username, scope, name, original_filename, processed_at,
                target_column, row_count, missing_strategies_json, scaler_strategies_json,
                message, records_json, metadata_json, statistics_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(username, version_id) DO UPDATE SET
                user_id = excluded.user_id,
                scope = excluded.scope,
                name = excluded.name,
                original_filename = excluded.original_filename,
                processed_at = excluded.processed_at,
                target_column = excluded.target_column,
                row_count = excluded.row_count,
                missing_strategies_json = excluded.missing_strategies_json,
                scaler_strategies_json = excluded.scaler_strategies_json,
                message = excluded.message,
                records_json = excluded.records_json,
                metadata_json = excluded.metadata_json,
                statistics_json = excluded.statistics_json
            """,
            values,
        )
        row = connection.execute(
            """
            SELECT *
            FROM preprocess_versions
            WHERE username = ? AND version_id = ?
            """,
            (username, version_id),
        ).fetchone()
        connection.commit()
    return _preprocess_version_from_row(row)


def list_preprocess_versions(user: dict[str, Any], path: str | Path | None = None) -> list[dict[str, Any]]:
    init_db(path)
    username = str(user["username"])
    role = str(user["role"])
    if role == CLIENT_ROLE:
        query = "SELECT * FROM preprocess_versions WHERE username = ? ORDER BY processed_at DESC, id DESC"
        params: tuple[Any, ...] = (username,)
    else:
        query = "SELECT * FROM preprocess_versions ORDER BY processed_at DESC, id DESC"
        params = ()
    with connect(path) as connection:
        rows = connection.execute(query, params).fetchall()
    return [_preprocess_version_from_row(row) for row in rows]
