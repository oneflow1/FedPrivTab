from __future__ import annotations

import auth_db


def test_init_db_seeds_demo_users_with_hashed_passwords(tmp_path) -> None:
    db_path = tmp_path / "auth.sqlite3"
    auth_db.init_db(db_path)

    admin = auth_db.get_user("admin", db_path)
    assert admin is not None
    assert admin["role"] == "系统管理员"
    assert admin["password_hash"] != "admin123"
    assert auth_db.verify_password("admin123", admin["salt"], admin["password_hash"]) is True
    assert auth_db.verify_password("wrong", admin["salt"], admin["password_hash"]) is False


def test_login_logout_and_audit_events(tmp_path) -> None:
    db_path = tmp_path / "auth.sqlite3"
    session = auth_db.login("client", "client123", db_path)

    assert session is not None
    assert session["username"] == "client"
    assert session["role"] == "客户端用户"

    active = auth_db.get_session(session["session_id"], db_path)
    assert active is not None
    assert active["username"] == "client"

    assert auth_db.logout(session["session_id"], db_path) is True
    assert auth_db.get_session(session["session_id"], db_path) is None

    events = auth_db.list_audit_events(db_path)
    assert [event["event_type"] for event in events] == ["login", "logout"]
    assert all(event["success"] == 1 for event in events)


def test_failed_login_records_audit_event(tmp_path) -> None:
    db_path = tmp_path / "auth.sqlite3"
    assert auth_db.login("researcher", "bad-password", db_path) is None

    events = auth_db.list_audit_events(db_path)
    assert len(events) == 1
    assert events[0]["event_type"] == "login"
    assert events[0]["username"] == "researcher"
    assert events[0]["success"] == 0
