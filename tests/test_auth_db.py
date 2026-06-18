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
    session = auth_db.login("admin", "admin123", db_path)

    assert session is not None
    assert session["username"] == "admin"
    assert session["role"] == "系统管理员"

    active = auth_db.get_session(session["session_id"], db_path)
    assert active is not None
    assert active["username"] == "admin"

    assert auth_db.logout(session["session_id"], db_path) is True
    assert auth_db.get_session(session["session_id"], db_path) is None

    events = auth_db.list_audit_events(db_path)
    assert [event["event_type"] for event in events] == ["login", "logout"]
    assert all(event["success"] == 1 for event in events)


def test_failed_login_records_audit_event(tmp_path) -> None:
    db_path = tmp_path / "auth.sqlite3"
    assert auth_db.login("client-1", "bad-password", db_path) is None

    events = auth_db.list_audit_events(db_path)
    assert len(events) == 1
    assert events[0]["event_type"] == "login"
    assert events[0]["username"] == "client-1"
    assert events[0]["success"] == 0


def test_change_password_and_delete_user(tmp_path) -> None:
    db_path = tmp_path / "auth.sqlite3"
    created = auth_db.create_user("client-x", "first-pass", auth_db.CLIENT_ROLE, db_path)
    assert created["username"] == "client-x"

    assert auth_db.change_password("client-x", "wrong", "second-pass", db_path) is False
    assert auth_db.change_password("client-x", "first-pass", "second-pass", db_path) is True
    assert auth_db.login("client-x", "first-pass", db_path) is None
    assert auth_db.login("client-x", "second-pass", db_path) is not None

    assert auth_db.delete_user("client-x", db_path) is True
    assert auth_db.login("client-x", "second-pass", db_path) is None


def test_preprocess_version_persistence_and_scoping(tmp_path) -> None:
    db_path = tmp_path / "auth.sqlite3"
    admin = auth_db.login("admin", "admin123", db_path)

    centralized = auth_db.save_preprocess_version(
        admin,
        {
            "id": "admin-v1",
            "scope": "centralized",
            "target_column": "target",
            "row_count": 1,
            "missing_strategies": {"feature": "median"},
            "scaler_strategies": {"feature": "standard"},
            "message": "ok",
            "records": [{"feature": 1, "target": 0}],
        },
        db_path,
    )

    assert centralized["scope"] == "centralized"
    assert centralized["missing_strategies"] == {"feature": "median"}
    admin_versions = auth_db.list_preprocess_versions(admin, db_path)
    assert [version["id"] for version in admin_versions] == ["admin-v1"]
    assert admin_versions[0]["records"] == [{"feature": 1, "target": 0}]
