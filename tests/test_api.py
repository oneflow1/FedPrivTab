import json

from app import app


def test_health_validate_train_and_report_endpoints() -> None:
    client = app.test_client()
    health = client.get("/health")
    assert health.status_code == 200
    assert health.get_json()["status"] == "ok"

    sample = client.get("/sample-data?samples=50&features=4&clients=2&seed=3")
    assert sample.status_code == 200
    records = sample.get_json()["records"]

    validation = client.post(
        "/validate",
        data=json.dumps({"records": records, "target_column": "target"}),
        content_type="application/json",
    )
    assert validation.status_code == 200
    assert validation.get_json()["valid"] is True

    dirty_records = [dict(record) for record in records]
    dirty_records[0]["feature_0"] = None
    preprocessed = client.post(
        "/validate",
        data=json.dumps(
            {
                "records": dirty_records,
                "target_column": "target",
                "apply_preprocess": True,
                "missing_strategy": "mean",
                "scaler": "none",
            }
        ),
        content_type="application/json",
    )
    assert preprocessed.status_code == 200
    assert preprocessed.get_json()["valid"] is True
    assert "records" in preprocessed.get_json()

    for mode in ["centralized", "fedavg", "dp_fedavg"]:
        payload = {
            "records": records,
            "mode": mode,
            "epochs": 1,
            "rounds": 1,
            "clients": 2,
            "seed": 3,
            "non_iid": True,
            "epsilon": 3.0,
            "delta": 1e-5,
        }
        response = client.post("/train", data=json.dumps(payload), content_type="application/json")
        assert response.status_code == 200
        body = response.get_json()
        assert body["mode"] == mode
        assert "metrics" in body
        assert "history" in body
        assert "accuracy" in body["history"]
        assert "client_distribution" in body
        if mode == "dp_fedavg":
            assert body["dp"] is not None
            assert body["dp"]["epsilon"] == 3.0

        report = client.post("/report", data=json.dumps({"result": body}), content_type="application/json")
        assert report.status_code == 200
        markdown = report.get_json()["markdown"]
        assert "FedPrivTab 实验报告" in markdown
        assert "Accuracy 曲线摘要" in markdown
        if mode == "dp_fedavg":
            assert "DP-FedAvg" in markdown
            assert "noise_multiplier" in markdown


def test_auth_endpoints() -> None:
    client = app.test_client()

    denied = client.post("/auth/login", data=json.dumps({"username": "admin", "password": "wrong"}), content_type="application/json")
    assert denied.status_code == 401

    login = client.post(
        "/auth/login",
        data=json.dumps({"username": "admin", "password": "admin123"}),
        content_type="application/json",
    )
    assert login.status_code == 200
    body = login.get_json()
    assert body["username"] == "admin"
    assert body["role"] == "系统管理员"
    assert body["session_id"]

    status = client.get(f"/auth/status?session_id={body['session_id']}")
    assert status.status_code == 200
    assert status.get_json()["authenticated"] is True

    logout = client.post("/auth/logout", data=json.dumps({"session_id": body["session_id"]}), content_type="application/json")
    assert logout.status_code == 200
    assert logout.get_json()["logged_out"] is True

    status = client.get(f"/auth/status?session_id={body['session_id']}")
    assert status.status_code == 200
    assert status.get_json()["authenticated"] is False
