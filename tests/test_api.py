import json
from io import BytesIO

import app as app_module
import auth_db
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

    column_records = [dict(record) for record in records]
    column_records[0]["feature_0"] = None
    column_records[1]["numeric_text"] = None
    for index, record in enumerate(column_records):
        record["numeric_text"] = None if index == 1 else str(index + 1)
    column_records = json.loads(json.dumps(column_records, allow_nan=False, default=str))
    column_preprocessed = client.post(
        "/preprocess",
        data=json.dumps(
            {
                "records": column_records,
                "target_column": "target",
                "missing_strategies": {"feature_0": "median", "numeric_text": "median"},
                "scaler_strategies": {"feature_0": "standard", "numeric_text": "minmax"},
            }
        ),
        content_type="application/json",
    )
    assert column_preprocessed.status_code == 200
    column_body = column_preprocessed.get_json()
    assert column_body["validation"]["valid"] is True
    assert column_body["recommendations"]["missing_strategies"]["numeric_text"] == "median"
    assert "records" in column_body

    csv_bytes = "feature_0,feature_1,target,client_id\n1,2,0,0\n,3,1,1\n4,5,0,0\n6,7,1,1\n8,9,0,0\n10,11,1,1\n12,13,0,0\n14,15,1,1\n16,17,0,0\n18,19,1,1\n20,21,0,0\n22,23,1,1\n24,25,0,0\n26,27,1,1\n28,29,0,0\n30,31,1,1\n32,33,0,0\n34,35,1,1\n36,37,0,0\n38,39,1,1\n".encode()
    multipart = client.post(
        "/preprocess",
        data={
            "file": (BytesIO(csv_bytes), "sample.csv"),
            "target_column": "target",
            "missing_strategies": json.dumps({"feature_0": "median"}),
            "scaler_strategies": json.dumps({"feature_0": "standard"}),
        },
        content_type="multipart/form-data",
    )
    assert multipart.status_code == 200
    assert multipart.get_json()["validation"]["valid"] is True

    adult_csv = (
        "age,workclass,hours_per_week,income\n"
        "39,State-gov,40,<=50K\n"
        "50,?,13,<=50K\n"
        "38,Private,40,<=50K\n"
        "53,Private,40,>50K\n"
        "28,Private,40,>50K\n"
    ).encode()
    inspection = client.post(
        "/preprocess/inspect",
        data={"file": (BytesIO(adult_csv), "adult.csv"), "target_column": "target", "preview_rows": "2"},
        content_type="multipart/form-data",
    )
    assert inspection.status_code == 200
    inspection_body = inspection.get_json()
    assert inspection_body["target_column"] == "income"
    assert inspection_body["columns"] == ["age", "workclass", "hours_per_week", "income"]
    assert inspection_body["row_count"] == 5
    assert len(inspection_body["rows"]) == 2
    assert "records" not in inspection_body
    workclass_summary = next(item for item in inspection_body["missing_summary"] if item["column"] == "workclass")
    assert workclass_summary["missing"] == 1
    assert inspection_body["numeric_columns"] == ["age", "hours_per_week"]

    for mode in ["centralized"]:
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
        if mode == "centralized":
            assert body["client_distribution"] == []
        else:
            assert len(body["client_distribution"]) == 2
            assert sum(item["size"] for item in body["client_distribution"]) > 0
            assert "服务端托管 FL 模拟" in body["protocol"]
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

    aggregate = client.post(
        "/federated/aggregate",
        data=json.dumps(
            {
                "mode": "fedavg",
                "client_updates": [
                    {"client_id": "client-1", "weights_delta": [1, 2], "weight": 2},
                    {"client_id": "client-2", "weights_delta": [3, 4], "weight": 1},
                ],
                "statistics": {"clients": [{"client": "client-1", "size": 2}]},
            }
        ),
        content_type="application/json",
    )
    assert aggregate.status_code == 200
    assert aggregate.get_json()["aggregated_update"] == [5 / 3, 8 / 3]

    aggregate_with_records = client.post(
        "/federated/aggregate",
        data=json.dumps({"mode": "fedavg", "records": records, "client_updates": [{"weights_delta": [1], "weight": 1}]}),
        content_type="application/json",
    )
    assert aggregate_with_records.status_code == 400


def test_federated_train_uses_prepared_client_datasets_with_aligned_features() -> None:
    client = app.test_client()

    def records_for(colors: list[str]) -> list[dict[str, object]]:
        records = []
        for index in range(24):
            records.append(
                {
                    "age": 20 + index,
                    "color": colors[index % len(colors)],
                    "target": index % 2,
                }
            )
        return records

    client_1 = client.post(
        "/preprocess",
        data=json.dumps({"records": records_for(["red", "blue"]), "target_column": "target", "summary_only": True}),
        content_type="application/json",
    )
    client_2 = client.post(
        "/preprocess",
        data=json.dumps({"records": records_for(["red", "green"]), "target_column": "target", "summary_only": True}),
        content_type="application/json",
    )
    assert client_1.status_code == 200
    assert client_2.status_code == 200
    assert set(client_1.get_json()["columns"]) != set(client_2.get_json()["columns"])

    client_datasets = [
        {"client_id": "client-1", "dataset_id": client_1.get_json()["dataset_id"]},
        {"client_id": "client-2", "dataset_id": client_2.get_json()["dataset_id"]},
    ]
    for mode in ["fedavg", "dp_fedavg"]:
        response = client.post(
            "/train",
            data=json.dumps(
                {
                    "mode": mode,
                    "target_column": "target",
                    "client_datasets": client_datasets,
                    "rounds": 1,
                    "clients": 2,
                    "local_epochs": 1,
                    "batch_size": 8,
                    "seed": 7,
                }
            ),
            content_type="application/json",
        )
        assert response.status_code == 200
        body = response.get_json()
        assert body["mode"] == mode
        assert body["rows"] == 48
        assert body["aligned_feature_dim"] == 4
        assert {item["client"] for item in body["client_distribution"]} == {"client-1", "client-2"}
        assert sum(item["size"] for item in body["client_distribution"]) == 48


def test_train_api_applies_mode_specific_defaults(monkeypatch) -> None:
    client = app.test_client()
    captured_configs = []

    def fake_train_model(x_train, y_train, x_test, y_test, config, partition_override=None, client_labels=None):
        captured_configs.append(config)
        return {
            "mode": config.mode,
            "history": {"loss": [], "accuracy": []},
            "metrics": {"accuracy": 0.0, "f1": 0.0, "auc": 0.0},
            "client_distribution": [],
            "predictions": [],
            "protocol": "",
            "dp": None,
        }

    monkeypatch.setattr(app_module, "train_model", fake_train_model)

    records = [
        {"age": 20 + index, "hours_per_week": 30 + index, "target": index % 2}
        for index in range(48)
    ]
    centralized = client.post(
        "/train",
        data=json.dumps(
            {
                "mode": "centralized",
                "records": records,
                "target_column": "target",
                "lr_schedule": "step_decay",
                "lr_decay": 0.25,
                "lr_step_size": 4,
                "lr_min": 0.002,
            }
        ),
        content_type="application/json",
    )
    assert centralized.status_code == 200
    assert captured_configs[-1].rounds == 50
    assert captured_configs[-1].epochs == 50
    assert captured_configs[-1].batch_size == 128
    assert captured_configs[-1].lr == 0.05
    assert captured_configs[-1].lr_schedule == "step_decay"
    assert captured_configs[-1].lr_decay == 0.25
    assert captured_configs[-1].lr_step_size == 4
    assert captured_configs[-1].lr_min == 0.002

    client_datasets = []
    for client_index in range(2):
        prepared = client.post(
            "/preprocess",
            data=json.dumps(
                {
                    "records": records[client_index * 24 : (client_index + 1) * 24],
                    "target_column": "target",
                    "summary_only": True,
                }
            ),
            content_type="application/json",
        )
        assert prepared.status_code == 200
        client_datasets.append({"client_id": f"client-{client_index + 1}", "dataset_id": prepared.get_json()["dataset_id"]})

    for mode in ["fedavg", "dp_fedavg"]:
        response = client.post(
            "/train",
            data=json.dumps({"mode": mode, "target_column": "target", "client_datasets": client_datasets}),
            content_type="application/json",
        )
        assert response.status_code == 200
        assert captured_configs[-1].rounds == 50
        assert captured_configs[-1].batch_size == 32
        assert captured_configs[-1].lr == (0.03 if mode == "dp_fedavg" else 0.05)
        assert captured_configs[-1].lr_schedule == "step_decay"
        assert captured_configs[-1].lr_step_size == 15
        assert captured_configs[-1].lr_min == 0.005
        assert captured_configs[-1].local_epochs == 1
        if mode == "dp_fedavg":
            assert captured_configs[-1].clip_norm == 1.0
            assert captured_configs[-1].noise_multiplier == 0.1


def test_train_api_rejects_invalid_lr_schedule() -> None:
    client = app.test_client()
    records = [
        {"age": 20 + index, "hours_per_week": 30 + index, "target": index % 2}
        for index in range(48)
    ]

    response = client.post(
        "/train",
        data=json.dumps(
            {
                "mode": "centralized",
                "records": records,
                "target_column": "target",
                "lr_schedule": "cosine",
            }
        ),
        content_type="application/json",
    )
    assert response.status_code == 400
    assert "学习率调度" in response.get_json()["error"]


def test_federated_train_rejects_centralized_dataset_id() -> None:
    client = app.test_client()
    sample = client.get("/sample-data?samples=50&features=4&clients=2&seed=13")
    assert sample.status_code == 200

    preprocessed = client.post(
        "/preprocess",
        data=json.dumps({"records": sample.get_json()["records"], "target_column": "target", "summary_only": True}),
        content_type="application/json",
    )
    assert preprocessed.status_code == 200
    dataset_id = preprocessed.get_json()["dataset_id"]

    response = client.post(
        "/train",
        data=json.dumps({"mode": "fedavg", "dataset_id": dataset_id, "target_column": "target", "rounds": 1, "clients": 2}),
        content_type="application/json",
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "分布式训练只能使用客户端数据准备页生成的客户端数据，请先完成客户端预处理"


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


def test_admin_is_the_only_default_user_and_can_list_users(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(auth_db, "DEFAULT_DB_PATH", tmp_path / "api-admin-only.sqlite3")
    client = app.test_client()
    login = client.post(
        "/auth/login",
        data=json.dumps({"username": "admin", "password": "admin123"}),
        content_type="application/json",
    )
    session_id = login.get_json()["session_id"]

    users = client.get("/users", headers={"X-Session-Id": session_id})
    assert users.status_code == 200
    listed_users = users.get_json()["users"]
    assert [user["username"] for user in listed_users] == ["admin"]
    assert listed_users[0]["role"] == "系统管理员"


def test_unknown_non_admin_user_cannot_manage_users(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(auth_db, "DEFAULT_DB_PATH", tmp_path / "api-admin-only-deny.sqlite3")
    client = app.test_client()
    login = client.post(
        "/auth/login",
        data=json.dumps({"username": "client-1", "password": "client123"}),
        content_type="application/json",
    )
    assert login.status_code == 401


def test_preprocess_version_api_auth_and_role_scope(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "api-auth.sqlite3"
    monkeypatch.setattr(auth_db, "DEFAULT_DB_PATH", db_path)
    client = app.test_client()

    unauthenticated = client.get("/preprocess/versions")
    assert unauthenticated.status_code == 401

    admin_login = client.post("/auth/login", data=json.dumps({"username": "admin", "password": "admin123"}), content_type="application/json")
    client_1_login = client.post("/auth/login", data=json.dumps({"username": "client-1", "password": "client123"}), content_type="application/json")
    admin_session = admin_login.get_json()["session_id"]
    assert client_1_login.status_code == 401

    admin_payload = {
        "id": "api-admin-v1",
        "scope": "centralized",
        "name": "centralized preprocess",
        "original_filename": "admin.csv",
        "target_column": "target",
        "row_count": 1,
        "missing_strategies": {"feature": "median"},
        "scaler_strategies": {"feature": "standard"},
        "message": "admin ok",
        "records": [{"feature": 1, "target": 0}],
    }
    admin_saved = client.post(
        "/preprocess/versions",
        data=json.dumps(admin_payload),
        content_type="application/json",
        headers={"X-Session-Id": admin_session},
    )
    assert admin_saved.status_code == 201
    assert admin_saved.get_json()["version"]["scope"] == "centralized"

    admin_versions = client.get("/preprocess/versions", headers={"X-Session-Id": admin_session})
    admin_version_rows = admin_versions.get_json()["versions"]
    assert [version["id"] for version in admin_version_rows] == ["api-admin-v1"]
    assert admin_version_rows[0]["records"] == [{"feature": 1, "target": 0}]
