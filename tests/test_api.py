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

    for mode in ["centralized", "fedavg", "dp_fedavg"]:
        payload = {
            "records": records,
            "mode": mode,
            "epochs": 1,
            "rounds": 1,
            "clients": 2,
            "seed": 3,
            "non_iid": True,
        }
        response = client.post("/train", data=json.dumps(payload), content_type="application/json")
        assert response.status_code == 200
        body = response.get_json()
        assert body["mode"] == mode
        assert "metrics" in body
        assert "history" in body
        assert "client_distribution" in body
        if mode == "dp_fedavg":
            assert body["dp"] is not None

        report = client.post("/report", data=json.dumps({"result": body}), content_type="application/json")
        assert report.status_code == 200
        assert "FedPrivTab 实验报告" in report.get_json()["markdown"]
