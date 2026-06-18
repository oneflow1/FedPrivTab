#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/mywork/final_outputs"
RESULTS_JSON = OUT_DIR / "experiment_results.json"
SUMMARY_CSV = OUT_DIR / "metrics_summary.csv"
API_BASE_DEFAULT = "http://127.0.0.1:5000"


def api_json(method: str, url: str, payload: dict[str, Any] | None = None, files: dict[str, Path] | None = None, data: dict[str, str] | None = None, timeout: int = 60) -> dict[str, Any]:
    if files:
        boundary = f"----fedprivtab-{int(time.time() * 1000)}"
        body = bytearray()
        for key, value in (data or {}).items():
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
        for key, path in files.items():
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(f'Content-Disposition: form-data; name="{key}"; filename="{path.name}"\r\n'.encode())
            body.extend(b"Content-Type: text/csv\r\n\r\n")
            body.extend(path.read_bytes())
            body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode())
        req = request.Request(url, data=bytes(body), method=method, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    else:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {} if body is None else {"Content-Type": "application/json"}
        req = request.Request(url, data=body, method=method, headers=headers)

    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Cannot reach {url}: {exc.reason}") from exc


def wait_job(api_base: str, job_id: str, poll_seconds: float = 2.0, timeout_seconds: int = 1800) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        payload = api_json("GET", f"{api_base}/jobs/{job_id}", timeout=30)
        job = payload.get("job", payload)
        status = job.get("status")
        if status == "completed":
            return job.get("result", job)
        if status == "failed":
            raise RuntimeError(f"Async job {job_id} failed: {job}")
        time.sleep(poll_seconds)
    raise TimeoutError(f"Async job {job_id} did not finish within {timeout_seconds} seconds")


def preprocess_csv(api_base: str, csv_path: Path, client_id: str | None = None, role: str = "centralized") -> dict[str, Any]:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    fields = {"role": role, "target_column": "income"}
    if client_id:
        fields["client_id"] = client_id
    started = api_json(
        "POST",
        f"{api_base}/preprocess?async=true",
        files={"file": csv_path},
        data=fields,
        timeout=120,
    )
    job = started.get("job", started)
    job_id = job.get("job_id") or job.get("id")
    if not job_id:
        return started
    result = wait_job(api_base, str(job_id))
    if client_id:
        result["client_id"] = result.get("client_id") or client_id
    return result


def train(api_base: str, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    result = api_json("POST", f"{api_base}/train", payload=payload, timeout=1800)
    return {"name": name, "requested_config": payload, **result}


def compact_preprocess_summary(preprocess: dict[str, Any]) -> dict[str, Any]:
    summary = preprocess.get("summary") if isinstance(preprocess.get("summary"), dict) else {}
    validation = preprocess.get("validation") if isinstance(preprocess.get("validation"), dict) else {}
    details = validation.get("details") if isinstance(validation.get("details"), dict) else {}
    columns = preprocess.get("columns") or details.get("columns") or []

    compact: dict[str, Any] = {
        "dataset_id": preprocess.get("dataset_id"),
        "client_id": preprocess.get("client_id"),
        "rows": preprocess.get("rows", summary.get("sample_count")),
        "row_count": preprocess.get("rows", summary.get("sample_count")),
        "sample_count": summary.get("sample_count", preprocess.get("rows")),
        "column_count": len(columns) if isinstance(columns, list) else preprocess.get("column_count"),
        "columns": columns if isinstance(columns, list) and len(columns) <= 200 else None,
        "feature_dim": summary.get("feature_dim", preprocess.get("feature_dim")),
        "target_column": preprocess.get("target_column"),
        "validation": {
            key: value
            for key, value in validation.items()
            if key != "details"
        },
    }
    if "label_distribution" in summary:
        compact["label_distribution"] = summary["label_distribution"]
    if "missing_summary" in summary and isinstance(summary["missing_summary"], dict) and len(summary["missing_summary"]) <= 200:
        compact["missing_summary"] = summary["missing_summary"]

    return {key: value for key, value in compact.items() if value is not None}


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    keep_keys = [
        "name",
        "mode",
        "metrics",
        "history",
        "requested_config",
        "dp",
        "client_distribution",
        "rows",
        "feature_dim",
        "aligned_feature_dim",
    ]
    return {key: result.get(key) for key in keep_keys}


def build_payloads(central_dataset_id: str, client_dataset_ids: list[str]) -> dict[str, dict[str, Any]]:
    client_datasets = [
        {"client_id": f"client-{index + 1}", "dataset_id": dataset_id}
        for index, dataset_id in enumerate(client_dataset_ids)
    ]
    common_mlp = {
        "target_column": "income",
        "hidden_layers": 2,
        "hidden_units": "64,32",
        "activation": "ReLU",
        "lr_schedule": "step_decay",
        "lr_decay": 0.5,
        "lr_step_size": 15,
        "lr_min": 0.005,
        "seed": 42,
    }
    centralized = {
        **common_mlp,
        "mode": "centralized",
        "dataset_id": central_dataset_id,
        "epochs": 50,
        "batch_size": 128,
        "lr": 0.05,
    }
    fedavg = {
        **common_mlp,
        "mode": "fedavg",
        "clients": 4,
        "rounds": 50,
        "local_epochs": 1,
        "batch_size": 32,
        "lr": 0.05,
        "client_fraction": 1.0,
        "dirichlet_alpha": 0.3,
        "non_iid": True,
        "client_dataset_ids": client_dataset_ids,
        "client_datasets": client_datasets,
    }
    dp_fedavg = {
        **fedavg,
        "mode": "dp_fedavg",
        "lr": 0.03,
        "clip_norm": 1.0,
        "noise_multiplier": 0.1,
        "epsilon": 4.0,
        "delta": 1e-5,
    }
    payloads = {
        "centralized_mlp": centralized,
        "fedavg_mlp": fedavg,
        "dp_fedavg_mlp": dp_fedavg,
        "dp_noise_0.1": {**dp_fedavg, "noise_multiplier": 0.1},
        "dp_noise_0.2": {**dp_fedavg, "noise_multiplier": 0.2},
        "dp_noise_0.5": {**dp_fedavg, "noise_multiplier": 0.5},
        "alpha_0.1": {**fedavg, "dirichlet_alpha": 0.1},
        "alpha_0.3": {**fedavg, "dirichlet_alpha": 0.3},
        "alpha_1.0": {**fedavg, "dirichlet_alpha": 1.0},
    }
    return payloads


def write_summary(results: list[dict[str, Any]], path: Path) -> None:
    metric_keys = ["accuracy", "precision", "recall", "f1", "auc"]
    config_keys = [
        "epochs",
        "rounds",
        "clients",
        "local_epochs",
        "batch_size",
        "lr",
        "lr_schedule",
        "lr_decay",
        "lr_step_size",
        "lr_min",
        "hidden_layers",
        "hidden_units",
        "activation",
        "client_fraction",
        "dirichlet_alpha",
        "clip_norm",
        "noise_multiplier",
        "epsilon",
        "delta",
        "seed",
    ]
    fields = ["name", "mode", "rows", "feature_dim", "aligned_feature_dim", *metric_keys, *config_keys]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for item in results:
            requested = item.get("requested_config") or {}
            metrics = item.get("metrics") or {}
            dp = item.get("dp") or {}
            row = {
                "name": item.get("name"),
                "mode": item.get("mode"),
                "rows": item.get("rows"),
                "feature_dim": item.get("feature_dim"),
                "aligned_feature_dim": item.get("aligned_feature_dim"),
            }
            row.update({key: metrics.get(key) for key in metric_keys})
            for key in config_keys:
                row[key] = dp.get(key, requested.get(key))
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate cached FedPrivTab post-project API results.")
    parser.add_argument("--api-base", default=API_BASE_DEFAULT)
    args = parser.parse_args()
    api_base = args.api_base.rstrip("/")

    try:
        health = api_json("GET", f"{api_base}/health", timeout=5)
    except RuntimeError as exc:
        raise SystemExit(f"Local API is not reachable at {api_base}/health: {exc}") from exc
    if health.get("status") != "ok":
        raise SystemExit(f"Unexpected health response from {api_base}/health: {health}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    central_preprocess = preprocess_csv(api_base, ROOT / "data/raw/adult.csv", role="centralized")
    client_preprocess = [
        preprocess_csv(api_base, ROOT / f"data/processed/adult_noniid_client_{index}.csv", client_id=f"client-{index}", role="client")
        for index in range(1, 5)
    ]
    central_dataset_id = central_preprocess["dataset_id"]
    client_dataset_ids = [item["dataset_id"] for item in client_preprocess]
    payloads = build_payloads(central_dataset_id, client_dataset_ids)
    results = [compact_result(train(api_base, name, payload)) for name, payload in payloads.items()]

    cache = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "api_base": api_base,
            "source": "real /preprocess?async=true, /jobs/<id>, and /train API responses",
            "central_csv": "data/raw/adult.csv",
            "client_csvs": [f"data/processed/adult_noniid_client_{index}.csv" for index in range(1, 5)],
            "central_preprocess": compact_preprocess_summary(central_preprocess),
            "client_preprocess": [compact_preprocess_summary(item) for item in client_preprocess],
            "dp_note": "DP-FedAvg demonstrates L2 clipping plus Gaussian noise; epsilon/delta are recorded experiment parameters, not strict accountant-derived guarantees.",
        },
        "results": results,
    }
    RESULTS_JSON.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(results, SUMMARY_CSV)
    print(f"Wrote {RESULTS_JSON}")
    print(f"Wrote {SUMMARY_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
