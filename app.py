from __future__ import annotations

from typing import Any, Callable
import io
import json
import threading
import traceback
import uuid

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.model_selection import train_test_split

import auth_db
from data_utils import (
    apply_column_preprocessing,
    client_partitions,
    generate_sample_data,
    infer_adult_target_column,
    is_numeric_like,
    normalize_missing_markers,
    preprocess_tabular_data,
    preprocessing_recommendations,
    split_features_target,
    train_test_data,
    validate_tabular_data,
)
from training import TrainConfig, parse_hidden_units, train_model


MANAGER_ROLES = {"系统管理员"}


def current_session() -> dict[str, Any] | None:
    session_id = request.headers.get("X-Session-Id") or request.args.get("session_id")
    return auth_db.get_session(session_id)


def require_manager() -> tuple[dict[str, Any] | None, tuple[dict[str, Any], int] | None]:
    session = current_session()
    if not session:
        return None, ({"error": "未登录"}, 401)
    if session["role"] not in MANAGER_ROLES:
        return None, ({"error": "权限不足"}, 403)
    return session, None


def require_session() -> tuple[dict[str, Any] | None, tuple[dict[str, Any], int] | None]:
    session = current_session()
    if not session:
        return None, ({"error": "未登录"}, 401)
    return session, None


def build_markdown_report(result: dict[str, Any]) -> str:
    metrics = result.get("metrics", {})
    history = result.get("history", {})
    lines = [
        "# FedPrivTab 实验报告",
        "",
        f"- 训练方案: {result.get('mode', '-')}",
        f"- 数据行数: {result.get('rows', '-')}",
        "- 模型结构: MLP（二分类表格特征输入，经隐藏层非线性变换后输出 logit）",
        "- 联邦方案: 服务端托管 FL 模拟；服务端按 client_id 列或确定性 4 路切分模拟客户端分区，每个分区训练本地模型更新，FedAvg 按样本量聚合参数更新。",
        "",
        "## 评价指标",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
    ]
    for key in ["accuracy", "precision", "recall", "f1", "auc"]:
        value = metrics.get(key)
        lines.append(f"| {key} | {value if value is not None else '-'} |")
    lines.extend([
        "",
        "## 混淆矩阵",
        "",
        f"```text\n{metrics.get('confusion_matrix', [])}\n```",
        "",
        "## 客户端分布",
        "",
        "| 客户端 | 样本数 |",
        "|---:|---:|",
    ])
    for item in result.get("client_distribution", []):
        lines.append(f"| {item.get('client')} | {item.get('size')} |")
    lines.extend([
        "",
        "## Accuracy 曲线摘要",
        "",
        "| 轮次 | Loss | Accuracy |",
        "|---:|---:|---:|",
    ])
    losses = history.get("loss") or []
    accuracies = history.get("accuracy") or []
    for index in range(max(len(losses), len(accuracies))):
        loss = losses[index] if index < len(losses) else "-"
        accuracy = accuracies[index] if index < len(accuracies) else "-"
        lines.append(f"| {index + 1} | {loss} | {accuracy} |")
    if result.get("dp"):
        dp = result["dp"]
        lines.extend([
            "",
            "## 差分隐私参数",
            "",
            f"- epsilon: {dp.get('epsilon')}",
            f"- delta: {dp.get('delta')}",
            f"- clip_norm: {dp.get('clip_norm')}",
            f"- noise_multiplier: {dp.get('noise_multiplier')}",
            "",
            "| epsilon | delta | clip_norm | noise_multiplier | Accuracy | F1 |",
            "|---:|---:|---:|---:|---:|---:|",
            f"| {dp.get('epsilon')} | {dp.get('delta')} | {dp.get('clip_norm')} | {dp.get('noise_multiplier')} | {metrics.get('accuracy')} | {metrics.get('f1')} |",
            "",
            "DP-FedAvg 使用裁剪 $\\bar{g}_k = g_k \\cdot \\min(1, C / \\|g_k\\|_2)$，并在聚合更新中加入高斯噪声 $\\mathcal{N}(0, \\sigma^2 C^2 I)$；本项目中的 epsilon/delta 是记录和配置的实验参数，不是严格隐私 accountant 推导的保证。",
        ])
    lines.extend([
        "",
        "## 结论分析",
        "",
        "集中式训练作为性能参考上限；FedAvg 训练用于观察服务端分区模拟的多客户端训练效果；DP-FedAvg 训练在聚合更新前加入裁剪和高斯噪声，用于分析隐私保护与模型性能之间的权衡。",
    ])
    return "\n".join(lines)


def server_hosted_partitions(frame: pd.DataFrame, target_column: str, clients: int, seed: int, non_iid: bool, alpha: float):
    train_frame, test_frame = train_test_split(
        frame,
        test_size=0.25,
        random_state=seed,
        stratify=frame[target_column],
    )
    train_frame = train_frame.reset_index(drop=True)
    test_frame = test_frame.reset_index(drop=True)
    x_train, y_train, _ = split_features_target(train_frame, target_column)
    x_test, y_test, _ = split_features_target(test_frame, target_column)
    if "client_id" not in train_frame.columns:
        return x_train, x_test, y_train, y_test, client_partitions(y_train, clients, non_iid, seed, alpha), None

    client_values = train_frame["client_id"].astype(str)
    labels = sorted(client_values.unique().tolist())
    if len(labels) > clients:
        labels = labels[:clients]
    partitions = [np.flatnonzero(client_values.to_numpy() == label).astype(int) for label in labels]
    if len(partitions) < clients:
        assigned = set(np.concatenate(partitions).tolist()) if partitions else set()
        remaining = np.asarray([index for index in range(len(train_frame)) if index not in assigned], dtype=int)
        splits = np.array_split(remaining, clients - len(partitions)) if len(partitions) < clients else []
        start = len(partitions)
        partitions.extend(np.asarray(split, dtype=int) for split in splits)
        labels.extend([f"client-{index + 1}" for index in range(start, clients)])
    return x_train, x_test, y_train, y_test, partitions, labels


def _feature_columns(frame: pd.DataFrame, target_column: str) -> list[str]:
    return [column for column in frame.columns if column not in {target_column, "client_id"}]


def _client_dataset_specs(payload: dict[str, Any]) -> list[dict[str, str]]:
    client_datasets = payload.get("client_datasets")
    if isinstance(client_datasets, list) and client_datasets:
        specs = []
        for index, item in enumerate(client_datasets):
            if isinstance(item, dict):
                dataset_id = item.get("dataset_id")
                client_id = item.get("client_id") or item.get("id") or f"client-{index + 1}"
            else:
                dataset_id = item
                client_id = f"client-{index + 1}"
            if dataset_id:
                specs.append({"dataset_id": str(dataset_id), "client_id": str(client_id)})
        return specs

    client_dataset_ids = payload.get("client_dataset_ids")
    if isinstance(client_dataset_ids, list):
        return [
            {"dataset_id": str(dataset_id), "client_id": f"client-{index + 1}"}
            for index, dataset_id in enumerate(client_dataset_ids)
            if dataset_id
        ]
    return []


def _load_aligned_client_datasets(payload: dict[str, Any], target_column: str) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    specs = _client_dataset_specs(payload)
    if not specs:
        return None, None

    frames: list[pd.DataFrame] = []
    missing_ids: list[str] = []
    with DATASETS_LOCK:
        for spec in specs:
            stored = DATASETS.get(spec["dataset_id"])
            if stored is None:
                missing_ids.append(spec["dataset_id"])
                continue
            frame = stored.copy()
            resolved_target = infer_adult_target_column(frame, target_column)
            if resolved_target != target_column and resolved_target in frame.columns:
                frame = frame.rename(columns={resolved_target: target_column})
            frame["client_id"] = spec["client_id"]
            frames.append(frame)
    if missing_ids:
        return None, {"error": f"客户端 dataset_id 不存在或已失效: {', '.join(missing_ids)}"}
    if not frames:
        return None, {"error": "缺少可用的客户端预处理数据"}

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if target_column not in combined.columns:
        return None, {"error": f"缺少目标列: {target_column}"}
    for column in _feature_columns(combined, target_column):
        combined[column] = pd.to_numeric(combined[column].fillna(0), errors="coerce").fillna(0)
    return combined, None


def _client_label_distribution(frame: pd.DataFrame, target_column: str) -> list[dict[str, Any]]:
    if "client_id" not in frame.columns or target_column not in frame.columns:
        return []
    distribution = []
    for client_id, group in frame.groupby(frame["client_id"].astype(str), sort=True):
        labels = group[target_column]
        positives = int(labels.sum()) if pd.api.types.is_numeric_dtype(labels) else 0
        distribution.append({
            "client": str(client_id),
            "size": int(len(group)),
            "positive": positives,
            "negative": int(len(group) - positives),
        })
    return distribution

app = Flask(__name__)

ASYNC_JOBS: dict[str, dict[str, Any]] = {}
ASYNC_JOBS_LOCK = threading.Lock()
DATASETS: dict[str, pd.DataFrame] = {}
DATASETS_LOCK = threading.Lock()


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if key != "result"}


def get_async_job(job_id: str) -> dict[str, Any] | None:
    with ASYNC_JOBS_LOCK:
        job = ASYNC_JOBS.get(job_id)
        if not job:
            return None
        public = _public_job(job)
        if job.get("status") == "completed":
            public["result"] = job.get("result")
        return public


def submit_async_job(kind: str, label: str, func: Callable[[], tuple[dict[str, Any], int]]) -> dict[str, Any]:
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    job = {"job_id": job_id, "kind": kind, "label": label, "status": "queued", "progress": 0, "message": "等待执行"}
    with ASYNC_JOBS_LOCK:
        ASYNC_JOBS[job_id] = job

    def runner() -> None:
        with ASYNC_JOBS_LOCK:
            ASYNC_JOBS[job_id].update({"status": "running", "progress": 10, "message": "后端处理中"})
        try:
            result, status = func()
            with ASYNC_JOBS_LOCK:
                ASYNC_JOBS[job_id].update({
                    "status": "completed" if status < 400 else "failed",
                    "progress": 100,
                    "message": "处理完成" if status < 400 else result.get("error", "处理失败"),
                    "http_status": status,
                    "result": result,
                })
        except Exception as exc:
            with ASYNC_JOBS_LOCK:
                ASYNC_JOBS[job_id].update({
                    "status": "failed",
                    "progress": 100,
                    "message": str(exc),
                    "http_status": 500,
                    "traceback": traceback.format_exc(limit=5),
                })

    threading.Thread(target=runner, daemon=True).start()
    return _public_job(job)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Session-Id"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PATCH, OPTIONS"
    return response


@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def options_preflight(path: str = ""):
    return "", 204


@app.get("/health")
def health() -> tuple[dict[str, str], int]:
    return {"status": "ok"}, 200


@app.post("/auth/login")
def auth_login() -> tuple[dict[str, Any], int]:
    payload = request.get_json(force=True)
    username = str(payload.get("username", ""))
    password = str(payload.get("password", ""))
    session = auth_db.login(username, password)
    if not session:
        return {"error": "用户名或密码错误"}, 401
    return {
        "session_id": session["session_id"],
        "username": session["username"],
        "role": session["role"],
    }, 200


@app.post("/auth/logout")
def auth_logout() -> tuple[dict[str, Any], int]:
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id") or request.headers.get("X-Session-Id")
    logged_out = auth_db.logout(str(session_id) if session_id else None)
    return {"logged_out": logged_out}, 200


@app.get("/auth/status")
def auth_status() -> tuple[dict[str, Any], int]:
    session_id = request.args.get("session_id") or request.headers.get("X-Session-Id")
    session = auth_db.get_session(session_id)
    if not session:
        return {"authenticated": False}, 200
    return {
        "authenticated": True,
        "username": session["username"],
        "role": session["role"],
        "session_id": session["session_id"],
    }, 200


@app.get("/users")
def list_users() -> tuple[dict[str, Any], int]:
    _, error = require_manager()
    if error:
        return error
    role = request.args.get("role")
    return {"users": auth_db.list_users(role=role)}, 200


@app.post("/users")
def create_user() -> tuple[dict[str, Any], int]:
    _, error = require_manager()
    if error:
        return error
    payload = request.get_json(force=True)
    try:
        user = auth_db.create_user(
            str(payload.get("username", "")),
            str(payload.get("password", "")),
            str(payload.get("role", auth_db.CLIENT_ROLE)),
        )
    except ValueError as exc:
        return {"error": str(exc)}, 400
    except Exception:
        return {"error": "账号已存在或创建失败"}, 409
    return {"user": user}, 201


@app.patch("/users/<username>/password")
def update_user_password(username: str) -> tuple[dict[str, Any], int]:
    _, error = require_manager()
    if error:
        return error
    payload = request.get_json(force=True)
    password = str(payload.get("password", ""))
    try:
        changed = auth_db.change_password(username, None, password)
    except ValueError as exc:
        return {"error": str(exc)}, 400
    if not changed:
        return {"error": "用户不存在"}, 404
    return {"updated": True, "username": username}, 200


@app.patch("/users/<username>/status")
def update_user_status(username: str) -> tuple[dict[str, Any], int]:
    _, error = require_manager()
    if error:
        return error
    payload = request.get_json(force=True)
    if "is_active" not in payload:
        return {"error": "缺少 is_active"}, 400
    updated = auth_db.set_user_active(username, bool(payload["is_active"]))
    if not updated:
        return {"error": "用户不存在"}, 404
    return {"updated": True, "username": username, "is_active": bool(payload["is_active"])}, 200


@app.get("/sample-data")
def sample_data() -> tuple[dict[str, Any], int]:
    samples = int(request.args.get("samples", 240))
    features = int(request.args.get("features", 6))
    clients = int(request.args.get("clients", 4))
    seed = int(request.args.get("seed", 42))
    frame = generate_sample_data(samples=samples, features=features, clients=clients, seed=seed)
    return {"columns": list(frame.columns), "records": frame.to_dict(orient="records")}, 200


@app.get("/jobs/<job_id>")
def async_job_status(job_id: str) -> tuple[dict[str, Any], int]:
    job = get_async_job(job_id)
    if not job:
        return {"error": "任务不存在"}, 404
    return {"job": job}, 200


@app.post("/validate")
def validate() -> tuple[dict[str, Any], int]:
    payload = request.get_json(force=True)
    frame = pd.DataFrame(payload.get("records", []))
    target_column = payload.get("target_column", "target")
    if payload.get("apply_preprocess", False):
        frame = preprocess_tabular_data(
            frame,
            target_column=target_column,
            missing_strategy=payload.get("missing_strategy", "drop"),
            scaler=payload.get("scaler", "standard"),
        )
    result = validate_tabular_data(frame, target_column=target_column)
    response = {"valid": result.valid, "message": result.message, "details": result.details}
    if payload.get("apply_preprocess", False):
        response["records"] = frame.to_dict(orient="records")
    return response, 200 if result.valid else 400


def preprocess_core(
    frame: pd.DataFrame,
    target_column: str,
    missing_strategies: dict[str, str] | None,
    scaler_strategies: dict[str, str] | None,
    include_records: bool = True,
) -> tuple[dict[str, Any], int]:
    frame = normalize_missing_markers(frame)
    target_column = infer_adult_target_column(frame, target_column)
    if frame.empty:
        return {"error": "数据为空"}, 400
    try:
        recommendations = preprocessing_recommendations(frame, target_column=target_column)
        missing_strategies = missing_strategies or recommendations["missing_strategies"]
        scaler_strategies = scaler_strategies or recommendations["scaler_strategies"]
        processed = apply_column_preprocessing(
            frame,
            target_column=target_column,
            missing_strategies=missing_strategies,
            scaler_strategies=scaler_strategies,
        )
    except Exception as exc:
        return {"error": f"预处理失败: {exc}"}, 400
    validation = validate_tabular_data(processed, target_column=target_column)
    label_distribution = processed[target_column].value_counts().to_dict() if target_column in processed.columns else {}
    dataset_id = f"dataset-{uuid.uuid4().hex[:12]}"
    with DATASETS_LOCK:
        DATASETS[dataset_id] = processed.copy()
    response = {
        "dataset_id": dataset_id,
        "target_column": target_column,
        "records": processed.to_dict(orient="records") if include_records else [],
        "columns": list(processed.columns),
        "recommendations": recommendations,
        "validation": {"valid": validation.valid, "message": validation.message, "details": validation.details},
        "rows": len(processed),
        "summary": {
            "sample_count": len(processed),
            "feature_dim": len([column for column in processed.columns if column not in {target_column, "client_id"}]),
            "label_distribution": {str(key): int(value) for key, value in label_distribution.items()},
        },
    }
    return response, 200 if validation.valid else 400


def inspect_preprocess_frame(frame: pd.DataFrame, target_column: str = "income", preview_rows: int = 20) -> tuple[dict[str, Any], int]:
    frame = normalize_missing_markers(frame)
    target_column = infer_adult_target_column(frame, target_column)
    if frame.empty:
        return {"error": "数据为空"}, 400
    recommendations = preprocessing_recommendations(frame, target_column=target_column)
    missing_summary = []
    missing_strategies = {}
    for column in frame.columns:
        missing = int(frame[column].isna().sum())
        recommendation = recommendations["missing_strategies"].get(column)
        if recommendation is None:
            recommendation = "median" if is_numeric_like(frame[column]) else "mode"
        missing_strategies[column] = recommendation
        missing_summary.append({
            "column": column,
            "missing": missing,
            "rate": float(missing / max(len(frame), 1)),
            "recommendation": recommendation,
        })
    numeric_columns = [
        column
        for column in frame.columns
        if column not in {target_column, "client_id"} and is_numeric_like(frame[column])
    ]
    preview = frame.head(preview_rows).astype(object).where(pd.notna(frame.head(preview_rows)), None)
    return {
        "columns": list(frame.columns),
        "target_column": target_column,
        "rows": preview.to_dict(orient="records"),
        "preview_rows": len(preview),
        "row_count": int(len(frame)),
        "missing_summary": missing_summary,
        "missing_strategies": missing_strategies,
        "numeric_columns": numeric_columns,
        "scaler_strategies": {
            column: recommendations["scaler_strategies"].get(column, "standard")
            for column in numeric_columns
        },
        "recommendations": recommendations,
    }, 200


@app.post("/preprocess/inspect")
def inspect_preprocess() -> tuple[dict[str, Any], int]:
    if not request.files.get("file"):
        return {"error": "缺少 CSV 文件"}, 400
    file_bytes = request.files["file"].read()
    target_column = request.form.get("target_column", "income")
    preview_rows = int(request.form.get("preview_rows", 20))
    try:
        frame = pd.read_csv(io.BytesIO(file_bytes), na_values=["?", " ?", "? "])
    except Exception as exc:
        return {"error": f"读取 CSV 失败: {exc}"}, 400
    return inspect_preprocess_frame(frame, target_column=target_column, preview_rows=max(1, min(preview_rows, 100)))


@app.post("/preprocess")
def preprocess() -> tuple[dict[str, Any], int]:
    run_async = request.args.get("async") == "true" or request.form.get("async") == "true"
    if request.files.get("file"):
        file_bytes = request.files["file"].read()
        target_column = request.form.get("target_column", "income")
        missing_strategies = json.loads(request.form.get("missing_strategies", "{}"))
        scaler_strategies = json.loads(request.form.get("scaler_strategies", "{}"))
        include_records = request.form.get("summary_only") != "true"

        def work() -> tuple[dict[str, Any], int]:
            frame = pd.read_csv(io.BytesIO(file_bytes), na_values=["?", " ?", "? "])
            return preprocess_core(frame, target_column, missing_strategies, scaler_strategies, include_records=include_records)
    else:
        payload = request.get_json(force=True)
        records = payload.get("records", [])
        target_column = payload.get("target_column", "income")
        missing_strategies = payload.get("missing_strategies") or None
        scaler_strategies = payload.get("scaler_strategies") or None
        include_records = not bool(payload.get("summary_only", False))

        def work() -> tuple[dict[str, Any], int]:
            return preprocess_core(pd.DataFrame(records), target_column, missing_strategies, scaler_strategies, include_records=include_records)

    if run_async:
        job = submit_async_job("preprocess", "数据预处理", work)
        return {"job": job}, 202
    return work()


@app.get("/preprocess/versions")
def list_preprocess_versions() -> tuple[dict[str, Any], int]:
    session, error = require_session()
    if error:
        return error
    return {"versions": auth_db.list_preprocess_versions(session)}, 200


@app.post("/preprocess/versions")
def create_preprocess_version() -> tuple[dict[str, Any], int]:
    session, error = require_session()
    if error:
        return error
    payload = request.get_json(force=True)
    try:
        version = auth_db.save_preprocess_version(session, payload)
    except ValueError as exc:
        return {"error": str(exc)}, 400
    return {"version": version}, 201


def train_core(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    mode = payload.get("mode", "centralized")
    if mode not in {"centralized", "fedavg", "dp_fedavg"}:
        return {"error": "不支持的训练模式"}, 400
    distributed_defaults = mode in {"fedavg", "dp_fedavg"}
    dp_defaults = mode == "dp_fedavg"
    lr_schedule = str(payload.get("lr_schedule", "step_decay"))
    if lr_schedule not in {"constant", "step_decay", "linear_decay"}:
        return {"error": "不支持的学习率调度策略"}, 400
    try:
        lr_decay = float(payload.get("lr_decay", 0.5))
        lr_step_size = int(payload.get("lr_step_size", 15))
        lr_min = float(payload.get("lr_min", 0.005))
    except (TypeError, ValueError):
        return {"error": "学习率调度参数必须为数值"}, 400
    if lr_step_size <= 0:
        return {"error": "lr_step_size 必须大于 0"}, 400
    if lr_min <= 0:
        return {"error": "lr_min 必须大于 0"}, 400
    if lr_schedule == "step_decay" and not (0 < lr_decay <= 1):
        return {"error": "lr_decay 必须在 (0, 1] 范围内"}, 400
    dataset_id = payload.get("dataset_id")
    target_column = payload.get("target_column", "income")
    client_distribution_override = None
    if mode in {"fedavg", "dp_fedavg"}:
        if dataset_id or "records" in payload or not _client_dataset_specs(payload):
            return {"error": "分布式训练只能使用客户端数据准备页生成的客户端数据，请先完成客户端预处理"}, 400
        client_frame, client_error = _load_aligned_client_datasets(payload, target_column)
        if client_error:
            return client_error, 400
    else:
        client_frame = None

    if client_frame is not None:
        frame = client_frame
        target_column = infer_adult_target_column(frame, target_column)
        client_distribution_override = _client_label_distribution(frame, target_column)
        payload["clients"] = max(int(payload.get("clients", len(client_distribution_override) or 1)), len(client_distribution_override) or 1)
    elif dataset_id:
        with DATASETS_LOCK:
            stored = DATASETS.get(str(dataset_id))
        if stored is None:
            return {"error": "dataset_id 不存在或已失效，请重新预处理数据"}, 400
        frame = stored.copy()
    elif "records" in payload:
        frame = normalize_missing_markers(pd.DataFrame(payload["records"]))
    else:
        return {"error": "请先完成集中式数据准备或客户端数据准备"}, 400
    target_column = infer_adult_target_column(frame, target_column)
    validation = validate_tabular_data(frame, target_column=target_column)
    if not validation.valid:
        return {"error": validation.message, "details": validation.details}, 400
    x_train, x_test, y_train, y_test, _ = train_test_data(frame, target_column=target_column)
    config = TrainConfig(
        mode=mode,
        epochs=int(payload.get("epochs", 50)),
        rounds=int(payload.get("rounds", 50 if distributed_defaults else 50)),
        clients=int(payload.get("clients", 4)),
        local_epochs=int(payload.get("local_epochs", 1)),
        batch_size=int(payload.get("batch_size", 32 if distributed_defaults else 128)),
        lr=float(payload.get("lr", 0.03 if dp_defaults else 0.05)),
        lr_schedule=lr_schedule,
        lr_decay=lr_decay,
        lr_step_size=lr_step_size,
        lr_min=lr_min,
        hidden_layers=int(payload.get("hidden_layers", 2)),
        hidden_units=parse_hidden_units(payload.get("hidden_units", "64,32"), int(payload.get("hidden_layers", 2))),
        activation=payload.get("activation", "ReLU"),
        client_fraction=float(payload.get("client_fraction", 1.0)),
        dirichlet_alpha=float(payload.get("dirichlet_alpha", 0.3)),
        clip_norm=float(payload.get("clip_norm", 1.0)),
        noise_multiplier=float(payload.get("noise_multiplier", 0.1)),
        epsilon=float(payload.get("epsilon", 4.0)),
        delta=float(payload.get("delta", 1e-5)),
        non_iid=bool(payload.get("non_iid", False)),
        seed=int(payload.get("seed", 42)),
    )
    partition_override = None
    client_labels = None
    if mode in {"fedavg", "dp_fedavg"}:
        x_train, x_test, y_train, y_test, partition_override, client_labels = server_hosted_partitions(
            frame,
            target_column=target_column,
            clients=config.clients,
            seed=config.seed,
            non_iid=config.non_iid,
            alpha=config.dirichlet_alpha,
        )
    result = train_model(x_train, y_train, x_test, y_test, config, partition_override, client_labels)
    result["rows"] = len(frame)
    result["feature_dim"] = len(_feature_columns(frame, target_column))
    if client_frame is not None:
        result["aligned_feature_dim"] = result["feature_dim"]
        if client_distribution_override:
            result["client_distribution"] = client_distribution_override
    return result, 200


@app.post("/train")
def train() -> tuple[dict[str, Any], int]:
    payload = request.get_json(force=True)
    run_async = request.args.get("async") == "true" or bool(payload.pop("async", False))
    if run_async:
        job = submit_async_job("train", f"{payload.get('mode', 'centralized')} 训练", lambda: train_core(payload))
        return {"job": job}, 202
    result, status = train_core(payload)
    return jsonify(result), status


@app.post("/federated/aggregate")
def federated_aggregate() -> tuple[dict[str, Any], int]:
    payload = request.get_json(force=True)
    mode = payload.get("mode", "fedavg")
    if mode not in {"fedavg", "dp_fedavg"}:
        return {"error": "仅支持 FedAvg/DP-FedAvg 聚合"}, 400
    if "records" in payload or any("records" in update for update in payload.get("client_updates", []) if isinstance(update, dict)):
        return {"error": "联邦聚合端点只接收模型更新/统计信息，不接收原始 records"}, 400
    updates = payload.get("client_updates")
    if not isinstance(updates, list) or not updates:
        return {"error": "缺少 client_updates"}, 400
    parsed_updates: list[dict[str, Any]] = []
    total_weight = 0.0
    vector_length: int | None = None
    for index, update in enumerate(updates):
        if not isinstance(update, dict):
            return {"error": f"client_updates[{index}] 格式错误"}, 400
        vector = update.get("weights_delta", update.get("gradient"))
        if not isinstance(vector, list) or not vector:
            return {"error": f"client_updates[{index}] 缺少 weights_delta/gradient"}, 400
        try:
            numeric_vector = [float(value) for value in vector]
            weight = float(update.get("weight") or update.get("samples") or 1)
        except (TypeError, ValueError):
            return {"error": f"client_updates[{index}] 包含非数值更新"}, 400
        if weight <= 0:
            return {"error": f"client_updates[{index}] weight 必须大于 0"}, 400
        if vector_length is None:
            vector_length = len(numeric_vector)
        elif len(numeric_vector) != vector_length:
            return {"error": "所有 client update 向量长度必须一致"}, 400
        parsed_updates.append({"client_id": update.get("client_id", f"client-{index + 1}"), "vector": numeric_vector, "weight": weight})
        total_weight += weight
    aggregated = [
        sum(update["vector"][index] * update["weight"] for update in parsed_updates) / total_weight
        for index in range(vector_length or 0)
    ]
    return {
        "mode": mode,
        "aggregated_update": aggregated,
        "client_count": len(parsed_updates),
        "total_weight": total_weight,
        "statistics": payload.get("statistics") or {},
        "protocol": "服务端仅聚合客户端模型参数/梯度更新和统计信息；原始记录不进入该端点。",
    }, 200


@app.post("/report")
def report() -> tuple[dict[str, str], int]:
    payload = request.get_json(force=True)
    result = payload.get("result")
    if not isinstance(result, dict):
        return {"error": "缺少 result 对象"}, 400
    return {"markdown": build_markdown_report(result)}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
