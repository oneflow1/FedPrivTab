from __future__ import annotations

from typing import Any

import pandas as pd
from flask import Flask, jsonify, request

import auth_db
from data_utils import (
    apply_column_preprocessing,
    generate_sample_data,
    preprocess_tabular_data,
    preprocessing_recommendations,
    train_test_data,
    validate_tabular_data,
)
from training import TrainConfig, parse_hidden_units, train_model


MANAGER_ROLES = {"系统管理员", "实验研究人员"}


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


def build_markdown_report(result: dict[str, Any]) -> str:
    metrics = result.get("metrics", {})
    history = result.get("history", {})
    lines = [
        "# FedPrivTab 实验报告",
        "",
        f"- 训练方案: {result.get('mode', '-')}",
        f"- 数据行数: {result.get('rows', '-')}",
        "- 模型结构: MLP（二分类表格特征输入，经隐藏层非线性变换后输出 logit）",
        "- 联邦方案: FedAvg 按客户端样本量对本地模型更新做加权平均；DP-FedAvg 在聚合前对更新裁剪并加入高斯噪声。",
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
            "DP-FedAvg 使用裁剪 $\\bar{g}_k = g_k \\cdot \\min(1, C / \\|g_k\\|_2)$，并在聚合更新中加入高斯噪声 $\\mathcal{N}(0, \\sigma^2 C^2 I)$；隐私预算以 $(\\epsilon, \\delta)$-DP 摘要呈现。",
        ])
    lines.extend([
        "",
        "## 结论分析",
        "",
        "集中式 MLP 作为性能参考上限；FedAvg + MLP 用于观察 Non-IID 多客户端训练带来的性能变化；DP-FedAvg + MLP 在上传更新前加入裁剪和高斯噪声，用于分析隐私保护与模型性能之间的权衡。",
    ])
    return "\n".join(lines)

app = Flask(__name__)


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


@app.post("/preprocess")
def preprocess() -> tuple[dict[str, Any], int]:
    payload = request.get_json(force=True)
    frame = pd.DataFrame(payload.get("records", []))
    target_column = payload.get("target_column", "target")
    if frame.empty:
        return {"error": "数据为空"}, 400
    try:
        recommendations = preprocessing_recommendations(frame, target_column=target_column)
        missing_strategies = payload.get("missing_strategies") or recommendations["missing_strategies"]
        scaler_strategies = payload.get("scaler_strategies") or recommendations["scaler_strategies"]
        processed = apply_column_preprocessing(
            frame,
            target_column=target_column,
            missing_strategies=missing_strategies,
            scaler_strategies=scaler_strategies,
        )
    except Exception as exc:
        return {"error": f"预处理失败: {exc}"}, 400
    validation = validate_tabular_data(processed, target_column=target_column)
    response = {
        "records": processed.to_dict(orient="records"),
        "columns": list(processed.columns),
        "recommendations": recommendations,
        "validation": {"valid": validation.valid, "message": validation.message, "details": validation.details},
        "rows": len(processed),
    }
    return response, 200 if validation.valid else 400


@app.post("/train")
def train() -> tuple[dict[str, Any], int]:
    payload = request.get_json(force=True)
    if "records" in payload:
        frame = pd.DataFrame(payload["records"])
    else:
        frame = generate_sample_data()
    validation = validate_tabular_data(frame, target_column=payload.get("target_column", "target"))
    if not validation.valid:
        return {"error": validation.message, "details": validation.details}, 400
    x_train, x_test, y_train, y_test, _ = train_test_data(frame, target_column=payload.get("target_column", "target"))
    config = TrainConfig(
        mode=payload.get("mode", "centralized"),
        epochs=int(payload.get("epochs", 3)),
        rounds=int(payload.get("rounds", 3)),
        clients=int(payload.get("clients", 4)),
        local_epochs=int(payload.get("local_epochs", 1)),
        batch_size=int(payload.get("batch_size", 16)),
        lr=float(payload.get("lr", 0.01)),
        hidden_layers=int(payload.get("hidden_layers", 2)),
        hidden_units=parse_hidden_units(payload.get("hidden_units", "64,32"), int(payload.get("hidden_layers", 2))),
        activation=payload.get("activation", "ReLU"),
        client_fraction=float(payload.get("client_fraction", 1.0)),
        dirichlet_alpha=float(payload.get("dirichlet_alpha", 0.3)),
        clip_norm=float(payload.get("clip_norm", 1.0)),
        noise_multiplier=float(payload.get("noise_multiplier", 0.2)),
        epsilon=float(payload.get("epsilon", 4.0)),
        delta=float(payload.get("delta", 1e-5)),
        non_iid=bool(payload.get("non_iid", False)),
        seed=int(payload.get("seed", 42)),
    )
    result = train_model(x_train, y_train, x_test, y_test, config)
    result["rows"] = len(frame)
    return jsonify(result), 200


@app.post("/report")
def report() -> tuple[dict[str, str], int]:
    payload = request.get_json(force=True)
    result = payload.get("result")
    if not isinstance(result, dict):
        return {"error": "缺少 result 对象"}, 400
    return {"markdown": build_markdown_report(result)}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
