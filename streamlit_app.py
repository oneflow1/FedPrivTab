from __future__ import annotations

from html import escape
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import streamlit as st

import auth_db
from app import build_markdown_report
from data_utils import generate_sample_data, preprocess_tabular_data, train_test_data, validate_tabular_data
from training import TrainConfig, parse_hidden_units, train_model


ROLE_HINTS = {
    "系统管理员": "可管理客户端、审核数据状态，并启动训练任务。",
    "客户端用户": "可上传本地 CSV 数据，查看校验结果和参与状态。",
    "实验研究人员": "可配置 MLP、联邦学习和差分隐私参数，分析实验结果。",
}

PAGES = [
    "首页",
    "客户端管理页",
    "数据预处理页",
    "数据分析页",
    "实验训练页",
    "结果分析页",
]

ROLE_PAGES = {
    "系统管理员": PAGES,
    "客户端用户": ["首页", "数据预处理页", "数据分析页", "结果分析页"],
    "实验研究人员": PAGES,
}

SCHEME_LABELS = {
    "centralized": "集中式 MLP",
    "fedavg": "FedAvg + MLP",
    "dp_fedavg": "DP-FedAvg + MLP",
}

PAGE_ICONS = {
    "首页": "⌂",
    "客户端管理页": "▦",
    "数据预处理页": "⇪",
    "数据分析页": "◌",
    "实验训练页": "↗",
    "结果分析页": "◈",
}


def default_clients(count: int = 4) -> list[dict[str, Any]]:
    return [
        {
            "id": f"client-{index + 1}",
            "name": f"客户端 {index + 1}",
            "enabled": True,
            "status": "待校验",
            "rows": 0,
            "features": 0,
        }
        for index in range(count)
    ]


def default_experiment_config() -> dict[str, Any]:
    return {
        "samples": 240,
        "features": 6,
        "clients": 4,
        "target_column": "target",
        "hidden_layers": 2,
        "hidden_units": "64,32",
        "activation": "ReLU",
        "epochs": 20,
        "rounds": 20,
        "local_epochs": 1,
        "batch_size": 32,
        "lr": 0.001,
        "data_mode": "Non-IID",
        "dirichlet_alpha": 0.5,
        "client_fraction": 1.0,
        "aggregation": "FedAvg",
        "clip_norm": 1.0,
        "noise_multiplier": 1.0,
        "epsilon": 4.0,
        "delta": 1e-5,
        "missing_strategy": "drop",
        "scaler": "standard",
        "seed": 42,
    }


def initialize_state() -> None:
    auth_db.init_db()
    if "clients" not in st.session_state:
        st.session_state.clients = default_clients()
    if "frame" not in st.session_state:
        st.session_state.frame = generate_sample_data()
    if "validation" not in st.session_state:
        st.session_state.validation = {"status": "待校验", "message": "尚未执行数据校验", "details": {}}
    if "experiment_config" not in st.session_state:
        st.session_state.experiment_config = default_experiment_config()
    if "training_results" not in st.session_state:
        st.session_state.training_results = {}
    if "preprocess_versions" not in st.session_state:
        st.session_state.preprocess_versions = []
    if "report_markdown" not in st.session_state:
        st.session_state.report_markdown = ""
    if "auth_session_id" not in st.session_state:
        st.session_state.auth_session_id = None
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None
    session = auth_db.get_session(st.session_state.auth_session_id)
    if session:
        st.session_state.auth_user = {
            "username": session["username"],
            "role": session["role"],
            "session_id": session["session_id"],
        }
    else:
        st.session_state.auth_session_id = None
        st.session_state.auth_user = None


def validation_status(frame: pd.DataFrame | None, target_column: str) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {"status": "待校验", "message": "尚未上传或生成数据", "details": {}}
    result = validate_tabular_data(frame, target_column=target_column)
    return {
        "status": "通过" if result.valid else "失败",
        "message": result.message,
        "details": result.details,
    }


def sync_clients_with_frame(
    clients: list[dict[str, Any]],
    frame: pd.DataFrame | None,
    status: str,
    target_column: str = "target",
) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return [{**client, "status": status, "rows": 0, "features": 0} for client in clients]

    feature_count = len([column for column in frame.columns if column not in {target_column, "client_id"}])
    if "client_id" in frame.columns:
        counts = frame["client_id"].value_counts().to_dict()
    else:
        counts = {}

    synced = []
    for index, client in enumerate(clients):
        rows = int(counts.get(index, counts.get(str(index), counts.get(client["id"], counts.get(index + 1, 0)))))
        synced.append({**client, "status": status, "rows": rows, "features": feature_count})
    return synced


def client_distribution_frame(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    if frame is None or frame.empty or "client_id" not in frame.columns or target_column not in frame.columns:
        return pd.DataFrame(columns=["client_id", "label", "samples"])
    grouped = frame.groupby(["client_id", target_column]).size().reset_index(name="samples")
    return grouped.rename(columns={target_column: "label"})


def missing_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["column", "missing", "missing_rate"])
    summary = frame.isna().sum().reset_index()
    summary.columns = ["column", "missing"]
    summary["missing_rate"] = summary["missing"] / max(len(frame), 1)
    return summary[summary["missing"] > 0].sort_values("missing", ascending=False)


def preprocess_scope() -> str:
    return "centralized" if is_manager() else "federated"


def preprocess_scope_label(scope: str) -> str:
    return "管理员集中式 MLP 数据" if scope == "centralized" else "客户端分布式 MLP 数据"


def add_preprocess_version(frame: pd.DataFrame, target_column: str, scope: str, config: dict[str, Any], columns_to_scale: list[str]) -> dict[str, Any]:
    versions = st.session_state.preprocess_versions
    version_id = f"v{len(versions) + 1}"
    version = {
        "id": version_id,
        "name": f"{preprocess_scope_label(scope)} {version_id}",
        "scope": scope,
        "owner": current_user()["username"] if current_user() else "system",
        "target_column": target_column,
        "rows": len(frame),
        "features": len([column for column in frame.columns if column not in {target_column, "client_id"}]),
        "missing_strategy": config.get("missing_strategy"),
        "scaler": config.get("scaler"),
        "scaled_columns": columns_to_scale,
        "frame": frame.copy(),
    }
    versions.append(version)
    return version


def version_options(scope: str | None = None) -> list[dict[str, Any]]:
    versions = st.session_state.get("preprocess_versions", [])
    if scope is None:
        return versions
    return [version for version in versions if version.get("scope") == scope]


def version_label(version: dict[str, Any]) -> str:
    return f"{version['id']} · {preprocess_scope_label(version['scope'])} · {version['rows']}行 · 标签:{version['target_column']}"


def preprocess_versions_table() -> pd.DataFrame:
    rows = []
    for version in st.session_state.get("preprocess_versions", []):
        rows.append({
            "版本": version["id"],
            "用途": preprocess_scope_label(version["scope"]),
            "创建人": version.get("owner", "-"),
            "标签列": version.get("target_column", "-"),
            "样本数": version.get("rows", 0),
            "特征数": version.get("features", 0),
            "缺失值处理": version.get("missing_strategy", "-"),
            "标准化": version.get("scaler", "-"),
            "标准化列": ", ".join(version.get("scaled_columns", [])) or "-",
        })
    return pd.DataFrame(rows)


def results_table(results: dict[str, dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for mode, result in results.items():
        metrics = result.get("metrics", {})
        rows.append(
            {
                "方案": SCHEME_LABELS.get(mode, mode),
                "数据方式": "全量集中训练" if mode == "centralized" else f"多客户端 {config['data_mode']} 数据",
                "是否联邦": "否" if mode == "centralized" else "是",
                "是否差分隐私": "是" if mode == "dp_fedavg" else "否",
                "epsilon": result.get("dp", {}).get("epsilon", config["epsilon"]) if mode == "dp_fedavg" else "-",
                "delta": result.get("dp", {}).get("delta", config["delta"]) if mode == "dp_fedavg" else "-",
                "clip_norm": result.get("dp", {}).get("clip_norm", config["clip_norm"]) if mode == "dp_fedavg" else "-",
                "noise_multiplier": result.get("dp", {}).get("noise_multiplier", config["noise_multiplier"]) if mode == "dp_fedavg" else "-",
                "Accuracy": metrics.get("accuracy"),
                "Precision": metrics.get("precision"),
                "Recall": metrics.get("recall"),
                "F1-score": metrics.get("f1"),
                "AUC": metrics.get("auc"),
                "Final Loss": (result.get("history", {}).get("loss") or [None])[-1],
                "Final Accuracy": (result.get("history", {}).get("accuracy") or [None])[-1],
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = ["-" if pd.isna(row[column]) else str(row[column]) for column in frame.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def generate_report(results: dict[str, dict[str, Any]], config: dict[str, Any]) -> str:
    if not results:
        return "# FedPrivTab 实验报告\n\n尚未生成训练结果。请先完成 CSV 数据校验，并在训练监控页启动训练。"

    dp_table = privacy_performance_table(results, config)
    lines = [
        "# FedPrivTab 实验报告",
        "",
        "## 实验配置",
        "",
        f"- 数据模式: {config['data_mode']}",
        f"- 客户端数量: {config['clients']}",
        f"- MLP: {config['hidden_layers']} 层 / {config['hidden_units']} 隐藏单元 / {config['activation']}",
        f"- 联邦聚合: {config['aggregation']}",
        f"- DP 参数: C={config['clip_norm']}, sigma={config['noise_multiplier']}, epsilon={config['epsilon']}, delta={config['delta']}",
        "",
        "## 方法说明",
        "",
        "- MLP: 表格特征输入多层感知机，通过隐藏层非线性映射输出二分类 logit。",
        "- FedAvg: 客户端本地训练后上传模型更新，服务端按样本量加权平均。",
        "- DP-FedAvg: 对客户端更新执行 L2 裁剪并加入高斯噪声，以 $(\\epsilon, \\delta)$-DP 描述隐私预算。",
        "- 差分隐私公式: $\\bar{g}_k = g_k \\cdot \\min(1, C / \\|g_k\\|_2)$，$\\tilde{g} = \\frac{1}{K}\\sum_k \\bar{g}_k + \\mathcal{N}(0, \\sigma^2 C^2 I)$。",
        "",
        "## 方案结果",
        "",
        dataframe_to_markdown(results_table(results, config)),
        "",
        "## 隐私-性能对比",
        "",
        dataframe_to_markdown(dp_table) if not dp_table.empty else "暂无 DP-FedAvg 结果。",
        "",
    ]
    for result in results.values():
        lines.extend([build_markdown_report(result), ""])
    return "\n".join(lines)


def history_frame(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for mode, result in results.items():
        history = result.get("history", {})
        losses = history.get("loss") or []
        accuracies = history.get("accuracy") or []
        for step in range(max(len(losses), len(accuracies))):
            rows.append(
                {
                    "方案": SCHEME_LABELS.get(mode, mode),
                    "轮次": step + 1,
                    "Loss": losses[step] if step < len(losses) else None,
                    "Accuracy": accuracies[step] if step < len(accuracies) else None,
                }
            )
    return pd.DataFrame(rows)


def privacy_performance_table(results: dict[str, dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for mode, result in results.items():
        if mode != "dp_fedavg" or not result.get("dp"):
            continue
        dp = result["dp"]
        metrics = result.get("metrics", {})
        rows.append(
            {
                "方案": SCHEME_LABELS.get(mode, mode),
                "epsilon": dp.get("epsilon", config["epsilon"]),
                "delta": dp.get("delta", config["delta"]),
                "clip_norm": dp.get("clip_norm", config["clip_norm"]),
                "noise_multiplier": dp.get("noise_multiplier", config["noise_multiplier"]),
                "Accuracy": metrics.get("accuracy"),
                "F1-score": metrics.get("f1"),
            }
        )
    return pd.DataFrame(rows)


def train_scheme(frame: pd.DataFrame, mode: str, config: dict[str, Any]) -> dict[str, Any]:
    x_train, x_test, y_train, y_test, _ = train_test_data(
        frame,
        target_column=config["target_column"],
        seed=int(config["seed"]),
    )
    train_config = TrainConfig(
        mode=mode,
        epochs=int(config["epochs"]),
        rounds=int(config["rounds"]),
        clients=int(config["clients"]),
        local_epochs=int(config["local_epochs"]),
        batch_size=int(config["batch_size"]),
        lr=float(config["lr"]),
        hidden_layers=int(config["hidden_layers"]),
        hidden_units=parse_hidden_units(config["hidden_units"], int(config["hidden_layers"])),
        activation=config["activation"],
        client_fraction=float(config["client_fraction"]),
        dirichlet_alpha=float(config["dirichlet_alpha"]),
        clip_norm=float(config["clip_norm"]),
        noise_multiplier=float(config["noise_multiplier"]),
        epsilon=float(config["epsilon"]),
        delta=float(config["delta"]),
        non_iid=config["data_mode"] == "Non-IID",
        seed=int(config["seed"]),
    )
    result = train_model(x_train, y_train, x_test, y_test, train_config)
    result["rows"] = len(frame)
    return result


def numeric_frame(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    excluded = {target_column, "client_id"}
    columns = [column for column in frame.select_dtypes(include=np.number).columns if column not in excluded]
    return frame[columns]


def inject_global_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --fedprivtab-bg: #f6f8fb;
            --fedprivtab-panel: #ffffff;
            --fedprivtab-ink: #172033;
            --fedprivtab-muted: #667085;
            --fedprivtab-line: #e5eaf2;
            --fedprivtab-primary: #2563eb;
            --fedprivtab-primary-dark: #1d4ed8;
            --fedprivtab-accent: #14b8a6;
            --fedprivtab-soft: #eef5ff;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.10), transparent 32rem),
                linear-gradient(180deg, #f8fbff 0%, var(--fedprivtab-bg) 36%, #ffffff 100%);
            color: var(--fedprivtab-ink);
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--fedprivtab-line);
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 1.25rem;
        }

        .fed-brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.9rem;
        }

        .fed-brand-mark {
            width: 2.35rem;
            height: 2.35rem;
            border-radius: 0.8rem;
            display: grid;
            place-items: center;
            color: #ffffff;
            font-weight: 800;
            letter-spacing: 0;
            background: linear-gradient(135deg, var(--fedprivtab-primary), var(--fedprivtab-accent));
            box-shadow: 0 12px 30px rgba(37, 99, 235, 0.22);
        }

        .fed-brand-title {
            font-weight: 800;
            color: var(--fedprivtab-ink);
            line-height: 1.05;
            font-size: 1.05rem;
        }

        .fed-brand-subtitle {
            color: var(--fedprivtab-muted);
            font-size: 0.78rem;
            margin-top: 0.15rem;
        }

        .fed-login-hero {
            margin-top: 1.3rem;
            padding: clamp(1.1rem, 2vw, 1.8rem);
            border: 1px solid rgba(37, 99, 235, 0.12);
            border-radius: 1.4rem;
            background:
                linear-gradient(135deg, rgba(37, 99, 235, 0.09), rgba(20, 184, 166, 0.10)),
                #ffffff;
            box-shadow: 0 24px 70px rgba(15, 23, 42, 0.09);
            min-height: 27rem;
        }

        .fed-login-kicker {
            color: var(--fedprivtab-primary-dark);
            font-weight: 700;
            font-size: 0.88rem;
            margin-bottom: 0.85rem;
        }

        .fed-login-title {
            color: var(--fedprivtab-ink);
            font-size: clamp(2rem, 4vw, 4rem);
            line-height: 1.02;
            font-weight: 850;
            letter-spacing: 0;
            margin-bottom: 1rem;
            max-width: 12ch;
        }

        .fed-login-copy {
            color: #475467;
            font-size: 1.04rem;
            line-height: 1.7;
            max-width: 42rem;
            margin-bottom: 1.25rem;
        }

        .fed-login-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
            margin: 1.1rem 0 0.35rem;
        }

        .fed-login-pill {
            padding: 0.55rem 0.75rem;
            border: 1px solid rgba(37, 99, 235, 0.14);
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.78);
            color: #344054;
            font-size: 0.9rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        }

        .fed-auth-card {
            margin-top: 1.3rem;
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid rgba(226, 232, 240, 0.95);
            border-radius: 1.05rem 1.05rem 0 0;
            border-bottom: 0;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
            padding: 1.25rem 1.25rem 0.25rem;
        }

        .fed-auth-heading {
            font-weight: 800;
            font-size: 1.28rem;
            color: var(--fedprivtab-ink);
            margin-bottom: 0.25rem;
        }

        .fed-auth-hint {
            color: var(--fedprivtab-muted);
            font-size: 0.9rem;
            line-height: 1.5;
            margin-bottom: 0.75rem;
        }

        .fed-status-hint {
            margin-top: 0;
            padding: 0.85rem 1rem 1rem;
            border-radius: 0 0 1.05rem 1.05rem;
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid rgba(226, 232, 240, 0.95);
            border-top: 0;
            color: #475467;
            font-size: 0.86rem;
            line-height: 1.45;
            box-shadow: 0 22px 46px rgba(15, 23, 42, 0.10);
        }

        div[data-testid="stForm"]:has(input[aria-label="用户名"]) {
            border: 1px solid rgba(226, 232, 240, 0.95);
            border-top: 0;
            border-bottom: 0;
            border-radius: 0;
            padding: 0.2rem 1.25rem 0.4rem;
            background: rgba(255, 255, 255, 0.94);
            box-shadow: none;
        }

        div[data-testid="stTextInput"] input {
            border-radius: 0.72rem;
            border-color: #d7deea;
            min-height: 2.8rem;
        }

        div[data-testid="stTextInput"] input:focus {
            border-color: var(--fedprivtab-primary);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
        }

        .stButton > button,
        div[data-testid="stFormSubmitButton"] button {
            border-radius: 0.72rem;
            min-height: 2.7rem;
            font-weight: 700;
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.13);
        }

        .fed-top-shell {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin: 0.2rem 0 1.1rem;
        }

        .fed-user-badge {
            margin-left: auto;
            display: inline-flex;
            align-items: center;
            justify-content: flex-end;
            gap: 0.55rem;
            padding: 0.42rem 0.55rem 0.42rem 0.46rem;
            border: 1px solid var(--fedprivtab-line);
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.055);
            max-width: 24rem;
        }

        .fed-user-avatar {
            width: 1.85rem;
            height: 1.85rem;
            border-radius: 999px;
            display: grid;
            place-items: center;
            color: #ffffff;
            background: var(--fedprivtab-primary);
            font-size: 0.76rem;
            font-weight: 850;
        }

        .fed-user-name {
            color: var(--fedprivtab-ink);
            font-weight: 800;
            line-height: 1.1;
            font-size: 0.9rem;
        }

        .fed-user-role {
            color: var(--fedprivtab-muted);
            font-size: 0.74rem;
            line-height: 1.15;
        }

        .fed-section-card {
            border: 1px solid var(--fedprivtab-line);
            border-radius: 0.95rem;
            background: rgba(255, 255, 255, 0.96);
            padding: 1rem;
            box-shadow: 0 12px 34px rgba(15, 23, 42, 0.045);
            margin-bottom: 0.85rem;
        }

        .fed-section-title {
            font-size: 1rem;
            font-weight: 850;
            color: var(--fedprivtab-ink);
            margin-bottom: 0.18rem;
        }

        .fed-section-copy {
            color: var(--fedprivtab-muted);
            font-size: 0.88rem;
            line-height: 1.45;
            margin-bottom: 0.8rem;
        }

        .fed-kpi-card {
            border: 1px solid var(--fedprivtab-line);
            border-radius: 0.85rem;
            background: #ffffff;
            padding: 0.85rem;
            min-height: 5.5rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.035);
        }

        .fed-kpi-label {
            color: var(--fedprivtab-muted);
            font-size: 0.78rem;
            font-weight: 750;
            margin-bottom: 0.3rem;
        }

        .fed-kpi-value {
            color: var(--fedprivtab-ink);
            font-size: 1.45rem;
            line-height: 1.1;
            font-weight: 850;
            overflow-wrap: anywhere;
        }

        .fed-kpi-note {
            color: #7b8798;
            font-size: 0.75rem;
            line-height: 1.35;
            margin-top: 0.35rem;
        }

        .fed-tag {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            border-radius: 999px;
            padding: 0.2rem 0.48rem;
            font-size: 0.76rem;
            font-weight: 800;
            border: 1px solid #dbe7ff;
            background: var(--fedprivtab-soft);
            color: #1e40af;
            white-space: nowrap;
        }

        .fed-tag-ok {
            background: #ecfdf3;
            border-color: #bbf7d0;
            color: #166534;
        }

        .fed-tag-warn {
            background: #fff7ed;
            border-color: #fed7aa;
            color: #9a3412;
        }

        .fed-tag-muted {
            background: #f2f4f7;
            border-color: #e4e7ec;
            color: #475467;
        }

        .fed-empty-state {
            max-width: 42rem;
            margin: 2rem auto 1rem;
            text-align: center;
            border: 1px dashed #cbd5e1;
            border-radius: 1rem;
            background: rgba(255, 255, 255, 0.9);
            padding: 1.4rem;
        }

        .fed-empty-icon {
            width: 2.6rem;
            height: 2.6rem;
            margin: 0 auto 0.8rem;
            border-radius: 0.85rem;
            display: grid;
            place-items: center;
            color: #1d4ed8;
            background: var(--fedprivtab-soft);
            font-weight: 900;
        }

        .fed-empty-title {
            color: var(--fedprivtab-ink);
            font-weight: 850;
            font-size: 1.08rem;
            margin-bottom: 0.25rem;
        }

        .fed-empty-copy {
            color: var(--fedprivtab-muted);
            line-height: 1.55;
            font-size: 0.9rem;
        }

        .fed-mini-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0 0.5rem;
        }

        .fed-mini-table th {
            color: #667085;
            font-size: 0.76rem;
            text-align: left;
            font-weight: 850;
            padding: 0 0.65rem;
        }

        .fed-mini-table td {
            background: #ffffff;
            border-top: 1px solid var(--fedprivtab-line);
            border-bottom: 1px solid var(--fedprivtab-line);
            padding: 0.68rem 0.65rem;
            color: #344054;
            font-size: 0.9rem;
        }

        .fed-mini-table td:first-child {
            border-left: 1px solid var(--fedprivtab-line);
            border-radius: 0.7rem 0 0 0.7rem;
            font-weight: 800;
            color: var(--fedprivtab-ink);
        }

        .fed-mini-table td:last-child {
            border-right: 1px solid var(--fedprivtab-line);
            border-radius: 0 0.7rem 0.7rem 0;
        }

        .fed-sidebar-section {
            color: #98a2b3;
            font-size: 0.73rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
            margin: 0.9rem 0 0.45rem;
        }

        .fed-role-card {
            border: 1px solid var(--fedprivtab-line);
            border-radius: 0.9rem;
            background: linear-gradient(180deg, #ffffff, #f8fbff);
            padding: 0.85rem;
            color: #475467;
            font-size: 0.86rem;
            line-height: 1.45;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
            margin-bottom: 0.95rem;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 0.42rem;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            position: relative;
            width: 100%;
            padding: 0.72rem 0.78rem 0.72rem 0.92rem;
            border: 1px solid transparent;
            border-radius: 0.75rem;
            background: transparent;
            transition: all 160ms ease;
            color: #344054;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: #f7faff;
            border-color: #dbe7ff;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: var(--fedprivtab-soft);
            border-color: #bfdbfe;
            box-shadow: 0 10px 26px rgba(37, 99, 235, 0.10);
            color: #1e3a8a;
            font-weight: 800;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked)::before {
            content: "";
            position: absolute;
            left: 0.35rem;
            top: 0.62rem;
            bottom: 0.62rem;
            width: 0.22rem;
            border-radius: 999px;
            background: linear-gradient(180deg, var(--fedprivtab-primary), var(--fedprivtab-accent));
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
            display: none;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label p {
            font-size: 0.93rem;
            line-height: 1.35;
            white-space: normal;
        }

        .fed-sidebar-metrics {
            display: grid;
            gap: 0.55rem;
            margin-top: 0.45rem;
        }

        .fed-sidebar-metric {
            display: grid;
            grid-template-columns: 2rem 1fr;
            gap: 0.65rem;
            align-items: center;
            border: 1px solid var(--fedprivtab-line);
            border-radius: 0.85rem;
            background: #ffffff;
            padding: 0.72rem;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.035);
        }

        .fed-sidebar-metric-icon {
            width: 2rem;
            height: 2rem;
            border-radius: 0.65rem;
            display: grid;
            place-items: center;
            color: var(--fedprivtab-primary-dark);
            background: var(--fedprivtab-soft);
            font-weight: 800;
        }

        .fed-sidebar-metric-label {
            color: var(--fedprivtab-muted);
            font-size: 0.74rem;
            line-height: 1.2;
        }

        .fed-sidebar-metric-value {
            color: var(--fedprivtab-ink);
            font-weight: 800;
            font-size: 1rem;
            line-height: 1.25;
            word-break: break-word;
        }

        @media (max-width: 760px) {
            .fed-login-title {
                max-width: none;
            }

            .fed-login-hero {
                border-radius: 1rem;
                min-height: auto;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_label(page: str) -> str:
    return f"{PAGE_ICONS.get(page, '•')}  {page}"


def status_tag(status: Any) -> str:
    text = escape(str(status))
    status_text = str(status)
    css_class = "fed-tag-muted"
    if status_text in {"通过", "已审核", "已参与训练", "启用"}:
        css_class = "fed-tag-ok"
    elif status_text in {"失败", "禁用", "待校验"}:
        css_class = "fed-tag-warn"
    return f'<span class="fed-tag {css_class}">{text}</span>'


def kpi_card(label: str, value: Any, note: str = "") -> str:
    note_html = f'<div class="fed-kpi-note">{escape(str(note))}</div>' if note else ""
    return f"""
    <div class="fed-kpi-card">
        <div class="fed-kpi-label">{escape(str(label))}</div>
        <div class="fed-kpi-value">{escape(str(value))}</div>
        {note_html}
    </div>
    """


def section_card(title: str, copy: str = "") -> None:
    copy_html = f'<div class="fed-section-copy">{escape(copy)}</div>' if copy else ""
    st.markdown(
        f"""
        <div class="fed-section-card">
            <div class="fed-section-title">{escape(title)}</div>
            {copy_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(title: str, copy: str, icon: str = "i") -> None:
    st.markdown(
        f"""
        <div class="fed-empty-state">
            <div class="fed-empty-icon">{escape(icon)}</div>
            <div class="fed-empty-title">{escape(title)}</div>
            <div class="fed-empty-copy">{escape(copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_row(cards: list[tuple[str, Any, str]]) -> None:
    columns = st.columns(len(cards))
    for column, (label, value, note) in zip(columns, cards):
        column.markdown(kpi_card(label, value, note), unsafe_allow_html=True)


def client_table_html(clients: list[dict[str, Any]]) -> str:
    rows = []
    for client in clients:
        enabled = "启用" if client.get("enabled") else "禁用"
        rows.append(
            "<tr>"
            f"<td>{escape(str(client.get('id', '')))}</td>"
            f"<td>{escape(str(client.get('name', '')))}</td>"
            f"<td>{status_tag(enabled)}</td>"
            f"<td>{status_tag(client.get('status', ''))}</td>"
            f"<td>{escape(str(client.get('rows', 0)))}</td>"
            f"<td>{escape(str(client.get('features', 0)))}</td>"
            "</tr>"
        )
    return (
        '<table class="fed-mini-table"><thead><tr>'
        "<th>ID</th><th>名称</th><th>参与</th><th>状态</th><th>样本</th><th>特征</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def render_brand(compact: bool = False) -> None:
    subtitle = "隐私保护表格联邦学习" if compact else "Federated Privacy for Tabular ML"
    st.markdown(
        f"""
        <div class="fed-brand">
            <div class="fed-brand-mark">FP</div>
            <div>
                <div class="fed-brand-title">FedPrivTab</div>
                <div class="fed-brand-subtitle">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_metric_card(label: str, value: Any, icon: str) -> str:
    return (
        '<div class="fed-sidebar-metric">'
        f'<div class="fed-sidebar-metric-icon">{escape(str(icon))}</div>'
        '<div>'
        f'<div class="fed-sidebar-metric-label">{escape(str(label))}</div>'
        f'<div class="fed-sidebar-metric-value">{escape(str(value))}</div>'
        '</div>'
        '</div>'
    )


def current_user() -> dict[str, Any] | None:
    user = st.session_state.get("auth_user")
    return dict(user) if user else None


def is_manager() -> bool:
    user = current_user()
    return bool(user and user.get("role") in {"系统管理员", "实验研究人员"})


def visible_frame_for_user(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    user = current_user()
    if frame is None or frame.empty or is_manager() or user is None or "client_id" not in frame.columns:
        return frame
    username = str(user.get("username", ""))
    client_ids = {"0", "client"}
    if username.startswith("client-"):
        try:
            client_ids.add(str(max(0, int(username.rsplit("-", 1)[-1]) - 1)))
        except ValueError:
            pass
    return frame[frame["client_id"].astype(str).isin(client_ids)]


def allowed_pages(role: str) -> list[str]:
    return ROLE_PAGES.get(role, ["首页"])


def render_top_bar() -> None:
    user = current_user()
    if user:
        left, right = st.columns([3.4, 1.2])
        with left:
            render_brand()
        with right:
            initials = "".join(part[:1] for part in str(user["username"]).split())[:2].upper() or "U"
            st.markdown(
                f"""
                <div class="fed-user-badge">
                    <div class="fed-user-avatar">{escape(initials)}</div>
                    <div>
                        <div class="fed-user-name">{escape(str(user["username"]))}</div>
                        <div class="fed-user-role">{escape(str(user["role"]))}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if hasattr(st, "popover"):
                with st.popover("账户", use_container_width=True):
                    st.caption(f"已登录为 {user['username']}")
                    st.caption(user["role"])
                    logout_clicked = st.button("退出登录", type="secondary", use_container_width=True)
            else:
                logout_clicked = st.button("退出登录", type="secondary", use_container_width=True)
            if logout_clicked:
                auth_db.logout(user["session_id"])
                st.session_state.auth_session_id = None
                st.session_state.auth_user = None
                st.rerun()
        return

    render_brand()
    hero, login = st.columns([1.45, 0.9], gap="large")
    with hero:
        st.markdown(
            """
            <div class="fed-login-hero">
                <div class="fed-login-kicker">Secure collaborative tabular learning</div>
                <div class="fed-login-title">FedPrivTab 实验控制台</div>
                <div class="fed-login-copy">
                    面向集中式 MLP、FedAvg 与 DP-FedAvg 的轻量实验工作台，统一管理客户端、
                    数据审核、训练监控和结果报告。
                </div>
                <div class="fed-login-pills">
                    <span class="fed-login-pill">Client governance</span>
                    <span class="fed-login-pill">Non-IID analysis</span>
                    <span class="fed-login-pill">Differential privacy</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with login:
        st.markdown(
            """
            <div class="fed-auth-card">
                <div class="fed-auth-heading">登录工作台</div>
                <div class="fed-auth-hint">使用演示账号进入对应角色页面，登录状态会在本机 SQLite 会话中保留。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("login-form"):
            username = st.text_input("用户名")
            password = st.text_input("密码", type="password")
            submitted = st.form_submit_button("登录", use_container_width=True)
        if submitted:
            session = auth_db.login(username, password)
            if session:
                st.session_state.auth_session_id = session["session_id"]
                st.session_state.auth_user = {
                    "username": session["username"],
                    "role": session["role"],
                    "session_id": session["session_id"],
                }
                st.rerun()
            else:
                st.error("用户名或密码错误")
        st.markdown(
            """
            <div class="fed-status-hint">状态提示: 未登录。默认账号见 README，系统会按角色开放页面。</div>
            """,
            unsafe_allow_html=True,
        )


def render_login_required() -> None:
    return None


def render_sidebar() -> str:
    user = current_user()
    role = user["role"] if user else ""
    pages = allowed_pages(role)
    with st.sidebar:
        render_brand(compact=True)
        st.markdown(f'<div class="fed-role-card">{escape(ROLE_HINTS.get(role, "请先登录。"))}</div>', unsafe_allow_html=True)
        st.markdown('<div class="fed-sidebar-section">Navigation</div>', unsafe_allow_html=True)
        page = st.radio("页面", pages, format_func=page_label, label_visibility="collapsed")
        st.markdown('<div class="fed-sidebar-section">Workspace Status</div>', unsafe_allow_html=True)
        status = st.session_state.validation["status"]
        metrics = "\n".join(
            [
                sidebar_metric_card("客户端", len(st.session_state.clients), "▦"),
                sidebar_metric_card("训练结果", len(st.session_state.training_results), "↗"),
            ]
        )
        st.markdown(f'<div class="fed-sidebar-metrics">{metrics}</div>', unsafe_allow_html=True)
    return page


def render_home() -> None:
    st.subheader("首页")
    st.caption("差分隐私 Non-IID 表格数据联邦学习实验系统")
    clients = pd.DataFrame(st.session_state.clients)
    enabled = int(clients["enabled"].sum()) if not clients.empty else 0
    passed = st.session_state.validation["status"] == "通过"

    total_rows = len(st.session_state.frame) if st.session_state.frame is not None else 0
    metric_cols = st.columns(5)
    metric_cols[0].metric("客户端总数", len(clients))
    metric_cols[1].metric("启用客户端", enabled)
    metric_cols[2].metric("数据校验", st.session_state.validation["status"])
    metric_cols[3].metric("样本数", total_rows)
    metric_cols[4].metric("已完成方案", len(st.session_state.training_results))

    st.subheader("实验概览")
    st.write(
        "系统围绕集中式 MLP、FedAvg + MLP、DP-FedAvg + MLP 三类方案，完成客户端数据管理、"
        "表格数据审核、Non-IID 配置、训练监控、结果分析和 Markdown 报告导出。"
    )
    if not passed:
        st.warning("当前数据尚未通过校验。请在数据预处理页生成可训练版本后再启动训练。")
    if st.session_state.training_results:
        st.subheader("最近结果")
        st.dataframe(results_table(st.session_state.training_results, st.session_state.experiment_config), use_container_width=True)
    st.dataframe(clients, use_container_width=True)


def render_client_management() -> None:
    st.title("客户端管理页")
    user = current_user()
    if not is_manager():
        st.error("权限不足：客户端用户不能管理其他客户端账号。")
        return

    users = [item for item in auth_db.list_users(role=auth_db.CLIENT_ROLE) if item["username"] in {"client-1", "client-2", "client-3", "client-4"}]
    st.caption("系统固定 4 个客户端账号，不支持新增或删除客户端；本页仅支持修改/重置客户端密码。")
    metric_row(
        [
            ("客户端账号", len(users), "固定 4 个客户端"),
            ("本地客户端", len(st.session_state.clients), "实验客户端清单"),
            ("训练结果", len(st.session_state.training_results), "已完成方案"),
        ]
    )

    st.markdown("#### 客户端账号清单")
    rows = [{"用户名": item["username"], "角色": item["role"], "创建时间": item["created_at"], "最近登录": item.get("last_login_at") or "-"} for item in users]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### 修改客户端密码")
    for item in users:
        with st.expander(f"修改密码 · {item['username']}", expanded=False):
            new_password = st.text_input("新密码", type="password", key=f"reset-{item['username']}", placeholder="输入该客户端的新密码")
            if st.button("保存新密码", key=f"save-pass-{item['username']}"):
                try:
                    if auth_db.change_password(item["username"], None, new_password):
                        st.success("密码已更新")
                    else:
                        st.error("账号不存在")
                except ValueError as exc:
                    st.error(str(exc))

    with st.container(border=True):
        st.markdown("#### 修改我的密码")
        with st.form("change-own-password", clear_on_submit=True):
            old_password = st.text_input("当前密码", type="password")
            new_password = st.text_input("新密码", type="password")
            submitted = st.form_submit_button("更新我的密码")
        if submitted and user:
            try:
                if auth_db.change_password(user["username"], old_password, new_password):
                    st.success("你的密码已更新")
                else:
                    st.error("当前密码错误")
            except ValueError as exc:
                st.error(str(exc))


def render_data_upload() -> None:
    st.title("数据预处理页")
    config = st.session_state.experiment_config
    scope = preprocess_scope()
    scope_label = preprocess_scope_label(scope)
    frame = st.session_state.frame if is_manager() else visible_frame_for_user(st.session_state.frame)

    st.caption(f"当前角色的数据处理用途：{scope_label}。一键处理后会保存为训练可选的数据版本。")
    metric_row(
        [
            ("处理用途", scope_label, "由当前登录角色决定"),
            ("原始样本", len(frame) if frame is not None else 0, "上传或示例数据"),
            ("版本数", len(version_options(scope)), "当前用途可用版本"),
        ]
    )

    with st.container(border=True):
        st.markdown("#### 1. 文件上传")
        uploaded = st.file_uploader("上传 CSV 数据", type=["csv"])
        if uploaded is not None:
            st.session_state.frame = pd.read_csv(uploaded)
            st.session_state.validation = {"status": "待处理", "message": "文件已上传，请选择目标变量并一键处理", "details": {}}
            st.rerun()
        with st.expander("没有 CSV？生成示例数据", expanded=False):
            samples = st.number_input("示例样本数", min_value=50, max_value=5000, value=int(config["samples"]), step=10)
            features = st.number_input("示例特征数", min_value=2, max_value=50, value=int(config["features"]), step=1)
            clients = st.number_input("示例客户端数", min_value=2, max_value=20, value=int(config["clients"]), step=1)
            if st.button("生成示例数据", use_container_width=True):
                config.update({"samples": int(samples), "features": int(features), "clients": int(clients)})
                st.session_state.frame = generate_sample_data(samples=int(samples), features=int(features), clients=int(clients), seed=int(config["seed"]))
                st.session_state.clients = default_clients(int(clients))
                st.session_state.validation = {"status": "待处理", "message": "示例数据已生成，请一键处理", "details": {}}
                st.rerun()

    frame = st.session_state.frame if is_manager() else visible_frame_for_user(st.session_state.frame)
    if frame is None or frame.empty:
        empty_state("等待数据", "上传 CSV 或生成示例数据后即可进行预处理。", "⇪")
        return

    columns = list(frame.columns)
    numeric_columns = [column for column in frame.select_dtypes(include=np.number).columns if column != "client_id"]
    target_default = columns.index(config["target_column"]) if config["target_column"] in columns else 0

    config_col, summary_col = st.columns([1.1, 1.3])
    with config_col:
        with st.container(border=True):
            st.markdown("#### 2. 目标变量选择")
            config["target_column"] = st.selectbox("目标变量", columns, index=target_default)
            st.markdown("#### 3. 缺失值摘要和处理方式")
            missing = missing_summary(frame)
            if missing.empty:
                st.success("当前数据没有缺失值")
            else:
                st.dataframe(missing, use_container_width=True, hide_index=True)
            config["missing_strategy"] = st.selectbox(
                "缺失值处理方式",
                ["drop", "mean", "median", "mode"],
                index=["drop", "mean", "median", "mode"].index(config.get("missing_strategy", "drop")),
                format_func={"drop": "删除缺失行", "mean": "均值填充", "median": "中位数填充", "mode": "众数填充"}.get,
            )
            st.markdown("#### 4. 数值标准化")
            scale_candidates = [column for column in numeric_columns if column != config["target_column"]]
            columns_to_scale = st.multiselect("选择要标准化的数值列", scale_candidates, default=scale_candidates)
            config["scaler"] = st.selectbox(
                "标准化方式",
                ["standard", "minmax", "none"],
                index=["standard", "minmax", "none"].index(config.get("scaler", "standard")),
                format_func={"standard": "StandardScaler", "minmax": "MinMaxScaler", "none": "不标准化"}.get,
            )
            if st.button("一键处理并保存版本", type="primary", use_container_width=True):
                processed = preprocess_tabular_data(frame, target_column=config["target_column"], missing_strategy=config["missing_strategy"], scaler="none")
                if config["scaler"] != "none" and columns_to_scale:
                    scaler_cls = StandardScaler if config["scaler"] == "standard" else MinMaxScaler
                    existing = [column for column in columns_to_scale if column in processed.columns and pd.api.types.is_numeric_dtype(processed[column])]
                    if existing:
                        processed[existing] = scaler_cls().fit_transform(processed[existing])
                validation = validation_status(processed, config["target_column"])
                st.session_state.frame = processed
                st.session_state.validation = validation
                if validation["status"] == "通过":
                    version = add_preprocess_version(processed, config["target_column"], scope, config, columns_to_scale)
                    st.success(f"已保存处理版本：{version['id']}")
                else:
                    st.error(validation["message"])
                st.session_state.clients = sync_clients_with_frame(st.session_state.clients, processed, validation["status"], config["target_column"])
                st.rerun()

    with summary_col:
        with st.container(border=True):
            st.markdown("#### 数据预览")
            st.caption("预览前 50 行；保存版本后，实验训练页可勾选该版本参与训练。")
            st.dataframe(frame.head(50), use_container_width=True)
        versions = preprocess_versions_table()
        with st.container(border=True):
            st.markdown("#### 处理版本记录")
            if versions.empty:
                st.info("暂无预处理版本。")
            else:
                st.dataframe(versions, use_container_width=True, hide_index=True)


def render_data_analysis() -> None:
    st.title("数据分析页")
    frame = visible_frame_for_user(st.session_state.frame)
    target = st.session_state.experiment_config["target_column"]
    if frame is None or frame.empty:
        empty_state("暂无可分析数据", "请先上传或生成数据，并完成必要的预处理。", "◌")
        return

    numeric = numeric_frame(frame, target)
    missing = int(frame.isna().sum().sum())
    metric_row(
        [
            ("样本", len(frame), "当前数据行数"),
            ("字段", len(frame.columns), "包含标签与客户端列"),
            ("数值特征", len(numeric.columns), "用于建模分析"),
            ("缺失值", missing, "全表缺失单元格"),
        ]
    )

    with st.container(border=True):
        st.markdown("#### 统计摘要")
        st.caption("按字段查看基础统计，便于快速识别异常范围、类别列和缺失风险。")
        st.dataframe(frame.describe(include="all").transpose(), use_container_width=True)

    distribution = client_distribution_frame(frame, target)
    chart_cols = st.columns(2)
    with chart_cols[0]:
        with st.container(border=True):
            st.markdown("#### 标签分布")
            st.caption("检查分类均衡性，避免训练指标被单一标签主导。")
            if target in frame.columns:
                st.plotly_chart(px.histogram(frame, x=target, color=target), use_container_width=True)
    with chart_cols[1]:
        with st.container(border=True):
            st.markdown("#### 特征均值")
            st.caption("比较数值特征中心位置，辅助判断标准化是否生效。")
            means = numeric.mean().reset_index()
            means.columns = ["feature", "mean"]
            st.plotly_chart(px.bar(means, x="feature", y="mean"), use_container_width=True)

    if not distribution.empty:
        with st.container(border=True):
            st.markdown("#### 客户端标签分布")
            st.caption("展示各客户端的标签构成，用于观察 IID / Non-IID 差异。")
            st.plotly_chart(px.bar(distribution, x="client_id", y="samples", color="label", barmode="group"), use_container_width=True)

    if not numeric.empty:
        detail_cols = st.columns(2)
        with detail_cols[0]:
            with st.container(border=True):
                st.markdown("#### 单特征分布")
                feature = st.selectbox("特征", numeric.columns)
                st.plotly_chart(px.histogram(frame, x=feature, color=target if target in frame.columns else None), use_container_width=True)
        with detail_cols[1]:
            with st.container(border=True):
                st.markdown("#### 相关性热力图")
                st.caption("用于发现高度相关特征，避免冗余输入影响模型解释。")
                corr = numeric.corr()
                st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r"), use_container_width=True)


def render_experiment_config() -> None:
    render_training_monitor()


def render_training_monitor() -> None:
    st.title("实验训练页")
    config = st.session_state.experiment_config
    centralized_versions = version_options("centralized")
    federated_versions = version_options("federated")

    workbench, summary = st.columns([2.2, 1])
    with workbench:
        with st.container(border=True):
            st.markdown("#### 实验参数配置")
            common_cols = st.columns(2)
            with common_cols[0]:
                config["data_mode"] = st.radio("数据模式", ["Non-IID"], index=0, horizontal=True)
                config["clients"] = st.number_input("客户端数量", 2, 20, int(config["clients"]), step=1)
                config["rounds"] = st.number_input("通信轮数", 1, 50, int(config["rounds"]), step=1)
                config["epochs"] = st.number_input("集中式 Epoch", 1, 50, int(config["epochs"]), step=1)
            with common_cols[1]:
                config["hidden_layers"] = st.slider("隐藏层数量", 1, 4, int(config["hidden_layers"]))
                config["hidden_units"] = st.text_input("隐藏层神经元数", value=str(config["hidden_units"]), help="支持 64,32 这样的多层结构")
                config["activation"] = st.selectbox("激活函数", ["ReLU", "Tanh", "LeakyReLU"], index=["ReLU", "Tanh", "LeakyReLU"].index(config["activation"]))
                config["lr"] = st.number_input("学习率", 0.0001, 1.0, float(config["lr"]), step=0.001, format="%.4f")

        with st.expander("高级联邦与差分隐私参数", expanded=True):
            adv_cols = st.columns(2)
            with adv_cols[0]:
                config["batch_size"] = st.number_input("Batch Size", 4, 128, int(config["batch_size"]), step=4)
                config["local_epochs"] = st.number_input("本地 Epoch", 1, 10, int(config["local_epochs"]), step=1)
                config["client_fraction"] = st.slider("客户端采样比例", 0.1, 1.0, float(config["client_fraction"]), step=0.1)
                config["dirichlet_alpha"] = st.slider("Dirichlet alpha", 0.05, 5.0, float(config["dirichlet_alpha"]), step=0.05)
            with adv_cols[1]:
                config["aggregation"] = st.selectbox("聚合方式", ["FedAvg"], index=0)
                config["clip_norm"] = st.number_input("裁剪阈值 C", 0.1, 10.0, float(config["clip_norm"]), step=0.1)
                config["noise_multiplier"] = st.number_input("噪声倍率 sigma", 0.0, 5.0, float(config["noise_multiplier"]), step=0.1)
                config["epsilon"] = st.number_input("隐私预算 epsilon", 0.1, 100.0, float(config["epsilon"]), step=0.1)
                config["delta"] = st.number_input("松弛参数 delta", 1e-8, 1e-2, float(config["delta"]), format="%.8f")
                config["seed"] = st.number_input("随机种子", 1, 9999, int(config["seed"]), step=1)

        with st.container(border=True):
            st.markdown("#### 训练方案与数据版本")
            modes = st.multiselect("勾选训练方案", list(SCHEME_LABELS.keys()), default=list(SCHEME_LABELS.keys()), format_func=SCHEME_LABELS.get)
            central_choice = st.selectbox(
                "集中式 MLP 数据版本（只能选择管理员预处理数据）",
                centralized_versions,
                format_func=version_label,
                index=0 if centralized_versions else None,
                placeholder="请先由管理员在数据预处理页保存版本",
            )
            federated_choice = st.selectbox(
                "FedAvg / DP-FedAvg 数据版本（只能选择客户端预处理数据）",
                federated_versions,
                format_func=version_label,
                index=0 if federated_versions else None,
                placeholder="请先由客户端在数据预处理页保存版本",
            )
            can_train = bool(modes) and all((mode != "centralized" or central_choice) and (mode == "centralized" or federated_choice) for mode in modes)
            if not can_train:
                st.info("请为勾选的训练方案选择符合角色来源限制的预处理版本。")
            if st.button("开始训练", type="primary", disabled=not can_train, use_container_width=True):
                progress = st.progress(0)
                for index, mode in enumerate(modes, start=1):
                    selected_version = central_choice if mode == "centralized" else federated_choice
                    frame = selected_version["frame"]
                    run_config = {**config, "target_column": selected_version["target_column"]}
                    with st.spinner(f"正在训练 {SCHEME_LABELS[mode]} · {selected_version['id']}"):
                        result = train_scheme(frame, mode, run_config)
                        result["data_version"] = selected_version["id"]
                        result["data_scope"] = selected_version["scope"]
                        result["data_version_label"] = version_label(selected_version)
                        st.session_state.training_results[mode] = result
                    progress.progress(index / len(modes))
                st.session_state.report_markdown = generate_report(st.session_state.training_results, config)
                st.success("训练完成")

    with summary:
        with st.container(border=True):
            st.markdown("#### 配置摘要")
            st.metric("客户端 / 轮数", f"{config['clients']} / {config['rounds']}")
            st.metric("MLP", f"{config['hidden_layers']}x{config['hidden_units']}")
            st.metric("集中式版本", len(centralized_versions))
            st.metric("分布式版本", len(federated_versions))
            with st.expander("版本记录", expanded=False):
                table = preprocess_versions_table()
                if table.empty:
                    st.info("暂无预处理版本。")
                else:
                    st.dataframe(table, use_container_width=True, hide_index=True)

    if st.session_state.training_results:
        with st.container(border=True):
            st.markdown("#### 训练结果")
            st.dataframe(results_table(st.session_state.training_results, config), use_container_width=True, hide_index=True)
        history = history_frame(st.session_state.training_results)
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.plotly_chart(px.line(history, x="轮次", y="Loss", color="方案", markers=True), use_container_width=True)
        with chart_cols[1]:
            st.plotly_chart(px.line(history, x="轮次", y="Accuracy", color="方案", markers=True), use_container_width=True)


def render_result_analysis() -> None:
    st.title("结果分析页")
    results = st.session_state.training_results
    config = st.session_state.experiment_config
    if not results:
        empty_state("暂无训练结果", "请先在实验训练页选择预处理版本并启动训练；报告导出会在结果生成后显示为本页附属功能。", "◈")
        st.info("报告导出：训练完成后可在本页生成并下载 Markdown 报告。")
        return

    table = results_table(results, config)
    first = table.iloc[0]
    metric_row(
        [
            ("Accuracy", f"{float(first['Accuracy'] or 0):.3f}", str(first["方案"])),
            ("F1-score", f"{float(first['F1-score'] or 0):.3f}", "首个可用方案"),
            ("方案数", len(results), "已完成训练"),
        ]
    )
    with st.container(border=True):
        st.markdown("#### 三方案对比表")
        st.dataframe(table, use_container_width=True, hide_index=True)
    metric_rows = table.melt(id_vars=["方案"], value_vars=["Accuracy", "Precision", "Recall", "F1-score", "AUC"], var_name="指标", value_name="数值")
    with st.container(border=True):
        st.markdown("#### 指标对比")
        st.plotly_chart(px.bar(metric_rows, x="方案", y="数值", color="指标", barmode="group"), use_container_width=True)
    privacy = privacy_performance_table(results, config)
    with st.container(border=True):
        st.markdown("#### 隐私-性能对比")
        if privacy.empty:
            st.info("暂无 DP-FedAvg 结果。训练 DP-FedAvg 后将展示 epsilon、delta、clip_norm、noise_multiplier 与 Accuracy/F1 的对照。")
        else:
            st.dataframe(privacy, use_container_width=True, hide_index=True)
            st.plotly_chart(px.bar(privacy, x="方案", y=["Accuracy", "F1-score"], barmode="group"), use_container_width=True)

    tabs = st.tabs([SCHEME_LABELS[mode] for mode in results])
    for tab, (mode, result) in zip(tabs, results.items()):
        with tab:
            st.caption(f"数据版本：{result.get('data_version_label', '-')}")
            tab_cols = st.columns(2)
            with tab_cols[0]:
                with st.container(border=True):
                    st.markdown("#### 混淆矩阵")
                    matrix = result.get("metrics", {}).get("confusion_matrix", [[0, 0], [0, 0]])
                    st.plotly_chart(ff.create_annotated_heatmap(z=matrix, colorscale="Blues", showscale=True), use_container_width=True)
            with tab_cols[1]:
                with st.container(border=True):
                    st.markdown("#### 客户端分布")
                    distribution = pd.DataFrame(result.get("client_distribution", []))
                    st.dataframe(distribution, use_container_width=True, hide_index=True)
                    if not distribution.empty:
                        st.plotly_chart(px.bar(distribution, x="client", y="size"), use_container_width=True)
            if result.get("dp"):
                with st.container(border=True):
                    st.markdown("#### DP 参数")
                    st.json({**result["dp"], "epsilon": config["epsilon"], "delta": config["delta"]})

    with st.container(border=True):
        st.markdown("#### 报告导出")
        markdown = st.session_state.report_markdown or generate_report(results, config)
        action_col, preview_col = st.columns([1, 2])
        with action_col:
            if st.button("生成 Markdown 报告", type="primary", use_container_width=True):
                st.session_state.report_markdown = generate_report(results, config)
                st.rerun()
            st.download_button("下载 Markdown 报告", markdown, file_name="fedprivtab_report.md", mime="text/markdown", use_container_width=True)
        with preview_col:
            mode = st.radio("预览模式", ["Markdown 预览", "源码"], horizontal=True, label_visibility="collapsed")
            if mode == "Markdown 预览":
                st.markdown(markdown)
            else:
                st.text_area("Markdown 报告内容", markdown, height=360)


def main() -> None:
    st.set_page_config(page_title="FedPrivTab", layout="wide")
    initialize_state()
    inject_global_styles()
    render_top_bar()
    if not current_user():
        render_login_required()
        return
    page = render_sidebar()
    if page == "首页":
        render_home()
    elif page == "客户端管理页":
        render_client_management()
    elif page == "数据预处理页":
        render_data_upload()
    elif page == "数据分析页":
        render_data_analysis()
    elif page == "实验训练页":
        render_training_monitor()
    elif page == "结果分析页":
        render_result_analysis()


if __name__ == "__main__":
    main()
