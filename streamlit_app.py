from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from app import build_markdown_report
from data_utils import generate_sample_data, train_test_data, validate_tabular_data
from training import TrainConfig, train_model


ROLE_HINTS = {
    "系统管理员": "可管理客户端、审核数据状态，并启动训练任务。",
    "客户端用户": "可上传本地 CSV 数据，查看校验结果和参与状态。",
    "实验研究人员": "可配置 MLP、联邦学习和差分隐私参数，分析实验结果。",
}

PAGES = [
    "首页",
    "客户端管理页",
    "数据上传与审核页",
    "数据分析页",
    "实验配置页",
    "训练监控页",
    "结果分析页",
    "报告导出页",
]

SCHEME_LABELS = {
    "centralized": "集中式 MLP",
    "fedavg": "FedAvg + MLP",
    "dp_fedavg": "DP-FedAvg + MLP",
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
        "hidden_units": 32,
        "activation": "ReLU",
        "epochs": 3,
        "rounds": 3,
        "local_epochs": 1,
        "batch_size": 16,
        "lr": 0.01,
        "data_mode": "Non-IID",
        "dirichlet_alpha": 0.3,
        "client_fraction": 1.0,
        "aggregation": "FedAvg",
        "clip_norm": 1.0,
        "noise_multiplier": 0.2,
        "epsilon": 4.0,
        "delta": 1e-5,
        "seed": 42,
    }


def initialize_state() -> None:
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
    if "report_markdown" not in st.session_state:
        st.session_state.report_markdown = ""


def validation_status(frame: pd.DataFrame | None, target_column: str) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {"status": "待校验", "message": "尚未上传或生成数据", "details": {}}
    result = validate_tabular_data(frame, target_column=target_column)
    return {
        "status": "通过" if result.valid else "失败",
        "message": result.message,
        "details": result.details,
    }


def sync_clients_with_frame(clients: list[dict[str, Any]], frame: pd.DataFrame | None, status: str) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return [{**client, "status": status, "rows": 0, "features": 0} for client in clients]

    feature_count = len([column for column in frame.columns if column not in {"target", "client_id"}])
    if "client_id" in frame.columns:
        counts = frame["client_id"].value_counts().to_dict()
    else:
        counts = {}

    synced = []
    for index, client in enumerate(clients):
        rows = int(counts.get(index, counts.get(client["id"], 0)))
        synced.append({**client, "status": status, "rows": rows, "features": feature_count})
    return synced


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
                "epsilon": config["epsilon"] if mode == "dp_fedavg" else "-",
                "Accuracy": metrics.get("accuracy"),
                "Precision": metrics.get("precision"),
                "Recall": metrics.get("recall"),
                "F1-score": metrics.get("f1"),
                "AUC": metrics.get("auc"),
                "Final Loss": (result.get("history", {}).get("loss") or [None])[-1],
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
        return "# FedPrivTab 实验报告\n\n尚未生成训练结果。"

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
        "## 方案结果",
        "",
        dataframe_to_markdown(results_table(results, config)),
        "",
    ]
    for result in results.values():
        lines.extend([build_markdown_report(result), ""])
    return "\n".join(lines)


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
        clip_norm=float(config["clip_norm"]),
        noise_multiplier=float(config["noise_multiplier"]),
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


def render_sidebar() -> str:
    with st.sidebar:
        st.header("FedPrivTab")
        role = st.selectbox("当前角色", list(ROLE_HINTS))
        st.info(ROLE_HINTS[role])
        page = st.radio("页面", PAGES)
        st.divider()
        status = st.session_state.validation["status"]
        st.metric("客户端", len(st.session_state.clients))
        st.metric("数据状态", status)
        st.metric("训练结果", len(st.session_state.training_results))
    return page


def render_home() -> None:
    st.title("FedPrivTab")
    st.caption("差分隐私 Non-IID 表格数据联邦学习实验系统")
    clients = pd.DataFrame(st.session_state.clients)
    enabled = int(clients["enabled"].sum()) if not clients.empty else 0
    passed = st.session_state.validation["status"] == "通过"

    metric_cols = st.columns(4)
    metric_cols[0].metric("客户端总数", len(clients))
    metric_cols[1].metric("启用客户端", enabled)
    metric_cols[2].metric("数据校验", st.session_state.validation["status"])
    metric_cols[3].metric("已完成方案", len(st.session_state.training_results))

    st.subheader("实验概览")
    st.write(
        "系统围绕集中式 MLP、FedAvg + MLP、DP-FedAvg + MLP 三类方案，完成客户端数据管理、"
        "表格数据审核、Non-IID 配置、训练监控、结果分析和 Markdown 报告导出。"
    )
    if not passed:
        st.warning("当前数据尚未通过校验。请在数据上传与审核页完成校验后再启动训练。")
    st.dataframe(clients, use_container_width=True)


def render_client_management() -> None:
    st.title("客户端管理页")
    with st.form("add-client", clear_on_submit=True):
        name = st.text_input("客户端名称")
        submitted = st.form_submit_button("新增客户端")
    if submitted and name.strip():
        next_id = f"client-{len(st.session_state.clients) + 1}"
        st.session_state.clients.append(
            {"id": next_id, "name": name.strip(), "enabled": True, "status": "待校验", "rows": 0, "features": 0}
        )
        st.rerun()

    for index, client in enumerate(st.session_state.clients):
        cols = st.columns([2, 2, 1, 1, 1])
        cols[0].write(client["id"])
        cols[1].write(client["name"])
        cols[2].write("启用" if client["enabled"] else "禁用")
        cols[3].write(client["status"])
        label = "禁用" if client["enabled"] else "启用"
        if cols[4].button(label, key=f"toggle-{client['id']}"):
            st.session_state.clients[index]["enabled"] = not client["enabled"]
            st.rerun()

    st.subheader("状态表")
    st.dataframe(pd.DataFrame(st.session_state.clients), use_container_width=True)


def render_data_upload() -> None:
    st.title("数据上传与审核页")
    config = st.session_state.experiment_config
    left, right = st.columns([1, 1])
    with left:
        uploaded = st.file_uploader("上传 CSV 数据", type=["csv"])
        if uploaded is not None:
            st.session_state.frame = pd.read_csv(uploaded)
            st.session_state.validation = {"status": "待校验", "message": "新数据已上传，等待校验", "details": {}}

        samples = st.number_input("示例样本数", min_value=50, max_value=5000, value=int(config["samples"]), step=10)
        features = st.number_input("示例特征数", min_value=2, max_value=50, value=int(config["features"]), step=1)
        clients = st.number_input("示例客户端数", min_value=2, max_value=20, value=int(config["clients"]), step=1)
        if st.button("生成示例数据"):
            config.update({"samples": int(samples), "features": int(features), "clients": int(clients)})
            st.session_state.frame = generate_sample_data(
                samples=int(samples),
                features=int(features),
                clients=int(clients),
                seed=int(config["seed"]),
            )
            st.session_state.clients = default_clients(int(clients))
            st.session_state.validation = {"status": "待校验", "message": "示例数据已生成，等待校验", "details": {}}
            st.rerun()

    frame = st.session_state.frame
    with right:
        columns = list(frame.columns) if frame is not None else []
        default_index = columns.index(config["target_column"]) if config["target_column"] in columns else 0
        if columns:
            config["target_column"] = st.selectbox("标签列", columns, index=default_index)
        if st.button("执行数据校验", type="primary"):
            st.session_state.validation = validation_status(frame, config["target_column"])
            st.session_state.clients = sync_clients_with_frame(
                st.session_state.clients,
                frame,
                st.session_state.validation["status"],
            )
            st.rerun()
        st.metric("校验状态", st.session_state.validation["status"])
        st.write(st.session_state.validation["message"])
        st.json(st.session_state.validation["details"])

    st.subheader("数据预览")
    st.dataframe(frame.head(50), use_container_width=True)


def render_data_analysis() -> None:
    st.title("数据分析页")
    frame = st.session_state.frame
    target = st.session_state.experiment_config["target_column"]
    if frame is None or frame.empty:
        st.warning("暂无数据。")
        return

    st.subheader("统计描述")
    st.dataframe(frame.describe(include="all").transpose(), use_container_width=True)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.subheader("标签分布")
        if target in frame.columns:
            st.plotly_chart(px.histogram(frame, x=target, color=target), use_container_width=True)
    with chart_cols[1]:
        st.subheader("特征均值")
        means = numeric_frame(frame, target).mean().reset_index()
        means.columns = ["feature", "mean"]
        st.plotly_chart(px.bar(means, x="feature", y="mean"), use_container_width=True)

    numeric = numeric_frame(frame, target)
    if not numeric.empty:
        feature = st.selectbox("特征分布", numeric.columns)
        st.plotly_chart(px.histogram(frame, x=feature, color=target if target in frame.columns else None), use_container_width=True)
        st.subheader("相关性热力图")
        corr = numeric.corr()
        st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r"), use_container_width=True)


def render_experiment_config() -> None:
    st.title("实验配置页")
    config = st.session_state.experiment_config
    tabs = st.tabs(["MLP", "IID / Non-IID", "FedAvg", "差分隐私"])
    with tabs[0]:
        config["hidden_layers"] = st.slider("隐藏层数量", 1, 4, int(config["hidden_layers"]))
        config["hidden_units"] = st.slider("隐藏层神经元数", 8, 128, int(config["hidden_units"]), step=8)
        config["activation"] = st.selectbox("激活函数", ["ReLU", "Tanh", "LeakyReLU"], index=["ReLU", "Tanh", "LeakyReLU"].index(config["activation"]))
        config["lr"] = st.number_input("学习率", 0.0001, 1.0, float(config["lr"]), step=0.001, format="%.4f")
        config["batch_size"] = st.number_input("Batch Size", 4, 128, int(config["batch_size"]), step=4)
        config["epochs"] = st.number_input("集中式 Epoch", 1, 50, int(config["epochs"]), step=1)
    with tabs[1]:
        config["data_mode"] = st.radio("数据模式", ["IID", "Non-IID"], index=1 if config["data_mode"] == "Non-IID" else 0, horizontal=True)
        config["dirichlet_alpha"] = st.slider("Dirichlet alpha", 0.05, 5.0, float(config["dirichlet_alpha"]), step=0.05)
        config["seed"] = st.number_input("随机种子", 1, 9999, int(config["seed"]), step=1)
    with tabs[2]:
        config["clients"] = st.number_input("客户端数量", 2, 20, int(config["clients"]), step=1)
        config["rounds"] = st.number_input("通信轮数", 1, 50, int(config["rounds"]), step=1)
        config["local_epochs"] = st.number_input("本地 Epoch", 1, 10, int(config["local_epochs"]), step=1)
        config["client_fraction"] = st.slider("客户端采样比例", 0.1, 1.0, float(config["client_fraction"]), step=0.1)
        config["aggregation"] = st.selectbox("聚合方式", ["FedAvg"], index=0)
    with tabs[3]:
        config["clip_norm"] = st.number_input("裁剪阈值 C", 0.1, 10.0, float(config["clip_norm"]), step=0.1)
        config["noise_multiplier"] = st.number_input("噪声倍率 sigma", 0.0, 5.0, float(config["noise_multiplier"]), step=0.1)
        config["epsilon"] = st.number_input("隐私预算 epsilon", 0.1, 100.0, float(config["epsilon"]), step=0.1)
        config["delta"] = st.number_input("松弛参数 delta", 1e-8, 1e-2, float(config["delta"]), format="%.8f")
    st.subheader("当前配置")
    st.json(config)


def render_training_monitor() -> None:
    st.title("训练监控页")
    frame = st.session_state.frame
    config = st.session_state.experiment_config
    valid = st.session_state.validation["status"] == "通过"
    scheme = st.selectbox("训练方案", ["全部方案", *SCHEME_LABELS.keys()], format_func=lambda key: "全部方案" if key == "全部方案" else SCHEME_LABELS[key])

    if not valid:
        st.warning("数据未通过校验，训练按钮暂不可用。")
    if st.button("开始训练", type="primary", disabled=not valid):
        modes = list(SCHEME_LABELS) if scheme == "全部方案" else [scheme]
        progress = st.progress(0)
        for index, mode in enumerate(modes, start=1):
            with st.spinner(f"正在训练 {SCHEME_LABELS[mode]}"):
                st.session_state.training_results[mode] = train_scheme(frame, mode, config)
            progress.progress(index / len(modes))
        st.session_state.report_markdown = generate_report(st.session_state.training_results, config)
        st.success("训练完成")

    if st.session_state.training_results:
        st.subheader("Loss 曲线")
        loss_rows = []
        for mode, result in st.session_state.training_results.items():
            for step, loss in enumerate(result["history"]["loss"], start=1):
                loss_rows.append({"方案": SCHEME_LABELS[mode], "轮次": step, "Loss": loss})
        st.plotly_chart(px.line(pd.DataFrame(loss_rows), x="轮次", y="Loss", color="方案", markers=True), use_container_width=True)


def render_result_analysis() -> None:
    st.title("结果分析页")
    results = st.session_state.training_results
    config = st.session_state.experiment_config
    if not results:
        st.warning("暂无训练结果。")
        return

    st.subheader("三方案对比表")
    st.dataframe(results_table(results, config), use_container_width=True)

    tabs = st.tabs([SCHEME_LABELS[mode] for mode in results])
    for tab, (mode, result) in zip(tabs, results.items()):
        with tab:
            st.subheader("混淆矩阵")
            matrix = result.get("metrics", {}).get("confusion_matrix", [[0, 0], [0, 0]])
            st.plotly_chart(ff.create_annotated_heatmap(z=matrix, colorscale="Blues", showscale=True), use_container_width=True)
            st.subheader("客户端分布")
            distribution = pd.DataFrame(result.get("client_distribution", []))
            st.dataframe(distribution, use_container_width=True)
            if not distribution.empty:
                st.plotly_chart(px.bar(distribution, x="client", y="size"), use_container_width=True)
            if result.get("dp"):
                st.subheader("DP 参数")
                st.json({**result["dp"], "epsilon": config["epsilon"], "delta": config["delta"]})


def render_report_export() -> None:
    st.title("报告导出页")
    if st.button("生成 Markdown 报告"):
        st.session_state.report_markdown = generate_report(
            st.session_state.training_results,
            st.session_state.experiment_config,
        )
    markdown = st.session_state.report_markdown or generate_report(
        st.session_state.training_results,
        st.session_state.experiment_config,
    )
    st.text_area("Markdown 报告内容", markdown, height=420)
    st.download_button("下载 Markdown 报告", markdown, file_name="fedprivtab_report.md", mime="text/markdown")


def main() -> None:
    st.set_page_config(page_title="FedPrivTab", layout="wide")
    initialize_state()
    page = render_sidebar()
    if page == "首页":
        render_home()
    elif page == "客户端管理页":
        render_client_management()
    elif page == "数据上传与审核页":
        render_data_upload()
    elif page == "数据分析页":
        render_data_analysis()
    elif page == "实验配置页":
        render_experiment_config()
    elif page == "训练监控页":
        render_training_monitor()
    elif page == "结果分析页":
        render_result_analysis()
    elif page == "报告导出页":
        render_report_export()


if __name__ == "__main__":
    main()
