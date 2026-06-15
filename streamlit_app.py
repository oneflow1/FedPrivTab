from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from data_utils import generate_sample_data, train_test_data, validate_tabular_data
from training import TrainConfig, train_model

st.set_page_config(page_title="FedPrivTab", layout="wide")
st.title("FedPrivTab")
st.caption("联邦学习表格数据实验：Centralized MLP / FedAvg / DP-FedAvg")

with st.sidebar:
    st.header("实验配置")
    mode = st.selectbox("训练方案", ["centralized", "fedavg", "dp_fedavg"])
    samples = st.number_input("样本数", min_value=50, max_value=2000, value=240, step=10)
    features = st.number_input("特征数", min_value=2, max_value=50, value=6, step=1)
    clients = st.number_input("客户端数", min_value=2, max_value=20, value=4, step=1)
    non_iid = st.checkbox("Non-IID 划分", value=True)
    epochs = st.number_input("集中式轮数", min_value=1, max_value=20, value=3, step=1)
    rounds = st.number_input("联邦轮数", min_value=1, max_value=20, value=3, step=1)
    local_epochs = st.number_input("本地轮数", min_value=1, max_value=10, value=1, step=1)
    batch_size = st.number_input("Batch Size", min_value=4, max_value=128, value=16, step=4)
    lr = st.number_input("学习率", min_value=0.0001, max_value=1.0, value=0.01, step=0.001, format="%.4f")
    clip_norm = st.number_input("裁剪阈值", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    noise_multiplier = st.number_input("噪声倍率", min_value=0.0, max_value=5.0, value=0.2, step=0.1)

if "frame" not in st.session_state:
    st.session_state.frame = generate_sample_data(samples=samples, features=features, clients=clients)

left, right = st.columns([1, 1])
with left:
    if st.button("生成示例数据"):
        st.session_state.frame = generate_sample_data(samples=samples, features=features, clients=clients)
    uploaded = st.file_uploader("上传 CSV 数据", type=["csv"])
    if uploaded is not None:
        st.session_state.frame = pd.read_csv(uploaded)
    st.subheader("数据预览")
    st.dataframe(st.session_state.frame.head(20), use_container_width=True)

validation = validate_tabular_data(st.session_state.frame)
with right:
    st.subheader("数据校验")
    st.write({"valid": validation.valid, "message": validation.message, "details": validation.details})
    if st.session_state.frame is not None:
        st.bar_chart(st.session_state.frame.select_dtypes(include="number").mean())

run = st.button("开始训练", type="primary")
if run:
    x_train, x_test, y_train, y_test, _ = train_test_data(st.session_state.frame)
    config = TrainConfig(
        mode=mode,
        epochs=int(epochs),
        rounds=int(rounds),
        clients=int(clients),
        local_epochs=int(local_epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        clip_norm=float(clip_norm),
        noise_multiplier=float(noise_multiplier),
        non_iid=bool(non_iid),
    )
    result = train_model(x_train, y_train, x_test, y_test, config)
    st.session_state.result = result

if "result" in st.session_state:
    result = st.session_state.result
    st.subheader("结果指标")
    st.dataframe(pd.DataFrame([result["metrics"]]), use_container_width=True)
    st.subheader("Loss 曲线")
    st.line_chart(result["history"]["loss"])
    st.subheader("客户端分布")
    st.dataframe(pd.DataFrame(result["client_distribution"]), use_container_width=True)
    st.subheader("混淆矩阵")
    st.json(result["metrics"]["confusion_matrix"])
    if result.get("dp"):
        st.subheader("DP 参数")
        st.write(result["dp"])
    report = [
        "# FedPrivTab 实验报告",
        f"- 训练方案: {result['mode']}",
        f"- 指标: {json.dumps(result['metrics'], ensure_ascii=False)}",
        f"- 客户端分布: {json.dumps(result['client_distribution'], ensure_ascii=False)}",
    ]
    if result.get("dp"):
        report.append(f"- DP 参数: {json.dumps(result['dp'], ensure_ascii=False)}")
    st.download_button("下载 Markdown 报告", "\n".join(report), file_name="fedprivtab_report.md")
