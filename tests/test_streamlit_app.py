from __future__ import annotations

import pandas as pd

import streamlit_app


def test_streamlit_app_helpers_are_importable() -> None:
    config = streamlit_app.default_experiment_config()
    results = {
        "centralized": {
            "mode": "centralized",
            "rows": 30,
            "metrics": {
                "accuracy": 0.8,
                "precision": 0.75,
                "recall": 0.7,
                "f1": 0.72,
                "auc": 0.81,
                "confusion_matrix": [[10, 2], [3, 15]],
            },
            "history": {"loss": [0.7, 0.5], "accuracy": [0.7, 0.8]},
            "client_distribution": [{"client": 0, "size": 15}, {"client": 1, "size": 15}],
            "dp": None,
        },
        "dp_fedavg": {
            "mode": "dp_fedavg",
            "rows": 30,
            "metrics": {
                "accuracy": 0.76,
                "precision": 0.7,
                "recall": 0.68,
                "f1": 0.69,
                "auc": 0.78,
                "confusion_matrix": [[9, 3], [4, 14]],
            },
            "history": {"loss": [0.75, 0.58], "accuracy": [0.66, 0.76]},
            "client_distribution": [{"client": 0, "size": 12}, {"client": 1, "size": 18}],
            "dp": {"epsilon": 4.0, "delta": 1e-5, "clip_norm": 1.0, "noise_multiplier": 0.2},
        },
    }

    table = streamlit_app.results_table(results, config)
    history = streamlit_app.history_frame(results)
    privacy = streamlit_app.privacy_performance_table(results, config)
    report = streamlit_app.generate_report(results, config)

    assert isinstance(table, pd.DataFrame)
    assert table.loc[0, "方案"] == "集中式 MLP"
    assert "Final Accuracy" in table.columns
    assert set(history.columns) == {"方案", "轮次", "Loss", "Accuracy"}
    assert privacy.loc[0, "noise_multiplier"] == 0.2
    assert "FedPrivTab 实验报告" in report
    assert "集中式 MLP" in report
    assert "FedAvg" in report
    assert "DP-FedAvg" in report
    assert "差分隐私公式" in report
    assert "Accuracy 曲线摘要" in report
    assert "隐私-性能对比" in report


def test_sidebar_and_client_cards_escape_html_values() -> None:
    metric = streamlit_app.sidebar_metric_card("数据状态", "<b>通过</b>", "<script>")
    clients = [
        {
            "id": "client-1",
            "name": "<img src=x>",
            "enabled": True,
            "status": "<b>待校验</b>",
            "rows": 10,
            "features": 4,
        }
    ]
    table = streamlit_app.client_table_html(clients)

    assert "<b>通过</b>" not in metric
    assert "&lt;b&gt;通过&lt;/b&gt;" in metric
    assert "<script>" not in metric
    assert "<img src=x>" not in table
    assert "&lt;img src=x&gt;" in table
