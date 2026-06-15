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
            "history": {"loss": [0.7, 0.5]},
            "client_distribution": [{"client": 0, "size": 15}, {"client": 1, "size": 15}],
            "dp": None,
        }
    }

    table = streamlit_app.results_table(results, config)
    report = streamlit_app.generate_report(results, config)

    assert isinstance(table, pd.DataFrame)
    assert table.loc[0, "方案"] == "集中式 MLP"
    assert "FedPrivTab 实验报告" in report
    assert "集中式 MLP" in report
