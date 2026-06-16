# FedPrivTab

FedPrivTab is a minimal end-to-end federated learning system for tabular data experiments with centralized MLP, FedAvg, and DP-FedAvg training.

## Features

- Streamlit multi-page front end for client management, data review, analysis, experiment configuration, training monitoring, result analysis, and report export
- Flask back end with health, data validation, sample generation, client-account management, and training endpoints
- Local SQLite login/session/audit persistence with demo role accounts
- PyTorch MLP training for centralized, FedAvg, and DP-FedAvg workflows
- Preprocessing helpers for missing values, categorical encoding, and numeric scaling
- Basic test coverage for data utilities and training pipelines

## Quick start

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Run the Flask API:

```bash
python app.py
```

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

On first run the app creates `fedprivtab_auth.sqlite3` in the project directory unless `FEDPRIVTAB_AUTH_DB` is set.

Default demo accounts:

| 用户名 | 密码 | 角色 |
|---|---|---|
| `admin` | `admin123` | 系统管理员 |
| `client` | `client123` | 客户端用户 |
| `researcher` | `research123` | 实验研究人员 |

## Streamlit pages

The Streamlit UI is organized into six workflow pages:

- 首页: experiment overview, client counts, preprocessing status, and completed training schemes
- 客户端管理页: manage the fixed four client accounts and update/reset passwords inline from the account list; client creation, deletion, and self-password changes are intentionally disabled
- 数据分析页: independently upload a CSV file, then inspect full-field summaries, selectable field distributions, numeric feature means, and correlation heatmaps
- 数据预处理页: upload CSV data, keep uploaded-file state across page switches, select the target variable, configure per-column missing-value handling and numeric scaling recommendations, and save processed data versions
- 实验训练页: configure MLP/FedAvg/DP parameters, select training schemes, and choose preprocessing versions for training; centralized MLP only uses administrator-created versions, while FedAvg and DP-FedAvg only use client-created versions
- 结果分析页: compare metrics, curves, confusion matrices, client distributions, DP parameters, and generate/download the Markdown report

The top bar includes login/logout controls. Users must log in before accessing pages; the authenticated role controls which pages are visible. The app stores clients, uploaded/generated data, preprocessing versions, experiment configuration, training results, and report content in `st.session_state`, while users, sessions, and login/logout audit events are stored in SQLite.

Run tests:

```bash
pytest
```

## Ubuntu deployment

Systemd unit examples live in `deploy/systemd/`, and the step-by-step Ubuntu deployment guide is in `docs/deployment.md`.

## API

- `GET /health` – health check
- `POST /auth/login` – authenticate with `username` and `password`
- `POST /auth/logout` – close a session by JSON `session_id` or `X-Session-Id`
- `GET /auth/status` – inspect a session by query `session_id` or `X-Session-Id`
- `GET /users` – manager-only user list, optionally filtered by `role`
- `POST /users` – manager-only client/research/admin account creation
- `PATCH /users/<username>/status` – legacy manager-only enable/disable account status endpoint
- The Streamlit client management page supports password changes/resets for the fixed four client accounts (`client-1` to `client-4`)
- `GET /sample-data` – generate a sample dataset
- `POST /validate` – validate uploaded or generated tabular data
- `POST /train` – run centralized, FedAvg, or DP-FedAvg training
- `POST /report` – convert a training result payload into a Markdown report

## Notes

- The default configuration uses small synthetic data, few epochs, and a small number of rounds so the system runs quickly in constrained environments.
- DP-FedAvg applies client update clipping and Gaussian noise to model updates.
