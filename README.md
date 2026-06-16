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

The Streamlit UI is organized into the eight sections described in `docs/requirements.md`:

- 首页: experiment overview, client counts, data validation state, and completed training schemes
- 客户端管理页: add clients, enable or disable clients, and inspect client status
- 数据上传与审核页: upload CSV data, generate sample data, select the label column, and validate data
- 数据分析页: statistical summaries, label distribution, client label distribution, feature means and distributions, and correlation heatmap
- 实验配置页: configure MLP, IID / Non-IID, FedAvg, and differential privacy parameters
- 训练监控页: run `centralized`, `fedavg`, `dp_fedavg`, or all schemes and compare loss curves
- 结果分析页: compare scheme metrics, confusion matrices, client distributions, and DP parameters
- 报告导出页: generate and download a Markdown experiment report

The top bar includes login/logout controls. Users must log in before accessing pages; the authenticated role controls which of the eight pages are visible. The app stores clients, uploaded or generated data, validation state, experiment configuration, training results, and report content in `st.session_state`, while users, sessions, and login/logout audit events are stored in SQLite.

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
- `PATCH /users/<username>/status` – manager-only enable/disable account status
- `GET /sample-data` – generate a sample dataset
- `POST /validate` – validate uploaded or generated tabular data
- `POST /train` – run centralized, FedAvg, or DP-FedAvg training
- `POST /report` – convert a training result payload into a Markdown report

## Notes

- The default configuration uses small synthetic data, few epochs, and a small number of rounds so the system runs quickly in constrained environments.
- DP-FedAvg applies client update clipping and Gaussian noise to model updates.
