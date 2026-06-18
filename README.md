# FedPrivTab

FedPrivTab 是一个面向 Adult Census Income 二分类任务的表格数据联邦学习实验系统，用于在同一套数据与流程下对比三类模型：集中式 MLP、FedAvg MLP 和 DP-FedAvg MLP。项目包含 Vue 3 + Vite 前端、Flask API 后端、PyTorch 训练代码、默认演示账号、Notebook 实验产物与 Ubuntu systemd 部署示例。

GitHub 地址：https://github.com/oneflow1/FedPrivTab

## 项目简介

本项目当前聚焦固定的 Adult Census Income 收入预测场景，支持从数据准备、字段检查、预处理、训练配置、模型训练到结果分析的完整流程。后端默认运行在 `5000` 端口，前端开发服务默认运行在 `8501` 端口。

系统提供三种训练模式：

- `centralized`：集中式 MLP，在集中式预处理数据上训练。
- `fedavg`：FedAvg MLP，使用客户端数据准备页生成的客户端数据进行联邦平均训练。
- `dp_fedavg`：DP-FedAvg MLP，在 FedAvg 基础上对客户端更新做 L2 裁剪并加入高斯噪声。

## 功能概览

- Vue 3 + Vite 前端：登录、首页概览、客户端管理、数据分析、数据预处理、实验训练、结果分析与报告导出。
- Flask 后端：健康检查、认证会话、用户管理、数据校验、预处理版本保存、异步任务、模型训练与 Markdown 报告生成。
- 认证与审计：本地 SQLite 保存用户、会话、审计日志和预处理版本。
- 模型训练：基于 PyTorch 的 MLP、FedAvg 和 DP-FedAvg 训练流程。
- 数据处理：缺失值处理、类别编码、数值缩放、Adult 目标列推断和客户端数据对齐。
- 实验产物：提供包含交互流程、API 脚本和 Matplotlib 图表的 Notebook。

## 快速启动

安装 Python 依赖：

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

启动 Flask 后端：

```bash
python app.py
```

后端默认监听 `http://127.0.0.1:5000`。首次启动时，如果没有设置 `FEDPRIVTAB_AUTH_DB`，会在项目目录创建 `fedprivtab_auth.sqlite3`。

启动 Vue 前端开发服务：

```bash
cd frontend
npm install
npm run dev
```

前端默认监听 `http://127.0.0.1:8501`。当前项目已经移除 Streamlit 旧版入口，不再提供 Streamlit 启动方式。

## 默认账号

默认演示账号会在用户表为空时自动创建。

| 用户名 | 密码 | 角色 |
|---|---|---|
| `admin` | `admin123` | 系统管理员 |
| `researcher` | `research123` | 实验研究人员 |
| `client-1` | `client123` | 客户端用户 |
| `client-2` | `client123` | 客户端用户 |
| `client-3` | `client123` | 客户端用户 |
| `client-4` | `client123` | 客户端用户 |

## 训练配置默认值

主实验默认值已经按当前前端和 `/train` API 对齐：

| 模式 | 轮次 | Batch | 学习率 | 学习率调度 | MLP 结构 | 激活函数 |
|---|---:|---:|---:|---|---|---|
| 集中式 MLP | `epochs=50` | `128` | `0.05` | `step_decay` | `hidden_layers=2`, `hidden_units='64,32'` | `ReLU` |
| FedAvg MLP | `rounds=50` | `32` | `0.05` | `step_decay` | `hidden_layers=2`, `hidden_units='64,32'` | `ReLU` |
| DP-FedAvg MLP | `rounds=50` | `32` | `0.03` | `step_decay` | `hidden_layers=2`, `hidden_units='64,32'` | `ReLU` |

通用调度默认值：

- `lr_schedule='step_decay'`
- `lr_decay=0.5`
- `lr_step_size=15`
- `lr_min=0.005`
- `local_epochs=1`
- `clients=4`
- `client_fraction=1.0`
- `dirichlet_alpha=0.3`
- `seed=42`

DP-FedAvg 默认隐私机制参数：

- `clip_norm=1.0`
- `noise_multiplier=0.1`
- `epsilon=4.0`
- `delta=1e-5`

## Notebook/实验产物

当前 Notebook 产物位于：

```text
data/mywork/final_outputs/FedPrivTab_PostProject_API_Notebook.ipynb
```

Notebook 标题为 `FedPrivTab：交互形式 + API 形式对比三个模型`，内容包括：

- Vue 交互式训练流程截图说明。
- 真实 `/preprocess` 与 `/train` API 调用脚本。
- 集中式 MLP、FedAvg MLP、DP-FedAvg MLP 三模型结果对比。
- 学习率轨迹、准确率、F1 等 Matplotlib 图表。

## 主要 API

- `GET /health`：健康检查。
- `POST /auth/login`：登录，参数为 `username` 和 `password`。
- `POST /auth/logout`：退出登录，支持 JSON `session_id` 或请求头 `X-Session-Id`。
- `GET /auth/status`：检查当前会话。
- `GET /users`：管理员查看用户列表，可按 `role` 过滤。
- `POST /users`：管理员创建用户。
- `PATCH /users/<username>/password`：管理员重置用户密码。
- `PATCH /users/<username>/status`：管理员启用或停用用户。
- `GET /sample-data`：生成示例数据。
- `POST /validate`：校验上传或生成的表格数据。
- `POST /preprocess`：执行缺失值处理、类别编码和数值缩放；支持 JSON records 或 multipart CSV。
- `GET /preprocess/versions`：查看预处理版本。
- `POST /preprocess/versions`：保存预处理版本。
- `POST /train`：训练 `centralized`、`fedavg` 或 `dp_fedavg` 模型。
- `POST /report`：将训练结果生成 Markdown 报告。
- `GET /jobs/<id>`：查询异步任务状态。

`/train` 支持的学习率调度参数包括 `lr_schedule`、`lr_decay`、`lr_step_size` 和 `lr_min`。其中 `lr_schedule` 可选 `constant`、`step_decay`、`linear_decay`。

## 测试与构建

运行后端测试：

```bash
pytest
```

构建前端静态资源：

```bash
cd frontend
npm install
npm run build
```

本地预览构建结果：

```bash
cd frontend
npm run preview
```

## 部署说明

Ubuntu systemd 示例位于 `deploy/systemd/`：

- `fedprivtab-api.service`：Flask API，端口 `5000`。
- `fedprivtab-ui.service`：Vue 静态 UI，端口 `8501`。

详细步骤见 `docs/deployment.md`。源码仓库建议不提交 `frontend/dist/`，部署时在目标环境执行 `npm run build` 生成静态文件。

## 注意事项

- 本项目当前实验范围固定为 Adult Census Income，不是通用数据集管理平台。
- DP-FedAvg 是机制演示：代码对客户端更新做 L2 裁剪并加入高斯噪声，但没有集成 Opacus，也没有提供严格的隐私会计证明。
- FedAvg 和 DP-FedAvg 训练要求先完成客户端数据准备，不能直接使用集中式 `dataset_id`。
- 默认账号仅用于演示和本地实验，公开部署前应修改密码并妥善配置 SQLite 存储路径。
- `frontend/node_modules/`、`frontend/dist/`、缓存、日志和本地运行数据库不应作为源码提交。
