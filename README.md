# FedPrivTab

FedPrivTab is a minimal end-to-end federated learning system for tabular data experiments with centralized MLP, FedAvg, and DP-FedAvg training.

## Features

- Streamlit front end for uploading or generating example tabular data
- Flask back end with health, data validation, sample generation, and training endpoints
- PyTorch MLP training for centralized, FedAvg, and DP-FedAvg workflows
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

Run tests:

```bash
pytest
```

## API

- `GET /health` – health check
- `GET /sample-data` – generate a sample dataset
- `POST /validate` – validate uploaded or generated tabular data
- `POST /train` – run centralized, FedAvg, or DP-FedAvg training
- `POST /report` – convert a training result payload into a Markdown report

## Notes

- The default configuration uses small synthetic data, few epochs, and a small number of rounds so the system runs quickly in constrained environments.
- DP-FedAvg applies client update clipping and Gaussian noise to model updates.
