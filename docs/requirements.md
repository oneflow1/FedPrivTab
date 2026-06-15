# FedPrivTab Requirements

## Functional Requirements

1. Provide a Streamlit UI for FedPrivTab experimentation.
2. Provide a Flask backend for health, sample data generation, validation, and training.
3. Support a shared PyTorch MLP model across centralized, FedAvg, and DP-FedAvg training modes.
4. Allow users to generate sample tabular data or upload their own CSV data.
5. Validate input schema, target column presence, sample count, missing values, and basic data consistency.
6. Support basic IID and Non-IID client partitioning for federated experiments.
7. Show training metrics, curves, confusion matrix, and client distribution.
8. Export a Markdown report summarizing the experiment.

## Non-Functional Requirements

1. Keep the system lightweight and dependency-minimal.
2. Use small default datasets and short training schedules for fast execution.
3. Ensure the code runs without external data sources.
4. Keep the implementation modular and testable.
5. Make the outputs deterministic when a seed is supplied.

## Metrics

- Accuracy
- Precision
- Recall
- F1
- Loss
- AUC when the binary classification setting allows it
