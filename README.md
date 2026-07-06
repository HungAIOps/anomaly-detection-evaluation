# Anomaly Detection Evaluation

This workspace is a Python monorepo for evaluating anomaly-detection approaches across multiple telemetry sources:

- `projects/log-evaluation/`
- `projects/trace-evaluation/`
- `projects/metrics-evaluation/`
- `projects/k8s-events-evaluation/`

Each project is isolated so the data model, preprocessing, and evaluation logic for one modality do not leak into the others.

## Working Rules

- Use Python 3.11 or newer.
- Keep each project in `src/` layout with its own `pyproject.toml`.
- Prefer typed, pure functions and dataclasses for configuration and results.
- Store evaluation outputs as JSON or CSV.
- Add `pytest` coverage for parsers, metrics, and edge cases before expanding model logic.

## Suggested Setup

Create a virtual environment in the repo root and install only the project you are working on in editable mode.

Example:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e projects/log-evaluation[dev]
```
