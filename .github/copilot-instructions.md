# Copilot Instructions

## Workspace Overview
This repository is a Python monorepo for anomaly-detection evaluation across four telemetry sources:

- logs
- traces
- metrics
- Kubernetes events

Keep each modality in its own project under `projects/` so data contracts, parsers, and metrics stay independent unless a shared abstraction is clearly justified.

## Repository Layout
- `projects/log-evaluation/` for log-oriented anomaly detection and scoring
- `projects/trace-evaluation/` for span, latency, and dependency-graph evaluation
- `projects/metrics-evaluation/` for time-series and signal-based evaluation
- `projects/k8s-events-evaluation/` for cluster-event correlation and anomaly scoring
- `.github/instructions/` for repo-wide Copilot instructions
- `.github/skills/` for task-specific Copilot skills

## Sub Porject structure
- Every sub-project in path ../projects/**/src/<project-name> should follow this structure:
    - __init__.py: re-export the project's public API (config, model, evaluation functions, etc.) via explicit imports and __all__. Do NOT put CLI logic or business logic here.
    - __main__.py: CLI entry point. Imports main() from cli.py and calls it, enabling `python -m <project-name>`.
    - cli.py: define input/output params (argparse) and the main() function that runs and tests the project locally.
    - preprocessing.py: parse the input file, preprocess data if needed (define as a separate file when the preprocessing step is complex).
    - model.py: implement the core model logic to classify the log lines (if using an ML approach).
    - train.py: implement the training logic to train the model (if using an ML approach).
    - evaluation.py: implement the scoring logic to evaluate the model performance.
    - algo.py: implement the main logic for algorithm execution if the approach is algorithm-based (not needed if using an ML approach).
    - config.py: define the fixed configuration params for the project.
    - utils.py: implement any utility functions needed for the project.

## Python Rules
- Target Python 3.11+.
- Use `src/` layout packages for every project.
- Add type hints to every public function, class, and module-level constant.
- Prefer `pathlib.Path` over string paths.
- Use `dataclass` for small immutable config and result objects.
- Keep I/O at the edges; make core parsing and scoring functions pure where possible.
- Prefer vectorized `numpy` or `pandas` operations over Python loops.
- Use `logging` instead of `print` for runtime diagnostics.
- Fail fast with specific exceptions and actionable error messages.
- Keep module import side effects minimal.
- Write docstrings for public APIs.
- Do not use emojis or icon characters in comments, docstrings, or commit messages.
- Keep comments short and only add them when the code is not self-explanatory; explain why, not what.
- Remember to update requirements.txt in root folder if needed.

## Evaluation Rules
- All code must pass the linter rules defined in `pyproject.toml` before being considered complete.
- DO NOT write unit tests (or any other types of test files) and run python code except being asked in prompt.
- Report metrics in structured outputs in CSV format if being asked to write unit tests or evaluation scripts.
- Use venv python path `~/.venv/bin/activate` to run python code if required. Do not use the system python path.

## LLM token usage rules
- Save and use input and output LLM token efficiently as much as possible.

## What To Avoid
- Do not introduce new dependencies unless the standard library, `numpy`, `pandas`, or `scikit-learn` cannot cover the need.
- Do not hardcode file paths or environment-specific values.
- Do not use global mutable state.
- Do not mix modality logic across projects unless a shared interface is actively reused.
