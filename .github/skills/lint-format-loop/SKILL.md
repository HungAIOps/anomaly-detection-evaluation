---
name: lint-format-loop
description: Run Ruff on the anomaly-detection-evaluation folder in a fix-check loop until it passes or needs a human decision.
---

# Anomaly Detection Evaluation Lint Loop

Use this skill when fixing lint problems in the anomaly-detection-evaluation workspace.

- Default the target to `d:\MyAIOps\anomaly-detection-evaluation`.
- Read `pyproject.toml` first so the run matches the repository's Ruff settings.
- Run Ruff on the chosen target, apply safe fixes, then format and re-check.
- If findings remain, apply unsafe fixes only when allowed, then format and re-check again.
- Keep looping until Ruff passes, the remaining findings require a judgment call, or the iteration budget is reached.
- Keep edits inside the anomaly-detection-evaluation folder unless the user explicitly expands scope.

## Expected inputs

- `target_path` (optional): folder or file to check. Default: `anomaly-detection-evaluation`.
- `ruff_runner` (optional): explicit Ruff command prefix such as `uv run ruff`, `ruff`, or `python -m ruff`.
- `allow_unsafe_fixes` (default: true): whether unsafe autofixes may be used after safe fixes.
- `max_iterations` (default: 10): maximum number of fix-check loops.
- `ask_on_ambiguity` (default: true): ask the user before choosing between valid fixes.

## Workflow

1. Resolve a Ruff command.
2. Run `<ruff_cmd> check <target_path>`.
3. If there are fixable findings, run `<ruff_cmd> check <target_path> --fix`.
4. Run `<ruff_cmd> format <target_path>`.
5. Run `<ruff_cmd> check <target_path>` again.
6. If findings remain and unsafe fixes are allowed, run `<ruff_cmd> check <target_path> --fix --unsafe-fixes`.
7. Format again and re-check.
8. Repeat with manual edits for anything Ruff cannot fix automatically.

## Stop conditions

Stop when Ruff passes for the chosen scope, a remaining issue needs a human choice, or the loop stops making progress.

## Output

Report the scope used, Ruff command used, number of iterations, fixes applied, manual edits, and any remaining findings.
