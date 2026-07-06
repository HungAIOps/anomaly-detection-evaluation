---
applyTo: "projects/**"
---

# Evaluation Workflow

- Keep each telemetry source isolated in its own project.
- Share code only through deliberate common helpers, never by accidental cross-imports.
- Name outputs, fixtures, and artifact folders after the modality being evaluated.
- Make metric definitions explicit and documented so evaluation remains reproducible.
- Prefer structured artifacts such as JSON, CSV, or Parquet over ad hoc text output.
- Add smoke tests before expanding model or feature extraction logic.
