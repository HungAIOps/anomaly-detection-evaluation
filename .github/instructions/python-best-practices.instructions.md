---
applyTo: "**/*.py"
---

# Python Best Practices

- Use Python 3.11+ features intentionally, especially `from __future__ import annotations` where it improves clarity.
- Type-hint all public functions, methods, and return values.
- Prefer `pathlib.Path` for filesystem work.
- Use `dataclass` for simple immutable records and config objects.
- Keep functions small and single-purpose.
- Put I/O at the boundary and keep business logic pure when possible.
- Use `logging` instead of `print`.
- Raise specific exceptions with actionable messages.
- Prefer `numpy` and `pandas` vectorization over manual loops for tabular processing.
- Use `pytest` tests with small synthetic fixtures.
- Keep imports sorted and remove unused code promptly.
- Preserve `src/` layout boundaries and avoid circular imports.
- Use `ruff`-friendly style and keep lines readable at or below 100 characters.
