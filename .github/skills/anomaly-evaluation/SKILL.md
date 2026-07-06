---
name: anomaly-evaluation
description: Scaffold and maintain the multi-project anomaly-detection evaluation workspace.
---

# Anomaly Evaluation Skill

Use this skill when creating or updating the overall workspace structure.

- Keep log, trace, metrics, and Kubernetes event work in separate projects.
- Prefer shared abstractions only when they are used by at least two projects.
- Keep repository guidance aligned with Python best practices and pytest-first testing.
- Favor explicit, typed APIs over implicit module-level behavior.
