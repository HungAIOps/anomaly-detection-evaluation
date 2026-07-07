---
name: plan-creation
description: Create a markdown report file describing what will change for a plan task.
---

## Plan ID
- Each plan gets a sequential 4-digit zero-padded ID, unique across the entire
  repository (not per-folder, not per-month).
- Before creating a new plan, check the highest existing ID across all files in
  `plans/**/*.md` and increment by 1.
- Filename format: `{id}-{short-kebab-case-slug}.md`

## Plan File Template
Every plan must follow this structure:

```markdown
# Plan {id}: {Title}

- **Date:** {YYYY-MM-DD}
- **Status:** draft | approved | in-progress | done | abandoned
- **Scope:** {affected project(s), e.g. projects/log-evaluation/}

## Goal
{One or two sentences describing the desired outcome.}

## Approach
{Step-by-step description of how the change will be made.}

## Files Affected
- {path/to/file.py}
   + {what changes}
   + ....
- {path/to/other_file.py}
   + {what changes}
   + ....

## Risks / Open Questions
{Anything uncertain, edge cases, or decisions needing user input.}

## Acceptance Criteria
- [ ] {testable condition 1}
- [ ] {testable condition 2}
```

## Rules
- Do not skip the ID lookup step — never reuse or guess an ID.
- Do not start editing code until the plan's Status is `approved`.
- Update Status to `in-progress` when work begins and `done` when complete.
- Keep plans concise; avoid restating requirements already covered in
  copilot-instructions.md.
- No emojis or icons in plan files, consistent with the repo's comment style rules.