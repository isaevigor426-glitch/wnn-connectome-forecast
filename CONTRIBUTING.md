# Contributing

Team project, 4 people. These rules exist to keep `main` green
and to avoid merge conflicts in `.ipynb` files (which are JSON
under the hood and conflict on every cell).

## Branches

- `main` — always green (`pytest` passes). Protected from direct pushes.
- `feat/*`, `fix/*`, `docs/*` — feature branches.
- Every PR requires at least 1 approve from another team member.

## Before push

Run locally and fix any failures:

```bash
ruff check .
pytest -q
