# Contributing

```bash
pip install -e ".[train,dev]"
ruff check src tests
pytest -q
```

Keep notebooks for exploration and move anything reused into `src/wnn/` with a test.
Do not commit data, checkpoints or submission archives (see `.gitignore`).
