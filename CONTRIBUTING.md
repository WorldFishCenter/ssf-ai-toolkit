# Contributing

## Dev setup
```bash
git clone <repo-url>
cd ssf-ai-toolkit
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
pre-commit install
pytest -q
```

## Pull requests
- Small, focused PRs.
- Update/record model cards for any new or updated model.
- Add/refresh tests and docs.

## Data
- Do not commit private data. Use `.gitignore`d `data/` for local only.
- Provide **schema** and **column docs** for any dataset used.
