# Differentiable Epidemiology — Inference Handbook

This repository contains a short, runnable handbook (book/site) plus minimal Python scaffolding for examples.

## Local dev

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

ruff check .
pytest

# Build the site/book
quarto render book --to html
# Optional:
quarto render book --to pdf
```
