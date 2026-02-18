# Differentiable Epidemiology — Inference Handbook

This repository contains a short, runnable handbook (book/site) plus minimal Python scaffolding for examples.

## Project docs

- Build and execution guide: [`docs/BUILD.md`](docs/BUILD.md)
- Code and chapter map: [`docs/SPEC_TO_REPO_MAP.md`](docs/SPEC_TO_REPO_MAP.md)
- Project brief: [`docs/DIFF_EPI_DEVELOPMENT_PLAN.md`](docs/DIFF_EPI_DEVELOPMENT_PLAN.md)

## Local acceptance gates

Run in this order from repo root:

```bash
./scripts/ci.sh
./scripts/test.sh
make test
```

`./scripts/ci.sh` includes a book-quality gate (`scripts/check_book_quality.sh`) that enforces:

- no roadmap-style section markers in chapters,
- bibliography wiring and citation usage,
- bounded Python chunk length for readability.

## Local dev

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

ruff check .
pytest
```

## Build the book/site

See [`docs/BUILD.md`](docs/BUILD.md) for prerequisites (Quarto + PDF deps) and render commands.

## Publishing

GitHub Pages is deployed by `.github/workflows/pages.yml` to the `gh-pages` branch from rendered `book/_book/`.
