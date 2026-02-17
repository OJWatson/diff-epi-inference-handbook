# Differentiable Epidemiology — Inference Handbook

This repository contains a short, runnable handbook (book/site) plus minimal Python scaffolding for examples.

## Project tracking

- Milestone roadmap: [`ROADMAP.md`](ROADMAP.md)
- Current status: [`docs/STATUS.md`](docs/STATUS.md)
- Spec-to-repo mapping: [`docs/SPEC_TO_REPO_MAP.md`](docs/SPEC_TO_REPO_MAP.md)

## Local acceptance gates

Run in this order from repo root:

```bash
./scripts/ci.sh
./scripts/test.sh
make test
```

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
