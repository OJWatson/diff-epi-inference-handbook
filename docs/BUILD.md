# Building the handbook locally

This repo uses a single-source **Quarto** book (in `book/`) with executable Python code blocks.

## Prerequisites

- Python **3.10+**
- Quarto CLI: https://quarto.org/docs/get-started/

### PDF builds

PDF output requires a LaTeX distribution. The CI uses TinyTeX.

Options:
- TinyTeX (recommended for a lightweight install): https://yihui.org/tinytex/
- TeX Live (system package manager)

## Set up a Python environment

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# Install the package + dev deps (includes Jupyter + PyYAML for Quarto execution)
pip install -e '.[dev]'

# Quick sanity checks
ruff check .
pytest
```

## Render the book

### HTML

```bash
quarto render book --to html
```

Output will be written to `book/_book/`.

### PDF

```bash
quarto render book --to pdf
```

If PDF rendering fails, check that a LaTeX distribution is installed and visible on your PATH.
