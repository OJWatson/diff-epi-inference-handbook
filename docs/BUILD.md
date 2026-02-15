# Building the handbook locally

This repo uses a single-source **Quarto** book (in `book/`) with executable Python code blocks.

## Prerequisites

- Python **3.11** (CI uses 3.11; 3.10+ may work but is not tested in CI)
- Quarto CLI **1.8.27** (CI pins this version for reproducibility)
  - https://quarto.org/docs/get-started/

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
python3 -m pip install -U pip

# Install the package + dev deps (includes Jupyter + PyYAML for Quarto execution)
pip install -e '.[dev]'

# If you want to execute the Modern SBI (JAX) examples as well:
pip install -e '.[modern-sbi]'

# Quick sanity checks
ruff check .
pytest
```

## Optional dependency extras ("extras matrix")

This repo keeps the base install lightweight (NumPy-only). Extra functionality is exposed via
Python *extras* in `pyproject.toml`.

Install extras like:

```bash
pip install -e '.[dev]'
pip install -e '.[modern-sbi]'
```

Available extras:

- `dev`: tooling + runtime deps for running tests and executing Quarto code blocks.
- `jax`: JAX runtime (CPU wheels).
- `nuts`: NUTS baseline stack (JAX CPU + BlackJAX). Tests that require this are skipped unless
  the extra is installed.
- `modern-sbi`: "modern SBI" stack used by the JAX-based conditional density/flow examples.

### CPU vs GPU notes for JAX

The extras intentionally depend on **CPU** wheels via `jax[cpu]` to keep installs reproducible
in CI.

If you want **GPU** acceleration, install the appropriate CUDA-enabled JAX wheel for your
platform *after* installing the repo extras, following the official JAX instructions:

- https://jax.readthedocs.io/en/latest/installation.html

For example (CUDA 12 on Linux; adjust for your setup):

```bash
# After installing one of the extras that pulls in jax[cpu]
python -m pip install -U "jax[cuda12]" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

If you do this, it’s normal for `pip` to replace the CPU-only `jaxlib` wheel with a CUDA one.

## Verify tool versions

```bash
python3 --version
quarto --version
```

Note: some systems do not provide a `python` executable (only `python3`), so the docs use `python3` explicitly.

### If `quarto` is not installed

If you don’t have Quarto available on your PATH (e.g. `quarto: command not found`), you can download the pinned CI version **without** needing system-wide install.

Note: this workflow downloads Quarto into a local `.tools/` directory. `.tools/` is intentionally **gitignored** and **excluded from linting** (e.g. `ruff`) because it is a local build artefact.

```bash
QUARTO_VER=1.8.27
ARCHIVE="quarto-${QUARTO_VER}-linux-amd64.tar.gz"
URL="https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VER}/${ARCHIVE}"

mkdir -p .tools
curl -fsSLo ".tools/${ARCHIVE}" "$URL"
tar -xzf ".tools/${ARCHIVE}" -C .tools

# Use the downloaded binary
.tools/quarto-${QUARTO_VER}/bin/quarto --version
```

Then render with:

```bash
.tools/quarto-${QUARTO_VER}/bin/quarto render book --to html
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
