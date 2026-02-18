#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v rg >/dev/null 2>&1; then
  SEARCH_CMD="rg"
else
  SEARCH_CMD="grep"
fi

search_matches() {
  local pattern="$1"
  shift
  if [[ "$SEARCH_CMD" == "rg" ]]; then
    rg -n -e "$pattern" "$@"
  else
    grep -nE "$pattern" "$@"
  fi
}

search_exists() {
  local pattern="$1"
  shift
  if [[ "$SEARCH_CMD" == "rg" ]]; then
    rg -q -e "$pattern" "$@"
  else
    grep -qE "$pattern" "$@"
  fi
}

echo "[book-quality] Checking for roadmap-style language..."
ROADMAP_PATTERN='## Goals|## Outline|## Plan|Deliverables:|to be expanded|working taxonomy|working sketch|milestone|next steps|work in progress'
if search_matches "$ROADMAP_PATTERN" book/*.qmd; then
  echo "[book-quality] Found roadmap-style sectioning or language."
  exit 1
fi

echo "[book-quality] Checking bibliography wiring..."
if ! search_exists '^bibliography:' book/_quarto.yml; then
  echo "[book-quality] Missing bibliography in book/_quarto.yml"
  exit 1
fi

if [[ ! -f book/references.bib ]]; then
  echo "[book-quality] Missing book/references.bib"
  exit 1
fi

if ! search_exists '\[@' book/*.qmd; then
  echo "[book-quality] No citation markers found in chapter files."
  exit 1
fi

echo "[book-quality] Checking python code chunk length..."
MAX_LINES=95
violations=0
while IFS=$'\t' read -r file idx start lines; do
  if (( lines > MAX_LINES )); then
    echo "[book-quality] Long code chunk: ${file} cell_${idx} start=${start} lines=${lines} (max=${MAX_LINES})"
    violations=1
  fi
done < <(
  for f in book/*.qmd; do
    awk 'BEGIN{inside=0;len=0;idx=0}
      /^```\\{python/{inside=1;len=0;idx++;start=NR;next}
      /^```/{if(inside){printf "%s\t%d\t%d\t%d\n", FILENAME, idx, start, len; inside=0} next}
      {if(inside)len++}
    ' "$f"
  done
)

if (( violations == 1 )); then
  exit 1
fi

echo "[book-quality] OK"
