#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NUITKA_ROOT="${NUITKA_ROOT:-"${PROJECT_ROOT}/../Nuitka"}"
ENTRY_SCRIPT="${ENTRY_SCRIPT:-"${PROJECT_ROOT}/src/core_pdf/__main__.py"}"
PGO_ARGS="${PGO_ARGS:-"$(find "${PROJECT_ROOT}/tests" -type f -name '*.pdf' | sort | while read -r f; do printf '%q ' "$f"; done)"}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-core_pdf_pgo}"
CACHE_DIR="${CACHE_DIR:-"$(mktemp -d "${TMPDIR:-/private/tmp}/core-pdf-nuitka-pgo-cache.XXXXXX")"}"

PYTHON="${PROJECT_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
    echo "==> Error: no virtualenv Python at ${PYTHON}; run 'uv sync' first." >&2
    exit 1
fi

if [[ ! -x "${NUITKA_ROOT}/bin/nuitka" ]]; then
    echo "==> Error: local Nuitka worktree not found at ${NUITKA_ROOT}." >&2
    exit 1
fi

echo "==> Building core-pdf CLI with C-level PGO (--pgo-c)..."
PYTHONPATH="${PROJECT_ROOT}/src" \
NUITKA_CACHE_DIR="${CACHE_DIR}" \
CCACHE_DISABLE=1 \
"${PYTHON}" "${NUITKA_ROOT}/bin/nuitka" \
    --mode=accelerated \
    --include-package=core_pdf \
    --include-package-data=core_pdf \
    --pgo-c \
    --lto=yes \
    --assume-yes-for-downloads \
    --pgo-args="${PGO_ARGS}" \
    --output-filename="${OUTPUT_FILENAME}" \
    "${ENTRY_SCRIPT}"

BINARY="${PWD}/${OUTPUT_FILENAME}"
if [[ ! -x "${BINARY}" ]]; then
    echo "==> Error: PGO build failed; no binary at ${BINARY}" >&2
    exit 1
fi

echo "==> PGO build succeeded: ${BINARY}"
ls -lh "${BINARY}"
