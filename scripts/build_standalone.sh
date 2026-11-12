#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# Standards packages and the compiled kernels core imports; every one ships
# Python or extension modules, so include both code and package data.
STANDARDS_PACKAGES=(core_records core_pdf_cythonized)
INCLUDE_ARGS=()
for package in "${STANDARDS_PACKAGES[@]}"; do
    INCLUDE_ARGS+=("--include-package=${package}" "--include-package-data=${package}")
done

echo "==> Building core-pdf standalone executable with Nuitka..."
uv run --group compiling nuitka "${INCLUDE_ARGS[@]}" \
  --python-flag=-m src/core_pdf

if [[ -f "${PROJECT_ROOT}/core_pdf.bin" ]]; then
    echo "==> Successfully created standalone binary: ${PROJECT_ROOT}/core_pdf.bin"
    ls -lh "${PROJECT_ROOT}/core_pdf.bin"
else
    echo "==> Error: core_pdf.bin build failed!" >&2
    exit 1
fi
