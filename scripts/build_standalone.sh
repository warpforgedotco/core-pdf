#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

echo "==> Building core-pdf standalone executable with Nuitka..."
uv run nuitka --python-flag=-m src/core_pdf

if [[ -f "${PROJECT_ROOT}/core_pdf.bin" ]]; then
    echo "==> Successfully created standalone binary: ${PROJECT_ROOT}/core_pdf.bin"
    ls -lh "${PROJECT_ROOT}/core_pdf.bin"
else
    echo "==> Error: core_pdf.bin build failed!" >&2
    exit 1
fi
