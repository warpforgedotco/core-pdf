#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# Standards packages beneath core-pdf-spec; every one ships Python and the
# fonts package ships CMap data, so include both code and package data.
STANDARDS_PACKAGES=(core_predictors core_postscript core_jbig2 core_pdf_crypto core_adobe_fonts)
INCLUDE_ARGS=(--include-package=core_pdf_spec --include-package-data=core_pdf_spec)
for package in "${STANDARDS_PACKAGES[@]}"; do
    INCLUDE_ARGS+=("--include-package=${package}" "--include-package-data=${package}")
done

echo "==> Building core-pdf standalone executable with Nuitka..."
uv run nuitka "${INCLUDE_ARGS[@]}" \
  --python-flag=-m src/core_pdf

if [[ -f "${PROJECT_ROOT}/core_pdf.bin" ]]; then
    echo "==> Successfully created standalone binary: ${PROJECT_ROOT}/core_pdf.bin"
    ls -lh "${PROJECT_ROOT}/core_pdf.bin"
else
    echo "==> Error: core_pdf.bin build failed!" >&2
    exit 1
fi
