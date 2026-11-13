# SPDX-License-Identifier: AGPL-3.0-only

"""Check that a plain text extraction leaves the deferred modules unloaded.

Modules import what a text extraction does not need with `lazy import`. An eager import
added later would load one of them again and quietly cost the time back, so this extracts
the first page of a text PDF through each facade, each in a fresh interpreter, and fails
naming any deferred module that loaded.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
TEXT_PDF = REPOSITORY_ROOT / "tests/upstream/pdfplumber/tests/pdfs/federal-register-2020-17221.pdf"

# Rendering, tables, images and their codecs, transparency, structure trees, the PDF MAC and
# cipher stacks, and TrueType programs: none of them is needed to read text from this page.
DEFERRED = (
    "asn1crypto",
    "cryptography",
    "core_pdf.impl.crypto.ciphers",
    "core_pdf.impl.crypto.pdf_mac",
    "imagecodecs",
    "core_pdf.impl.fonts.sfnt",
    "core_pdf.impl.document.structure",
    "core_pdf.impl.extract.table_core",
    "core_pdf.impl.graphics.images",
    "core_pdf.impl.graphics.shading",
    "core_pdf.impl.render.page",
    "core_pdf.impl.render.target",
    "core_pdf.impl.spec.s_11_transparency.blend",
)

WORKLOADS = {
    "pypdf": (
        "import core_pdf_compat.pypdf as pypdf\npypdf.PdfReader(path).pages[0].extract_text()\n"
    ),
    "pdfplumber": (
        "import core_pdf_compat.pdfplumber as pdfplumber\n"
        "with pdfplumber.open(path) as pdf:\n"
        "    pdf.pages[0].extract_text()\n"
    ),
}


def loaded_modules(workload: str) -> set[str]:
    program = (
        f"import json, sys\npath = {str(TEXT_PDF)!r}\n{WORKLOADS[workload]}"
        "print(json.dumps(sorted(sys.modules)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )
    return set(json.loads(result.stdout))


def main() -> int:
    # A renamed module would otherwise never appear in sys.modules and pass unguarded.
    missing = [name for name in DEFERRED if importlib.util.find_spec(name) is None]
    if missing:
        print(f"DEFERRED names that do not exist: {', '.join(missing)}")
        return 1
    failed = False
    for workload in WORKLOADS:
        loaded = sorted(set(DEFERRED) & loaded_modules(workload))
        if loaded:
            failed = True
            print(f"{workload}: text extraction loaded {', '.join(loaded)}")
        else:
            print(f"{workload}: {len(DEFERRED)} deferred modules stayed unloaded")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
