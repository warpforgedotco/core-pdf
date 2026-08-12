#!/usr/bin/env python3
"""Run pdfminer.six's high-level tests against upstream and core-pdf."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
PDFMINER_ROOT = ROOT / "tests" / "fixtures" / "pdfminer.six"
UPSTREAM_TEST = PDFMINER_ROOT / "tests" / "test_highlevel_extracttext.py"
UPSTREAM_ENV_MARKER = "CORE_PDF_PDFMINER_UPSTREAM_ENV"


def internal_install_core_pdf_facade() -> None:
    """Install facade-backed public pdfminer modules before pytest collection."""
    from core_pdf.api.compat import pdfminer as facade

    package = ModuleType("pdfminer")
    package.__path__ = []
    high_level = ModuleType("pdfminer.high_level")
    vars(high_level).update(
        extract_pages=facade.extract_pages,
        extract_text=facade.extract_text,
        extract_text_to_fp=facade.extract_text_to_fp,
    )
    layout = ModuleType("pdfminer.layout")
    for name in (
        "LAParams",
        "LTAnno",
        "LTChar",
        "LTImage",
        "LTItem",
        "LTPage",
        "LTText",
        "LTTextBox",
        "LTTextBoxHorizontal",
        "LTTextBoxVertical",
        "LTTextContainer",
        "LTTextLine",
        "LTTextLineHorizontal",
        "LTTextLineVertical",
    ):
        setattr(layout, name, getattr(facade, name))
    vars(package).update(high_level=high_level, layout=layout)
    sys.modules.update(
        {
            "pdfminer": package,
            "pdfminer.high_level": high_level,
            "pdfminer.layout": layout,
        }
    )


def internal_run_one(implementation: str, pytest_args: list[str]) -> int:
    """Run one implementation in the current process."""
    if implementation == "upstream" and not os.environ.get(UPSTREAM_ENV_MARKER):
        environment = os.environ.copy()
        environment[UPSTREAM_ENV_MARKER] = "1"
        command = [
            "uv",
            "run",
            "--with",
            str(PDFMINER_ROOT),
            "python",
            str(Path(__file__).resolve()),
            "--implementation",
            "upstream",
            "--",
            *pytest_args,
        ]
        return subprocess.run(command, cwd=ROOT, env=environment, check=False).returncode

    os.chdir(PDFMINER_ROOT)
    sys.path.insert(0, str(PDFMINER_ROOT))
    if implementation == "core-pdf":
        internal_install_core_pdf_facade()
    import pytest

    return pytest.main([str(UPSTREAM_TEST), *pytest_args])


def internal_run_both(pytest_args: list[str]) -> int:
    """Run both implementations in isolated child interpreters."""
    results: list[int] = []
    for implementation in ("upstream", "core-pdf"):
        print(f"\n=== pdfminer.six high-level tests: {implementation} ===", flush=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--implementation",
            implementation,
            "--",
            *pytest_args,
        ]
        results.append(subprocess.run(command, cwd=ROOT, check=False).returncode)
    return max(results, default=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation",
        choices=("upstream", "core-pdf", "both"),
        default="both",
    )
    arguments, pytest_args = parser.parse_known_args()
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    if arguments.implementation == "both":
        return internal_run_both(pytest_args)
    return internal_run_one(arguments.implementation, pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
