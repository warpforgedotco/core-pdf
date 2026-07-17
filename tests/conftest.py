# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import importlib
import sys

import pytest

PDFMINER_MODULES = (
    "ascii85",
    "cmapdb",
    "converter",
    "high_level",
    "layout",
    "pdffont",
    "pdfinterp",
    "pdfpage",
    "pdftypes",
    "psexceptions",
    "psparser",
    "settings",
    "utils",
)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("pdfminer compatibility")
    group.addoption(
        "--pdfminer-core-compat",
        action="store_true",
        help="Run PDFMiner tests against core_pdf.integrations.pdfminer.six.",
    )


def pytest_configure(config: pytest.Config) -> None:
    if not config.getoption("--pdfminer-core-compat"):
        return

    already_imported = sorted(
        name for name in sys.modules if name == "pdfminer" or name.startswith("pdfminer.")
    )
    if already_imported:
        raise pytest.UsageError(
            "PDFMiner was imported before compatibility aliases were installed: "
            + ", ".join(already_imported)
        )

    compat_package = importlib.import_module("core_pdf.integrations.pdfminer.six")
    sys.modules["pdfminer"] = compat_package

    for module_name in PDFMINER_MODULES:
        compat_name = f"core_pdf.integrations.pdfminer.six.{module_name}"
        module = importlib.import_module(compat_name)
        sys.modules[f"pdfminer.{module_name}"] = module
        setattr(compat_package, module_name, module)
