# SPDX-License-Identifier: AGPL-3.0-only

"""Check that every module of core-pdf and its facade packages imports, lazy imports included.

A `lazy import` or `lazy from ... import` of a module or name that does not exist raises only
when the name is first used, so importing a module proves nothing about its lazy imports.
This imports every module under the given packages and then reads every name in each
module's namespace, which resolves the lazy ones, and fails on the first error per module.
It catches an import left pointing at a moved or renamed module.

uv run --all-packages python scripts/check_impl_imports.py [package ...]
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import traceback

PACKAGES = ("core_pdf", "core_pdf_compat", "core_pdf_ocr")
# Vendored code is imported as upstream ships it.
SKIP_PREFIXES = ("core_pdf._vendor.", "core_pdf.impl.adobe_fonts._vendor.")


def module_names(package_name: str) -> list[str]:
    package = importlib.import_module(package_name)
    return [
        package_name,
        *(
            info.name
            for info in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}.")
            if not info.name.startswith(SKIP_PREFIXES) and not info.name.endswith("__main__")
        ),
    ]


def check_module(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
        for attribute in list(vars(module)):
            getattr(module, attribute)
    except Exception:
        return f"{name}:\n{traceback.format_exc(limit=-2)}"
    return None


def main(argv: list[str]) -> int:
    packages = argv[1:] or PACKAGES
    failures: list[str] = []
    count = 0
    for package in packages:
        for name in module_names(package):
            count += 1
            failure = check_module(name)
            if failure is not None:
                failures.append(failure)
    if failures:
        print("\n".join(failures))
        print(f"{len(failures)} of {count} modules failed")
        return 1
    print(f"{count} modules imported, lazy imports resolved")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
