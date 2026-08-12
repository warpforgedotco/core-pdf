#!/usr/bin/env python3
"""Run the vendored x-ray tests with core-pdf's local inspect facade."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XRAY_ROOT = ROOT / "tests" / "fixtures" / "x-ray"


# Upstream implementation tests that call x-ray's private PyMuPDF helpers
# directly (not the patched public ``xray.inspect``). They exercise upstream
# x-ray against the installed PyMuPDF, fail with current PyMuPDF releases even
# without core-pdf involved, and are not compatibility signal. Verified
# 2026-08-10: core-pdf's facade returns the expected 3 findings for
# rectangles_yes.pdf while upstream get_bad_redactions returns 0 standalone.
UPSTREAM_ONLY_DESELECTS = (
    "tests/test_utils.py::IntegrationTest::test_bad_redactions_on_single_page",
)


def main() -> int:
    """Patch x-ray's public inspect function and run its pytest suite."""
    sys.path.insert(0, str(XRAY_ROOT))
    import pytest
    import xray  # ty: ignore[unresolved-import]

    from core_pdf.api.v0.compat import inspect_xray

    xray.inspect = inspect_xray  # type: ignore[assignment]
    xray_base = XRAY_ROOT
    with contextlib.suppress(ValueError):
        xray_base = XRAY_ROOT.relative_to(Path.cwd())
    deselects = [f"--deselect={xray_base / test_id}" for test_id in UPSTREAM_ONLY_DESELECTS]
    return pytest.main([str(XRAY_ROOT / "tests"), *deselects, *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
