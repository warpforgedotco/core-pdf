#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XRAY_ROOT = ROOT / "tests" / "fixtures" / "x-ray"


UPSTREAM_ONLY_DESELECTS = (
    "tests/test_utils.py::IntegrationTest::test_bad_redactions_on_single_page",
)


def main() -> int:
    sys.path.insert(0, str(XRAY_ROOT))
    import pytest
    import xray  # ty: ignore[unresolved-import]

    from core_pdf.api.compat import inspect_xray

    xray.inspect = inspect_xray  # type: ignore[assignment]
    xray_base = XRAY_ROOT
    with contextlib.suppress(ValueError):
        xray_base = XRAY_ROOT.relative_to(Path.cwd())
    deselects = [f"--deselect={xray_base / test_id}" for test_id in UPSTREAM_ONLY_DESELECTS]
    return pytest.main([str(XRAY_ROOT / "tests"), *deselects, *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
