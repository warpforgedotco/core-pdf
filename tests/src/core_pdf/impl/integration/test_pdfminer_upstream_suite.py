from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).parents[4]
PDFMINER_ROOT = TESTS_DIR / "fixtures" / "pdfminer.six"


def test_vendored_pdfminer_passes_upstream_test_suite() -> None:
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--pdfminer-core-compat",
            "tests",
            "-q",
        ],
        cwd=PDFMINER_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
