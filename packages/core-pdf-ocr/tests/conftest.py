"""Tesseract setup belongs exclusively to the OCR test suite."""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess


def internal_export_tessdata_prefix() -> None:
    """Resolve English language data once before recognition tests run."""
    if os.environ.get("TESSDATA_PREFIX"):
        return
    executable = shutil.which("tesseract")
    if executable is None:
        return
    try:
        completed = subprocess.run(
            [executable, "--list-langs"], capture_output=True, check=False, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return
    match = re.search(
        r'List of available languages in "([^"]+)"', completed.stdout + completed.stderr
    )
    if match is None:
        return
    tessdata = pathlib.Path(match.group(1)).expanduser()
    if (tessdata / "eng.traineddata").is_file():
        os.environ["TESSDATA_PREFIX"] = str(tessdata.resolve())


def pytest_configure() -> None:
    internal_export_tessdata_prefix()
