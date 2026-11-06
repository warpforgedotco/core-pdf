"""Record every file pypdf's tests download, by name and URL.

scripts/fetch_pypdf_test_files.py --record loads this plugin into a run of pypdf's whole
suite on the real pypdf and writes the files it names to PYPDF_FILE_RECORD, as JSON. The
tests call tests.get_data_from_url, which this wraps before any test module imports it.
"""

import importlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

RECORDED: dict[str, str | None] = {}


def pytest_configure(config: pytest.Config) -> None:
    tests: Any = importlib.import_module("tests")  # pypdf's, from its checkout

    original = tests.get_data_from_url

    def recording(*, url: str | None = None, name: str) -> Any:
        if url is None or not url.startswith("file://"):
            RECORDED[name] = url or tests._get_urls_of_example_files().get(name)  # noqa: SLF001
        return original(url=url, name=name)

    tests.get_data_from_url = recording


def pytest_unconfigure(config: pytest.Config) -> None:
    tests: Any = importlib.import_module("tests")  # pypdf's, from its checkout

    # pypdf's CI fetches these before the tests, so include them even if no test ran one.
    for name, url in tests._get_urls_of_example_files().items():  # noqa: SLF001
        RECORDED.setdefault(name, url)
    Path(os.environ["PYPDF_FILE_RECORD"]).write_text(json.dumps(RECORDED, sort_keys=True))
