"""Deselect upstream tests whose source mentions a name in CORE_PDF_COMPAT_OUT_OF_SCOPE."""

import inspect
import os
import re

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    names = [name for name in os.environ.get("CORE_PDF_COMPAT_OUT_OF_SCOPE", "").split(",") if name]
    if not names:
        return
    pattern = re.compile(rf"\b({'|'.join(map(re.escape, names))})\b")
    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        function = getattr(item, "function", None)
        source = inspect.getsource(function) if function is not None else ""
        (deselected if pattern.search(source) else selected).append(item)
    config.hook.pytest_deselected(items=deselected)
    items[:] = selected
