from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

import pytest

FIXTURES_ROOT = Path("tests/fixtures")
ALL_PDFS = tuple(sorted(FIXTURES_ROOT.rglob("*.pdf")))
XRAY_ROOT = Path("tests/fixtures/x-ray").resolve()


def open_pair(
    stack: ExitStack,
    expected_factory: Callable[[], Any],
    actual_factory: Callable[[], Any],
) -> tuple[Any, Any] | None:
    try:
        expected = stack.enter_context(expected_factory())
    except Exception:
        try:
            stack.enter_context(actual_factory())
        except Exception:
            return None
        pytest.fail("reference rejected the PDF but compat accepted it")
    try:
        actual = stack.enter_context(actual_factory())
    except Exception as error:
        pytest.fail(f"reference accepted the PDF but compat rejected it: {error!r}")
    return expected, actual


def call_pair(
    expected_factory: Callable[[], Any],
    actual_factory: Callable[[], Any],
) -> tuple[Any, Any] | None:
    try:
        expected = expected_factory()
    except Exception:
        try:
            actual_factory()
        except Exception:
            return None
        pytest.fail("reference rejected the PDF but compat accepted it")
    try:
        actual = actual_factory()
    except Exception as error:
        pytest.fail(f"reference accepted the PDF but compat rejected it: {error!r}")
    return expected, actual


def metadata(value: Any) -> dict[str, str]:
    return {
        str(key).lstrip("/"): str(item)
        for key, item in value.items()
        if str(key) not in {"info", "xmp"}
    }


def words(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: round(value, 5) if isinstance(value, float) else value for key, value in word.items()}
        for word in values
    ]
