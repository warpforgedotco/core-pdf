from __future__ import annotations

import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Literal

import pytest

FIXTURES_ROOT = Path(__file__).resolve().parents[4] / "fixtures"
XRAY_ROOT = FIXTURES_ROOT / "x-ray"
FULL_ENV = "CORE_PDF_COMPAT_DIFFERENTIAL_FULL"
FacadeName = Literal["llamaindex", "pdfplumber", "pikepdf", "pypdf", "unstructured", "xray"]
FACADE_ROOTS: dict[FacadeName, tuple[str, ...]] = {
    "llamaindex": ("llama_index",),
    "pdfplumber": ("pdfplumber",),
    "pikepdf": ("pikepdf",),
    "pypdf": ("pypdf",),
    "unstructured": ("unstructured",),
    "xray": ("x-ray",),
}
XRAY_EXTRA_PATHS = (
    "PyMuPDF/tests/resources/test-707448.pdf",
    "PyMuPDF/tests/resources/test_1824.pdf",
    "PyMuPDF/tests/resources/test_2596.pdf",
    "PyMuPDF/tests/resources/test_2957_1.pdf",
    "PyMuPDF/tests/resources/test_2957_2.pdf",
    "PyMuPDF/tests/resources/test_4079_after.pdf",
    "PyMuPDF/tests/resources/test_4079_after_1.25.pdf",
)
FULL_ONLY_PATHS = frozenset(
    {
        "llama_index/docs/examples/data/10k/lyft_2021.pdf",
        "llama_index/docs/examples/data/10k/uber_2021.pdf",
        "llama_index/docs/examples/data/10q/uber_10q_june_2022.pdf",
        "llama_index/docs/examples/data/10q/uber_10q_march_2022.pdf",
        "llama_index/docs/examples/data/10q/uber_10q_sept_2022.pdf",
        "pypdf/resources/issue-604.pdf",
        "unstructured/example-docs/pdf/DA-619p.pdf",
        "unstructured/example-docs/pdf/failure-after-repair.pdf",
        "unstructured/example-docs/pdf/pdf2image-memory-error-test-400p.pdf",
    }
)


def differential_pdfs(facade: FacadeName) -> tuple[Path, ...]:
    """Return the facade-owned corpus, or the exhaustive corpus when explicitly requested."""
    if os.environ.get(FULL_ENV) == "1":
        return tuple(sorted(path for path in FIXTURES_ROOT.rglob("*.pdf") if path.is_file()))

    paths = {
        path
        for root_name in FACADE_ROOTS[facade]
        for path in (FIXTURES_ROOT / root_name).rglob("*.pdf")
        if path.is_file()
    }
    if facade == "xray":
        paths.update(FIXTURES_ROOT / relative for relative in XRAY_EXTRA_PATHS)
    return tuple(
        sorted(
            path
            for path in paths
            if path.relative_to(FIXTURES_ROOT).as_posix() not in FULL_ONLY_PATHS
        )
    )


def pdf_id(path: Path) -> str:
    return path.relative_to(FIXTURES_ROOT).as_posix()


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
