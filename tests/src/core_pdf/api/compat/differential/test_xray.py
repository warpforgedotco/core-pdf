import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from .support import XRAY_ROOT, call_pair, differential_pdfs, pdf_id

real_pymupdf = pytest.importorskip("pymupdf")
pytestmark = pytest.mark.compat_differential


@pytest.fixture(scope="module")
def real_xray() -> Iterator[Any]:
    """Import the reference x-ray package once, against the real PyMuPDF."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.syspath_prepend(str(XRAY_ROOT))
        monkeypatch.setitem(sys.modules, "fitz", real_pymupdf)
        for module_name in tuple(sys.modules):
            if module_name == "xray" or module_name.startswith("xray."):
                sys.modules.pop(module_name)
        yield pytest.importorskip("xray")


@pytest.mark.parametrize("pdf_path", differential_pdfs("xray"), ids=pdf_id)
def test_matches_real_library_on_fixture_corpus(real_xray: Any, pdf_path: Path) -> None:
    from core_pdf.api.compat.xray import inspect

    pair = call_pair(lambda: real_xray.inspect(pdf_path), lambda: inspect(pdf_path))
    if pair is not None:
        assert pair[1] == pair[0]
