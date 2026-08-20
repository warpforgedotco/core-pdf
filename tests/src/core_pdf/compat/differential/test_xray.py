import sys
from pathlib import Path

import pytest

from .support import ALL_PDFS, XRAY_ROOT, call_pair


@pytest.mark.parametrize("pdf_path", ALL_PDFS, ids=lambda path: path.name)
def test_matches_real_library_on_all_fixture_pdfs(
    monkeypatch: pytest.MonkeyPatch,
    pdf_path: Path,
) -> None:
    monkeypatch.syspath_prepend(str(XRAY_ROOT))
    real_pymupdf = pytest.importorskip("pymupdf")
    monkeypatch.setitem(sys.modules, "fitz", real_pymupdf)
    for module_name in tuple(sys.modules):
        if module_name == "xray" or module_name.startswith("xray."):
            sys.modules.pop(module_name)
    real_xray = pytest.importorskip("xray")
    from core_pdf.api.compat.xray import inspect

    pair = call_pair(lambda: real_xray.inspect(pdf_path), lambda: inspect(pdf_path))
    if pair is not None:
        assert pair[1] == pair[0]
