import sys
from pathlib import Path

import pytest

from .support import XRAY_ROOT, call_pair, differential_pdfs, pdf_id

real_pymupdf = pytest.importorskip("pymupdf")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("pdf_path", differential_pdfs("xray"), ids=pdf_id)
def test_matches_real_library_on_fixture_corpus(
    monkeypatch: pytest.MonkeyPatch,
    pdf_path: Path,
) -> None:
    monkeypatch.syspath_prepend(str(XRAY_ROOT))
    monkeypatch.setitem(sys.modules, "fitz", real_pymupdf)
    for module_name in tuple(sys.modules):
        if module_name == "xray" or module_name.startswith("xray."):
            sys.modules.pop(module_name)
    real_xray = pytest.importorskip("xray")
    from core_pdf.api.compat.xray import inspect

    pair = call_pair(lambda: real_xray.inspect(pdf_path), lambda: inspect(pdf_path))
    if pair is not None:
        assert pair[1] == pair[0]
