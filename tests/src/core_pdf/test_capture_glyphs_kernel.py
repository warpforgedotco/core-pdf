import math
from typing import Any

import pytest

import core_pdf.impl.capture_glyphs as capture_glyphs_module
import core_pdf.impl.capture_recording as capture_recording_module
from core_pdf import PdfDocument
from core_pdf.impl.glyphs import GlyphObservation
from core_pdf_cythonized import SlotLayout
from tests.src.core_pdf.extraction_snapshot import FIXTURES

# Horizontal text, CID fonts, rotated text and ligature splits between them.
DOCUMENTS = (
    "llama_index/docs/examples/data/10k/lyft_2021.pdf",
    "pdfminer.six/samples/nonfree/i1040nr.pdf",
    "SCORE-Bench/src/fhhd0346-p009.pdf",
    "PyMuPDF/tests/resources/test_2907.pdf",
    "pdfplumber/tests/pdfs/issue-71-duplicate-chars-2.pdf",
)


def same(left: Any, right: Any) -> bool:
    if type(left) is float and type(right) is float:
        return (math.isnan(left) and math.isnan(right)) or (
            left == right and math.copysign(1.0, left) == math.copysign(1.0, right)
        )
    if type(left) is tuple and type(right) is tuple:
        return len(left) == len(right) and all(map(same, left, right, strict=True))
    if hasattr(left, "__slots__") and not isinstance(left, (int, float, str, bytes)):
        return left is right
    return type(left) is type(right) and left == right


@pytest.mark.parametrize("document", DOCUMENTS)
def test_compiled_capture_matches_the_python_loop(
    document: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = FIXTURES / document
    if not path.is_file() or path.stat().st_size == 0:
        pytest.skip("fixture corpus not initialized")
    compiled = capture_glyphs_module.capture_horizontal_glyphs
    capture = capture_glyphs_module.capture_glyphs
    compared = 0

    def both(*arguments: Any) -> Any:
        nonlocal compared
        fast = capture(*arguments)
        monkeypatch.setattr(capture_glyphs_module, "capture_horizontal_glyphs", lambda *_: None)
        try:
            slow = capture(*arguments)
        finally:
            monkeypatch.setattr(capture_glyphs_module, "capture_horizontal_glyphs", compiled)
        assert fast.cluster_count == slow.cluster_count
        assert len(fast.glyphs) == len(slow.glyphs)
        assert len(fast.clusters) == len(slow.clusters)
        for have, want in zip(fast.glyphs, slow.glyphs, strict=True):
            for name in GlyphObservation.__slots__:
                assert same(getattr(have, name), getattr(want, name)), name
        for name in ("started", "advance", "ink", "confidence"):
            assert same(getattr(fast.geometry, name), getattr(slow.geometry, name)), name
        compared += len(fast.glyphs)
        return fast

    monkeypatch.setattr(capture_recording_module, "capture_glyphs", both)
    with PdfDocument(path.read_bytes()) as pdf:
        for page in pdf.pages[:2]:
            page.extract()
    assert compared


def test_slot_layout_refuses_attributes_that_are_not_slots() -> None:
    class Plain:
        value = 1

    with pytest.raises(TypeError):
        SlotLayout(Plain, ("value",))
