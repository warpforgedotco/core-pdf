from pathlib import Path

import pytest

from core_pdf.api.compat.llamaindex import load_data
from core_pdf.api.compat.llamaindex._operator_text import (
    internal_difference_text,
    internal_TextState,
)
from core_pdf.impl.engine.parse import layout as native_layout
from core_pdf.impl.objects import PdfName


def test_unknown_adobe_glyph_names_remain_visible() -> None:
    assert internal_difference_text("i255", 0) == "/i255"
    assert internal_difference_text("17", 4) == "/17"


def test_notdef_uses_the_adobe_missing_glyph_marker() -> None:
    assert internal_difference_text(".notdef", 10) == "□"


def test_name_text_operand_preserves_its_pdf_lexical_form() -> None:
    state = internal_TextState({})

    state.show_name(PdfName.of("BOGUS"))

    assert state.text == "/BOGUS"


def test_load_data_does_not_call_native_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_native_layout(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("compat projection called the native layout pipeline")

    monkeypatch.setattr(native_layout, "layout_blocks", reject_native_layout)
    monkeypatch.setattr(native_layout, "layout_blocks_with_evidence", reject_native_layout)

    documents = load_data(Path("tests/fixtures/pypdf/resources/hello-world.pdf"))

    assert documents


@pytest.mark.parametrize(
    ("path", "page_index", "expected_text"),
    [
        (
            "tests/fixtures/pdfminer.six/samples/nonfree/naacl06-shinyama.pdf",
            6,
            "pairscore i",
        ),
        (
            "tests/fixtures/unstructured/example-docs/language-docs/fr_olap.pdf",
            7,
            "lookup(Pepsi )",
        ),
        (
            "tests/fixtures/unstructured/example-docs/pdf/layout-parser-paper-fast.pdf",
            1,
            "challenges, LayoutParser",
        ),
    ],
)
def test_load_data_preserves_pypdf_run_boundary_spacing(
    path: str,
    page_index: int,
    expected_text: str,
) -> None:
    documents = load_data(Path(path))

    assert expected_text in documents[page_index].text
