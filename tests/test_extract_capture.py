from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf import PdfDocument
from core_pdf.impl._impl.extract.capture import (
    capture_page,
    internal_apply_structure_actual_text,
    internal_extractable_runs,
    internal_layout_bbox_for_run,
)
from core_pdf.impl._impl.extract.pipeline import internal_PageExtraction
from core_pdf.impl._impl.model.runs import TextRun
from tests.helpers.extract_fakes import text_run
from tests.helpers.paths import SCORE_BENCH
from tests.helpers.pdf_bytes import one_page_pdf, stream_obj


def run(
    text: str,
    *,
    depth: int = 0,
    clip: tuple[float, float, float, float] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> TextRun:
    provenance = (("clip_bbox", clip),) if clip is not None else ()
    x0, y0, x1, y1 = bbox or clip or (0.0, 0.0, 100.0, 100.0)
    return text_run(text, x0, y0, x1, y1, xobject_depth=depth, provenance=provenance)


def test_capture_discards_duplicate_nested_text_layer() -> None:
    text = " ".join(f"token{index}" for index in range(30))
    page_run = run(text)
    nested_run = run(text, depth=1)
    distinct_nested_run = run(" ".join(f"other{index}" for index in range(30)), depth=2)

    assert internal_extractable_runs(cast(Any, (page_run, nested_run, distinct_nested_run))) == (
        page_run,
        distinct_nested_run,
    )


@pytest.mark.parametrize("variant", ["sibling", "mixed", "translated", "extended"])
def test_capture_preserves_distinct_content_beside_duplicate_form_text(variant: str) -> None:
    repeated = " ".join(f"word{index}" for index in range(24))
    unique = "UNIQUE FORM CONTENT MUST SURVIVE"

    def content(text: str, y: int = 700) -> bytes:
        return f"BT /F1 6 Tf 10 {y} Td ({text}) Tj ET".encode()

    form_metadata = (
        b"/Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >>"
    )
    forms = [stream_obj(content(repeated), form_metadata)]
    expected = (repeated, unique)
    match variant:
        case "sibling":
            invocation = b"/Dup Do /Other Do"
            forms.append(stream_obj(content(unique, 600), form_metadata))
        case "mixed":
            invocation = b"/Other Do"
            forms.append(stream_obj(content(repeated) + b" " + content(unique, 600), form_metadata))
        case "translated":
            invocation = b"/Dup Do q 1 0 0 1 0 -100 cm /Dup Do Q"
            expected = (repeated, repeated)
        case "extended":
            invocation = b"/Other Do"
            forms.append(stream_obj(content(repeated + " " + unique), form_metadata))
            # Poppler 26.07.0 pdftotext -layout preserves the complete overlay once.
            expected = (repeated + " " + unique,)
        case _:
            raise AssertionError(variant)
    pdf = one_page_pdf(
        content(repeated) + b" " + invocation,
        resources=(b"<< /Font << /F1 5 0 R >> /XObject << /Dup 6 0 R /Other 7 0 R >> >>"),
        extra_objects=forms,
    )

    with PdfDocument(pdf) as document:
        captured = capture_page(document.pages[0])
        assert captured.observations.text == expected
        extracted = document.extract().text
        if variant == "translated":
            assert extracted.count("word0") == 2
        else:
            assert unique in extracted


def test_capture_deduplicates_parent_form_across_nested_drawing() -> None:
    first = " ".join(f"word{index}" for index in range(12))
    second = " ".join(f"word{index}" for index in range(12, 24))

    def content(text: str, y: int) -> bytes:
        return f"BT /F1 6 Tf 10 {y} Td ({text}) Tj ET".encode()

    form_metadata = b"/Type /XObject /Subtype /Form /BBox [0 0 612 792] "
    page_content = content(first, 700) + b" " + content(second, 680)
    form_content = content(first, 700) + b" /Inner Do " + content(second, 680)
    pdf = one_page_pdf(
        page_content + b" /Outer Do",
        resources=b"<< /Font << /F1 5 0 R >> /XObject << /Outer 6 0 R >> >>",
        extra_objects=(
            stream_obj(
                form_content,
                form_metadata + b"/Resources << /Font << /F1 5 0 R >> "
                b"/XObject << /Inner 7 0 R >> >>",
            ),
            stream_obj(b"0 0 5 5 re f", form_metadata),
        ),
    )

    with PdfDocument(pdf) as document:
        assert capture_page(document.pages[0]).observations.text == (first, second)
        assert document.extract().text.count("word0") == 1


def test_capture_discards_duplicate_alternate_clip_layer() -> None:
    text = " ".join(f"token{index}" for index in range(30))
    page_run = run(text, clip=(0.0, 0.0, 100.0, 100.0))
    alternate_run = run(text, clip=(10.0, 10.0, 90.0, 90.0))
    distinct_clipped_run = run(
        " ".join(f"other{index}" for index in range(30)),
        clip=(20.0, 20.0, 80.0, 80.0),
    )

    assert internal_extractable_runs(
        cast(Any, (page_run, alternate_run, distinct_clipped_run))
    ) == (page_run, distinct_clipped_run)


def test_capture_preserves_similar_text_in_a_separate_clipped_region() -> None:
    text = " ".join(f"security-handler-token-{index}" for index in range(30))
    page_run = run(
        text,
        clip=(0.0, 0.0, 100.0, 100.0),
        bbox=(0.0, 0.0, 100.0, 20.0),
    )
    table_cell_run = run(
        text,
        clip=(0.0, 50.0, 100.0, 100.0),
        bbox=(0.0, 50.0, 100.0, 70.0),
    )

    assert internal_extractable_runs(cast(Any, (page_run, table_cell_run))) == (
        page_run,
        table_cell_run,
    )


def test_capture_repairs_font_wide_vertical_metrics_for_layout() -> None:
    text_run = TextRun(
        "Permission row",
        10.0,
        -24.0,
        110.0,
        31.0,
        0.0,
        0.0,
        10.0,
        4.0,
        0,
        0,
        0,
        advance_bbox=(10.0, -24.0, 110.0, 31.0),
        ink_bbox=(10.0, -0.1, 110.0, 6.7),
        baseline=(10.0, 0.0, 110.0, 0.0),
    )

    assert internal_layout_bbox_for_run(text_run) == (10.0, -2.0, 110.0, 8.0)


def test_capture_does_not_rewrite_vertical_text_layout_geometry() -> None:
    text_run = TextRun(
        "Vertical",
        10.0,
        -24.0,
        20.0,
        80.0,
        0.0,
        0.0,
        10.0,
        4.0,
        0,
        0,
        0,
        is_vertical=True,
        advance_bbox=(10.0, -24.0, 20.0, 80.0),
        ink_bbox=(10.0, 0.0, 20.0, 70.0),
        baseline=(15.0, 0.0, 15.0, 70.0),
    )

    assert internal_layout_bbox_for_run(text_run) == (10.0, -24.0, 20.0, 80.0)


def test_structure_actual_text_replaces_mcid_text_before_routing() -> None:
    source = TextRun(
        "broken",
        0.0,
        0.0,
        10.0,
        10.0,
        0.0,
        0.0,
        12.0,
        4.0,
        0,
        0,
        0,
        provenance=(("mcid", 0),),
    )
    page = SimpleNamespace(structure=(SimpleNamespace(actual_text="correct"),))

    result = internal_apply_structure_actual_text(page, (source,))

    assert result[0].text == "correct"
    assert ("unicode_source", "structure_actual_text") in result[0].provenance


def test_page_extraction_owns_one_capture_without_page_cache() -> None:
    fixture = SCORE_BENCH / "Employee_Health_Benefits_Assess-p006.pdf"
    with PdfDocument.open(fixture) as document:
        page = document.pages[0]
        program = page.get_page_program()
        extraction = internal_PageExtraction(page)
        captured = extraction.capture

    assert captured.program is not program
    assert captured.evidence.image_count == 0
