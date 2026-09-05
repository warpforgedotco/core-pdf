# SPDX-License-Identifier: AGPL-3.0-only
"""Resource references retain their identity regardless of dictionary storage.

These PDFs pass qpdf 12.3.2 --check. Poppler 26.07.0 pdftotext -raw emits One/Two,
and pdfminer.six 20260107 emits two separate three-character LTFigures for every
direct/indirect combination. Verified before adding the expected projections.
"""

import io

import pytest

from core_pdf import PdfDocument
from core_pdf.api.compat.pdfminer import LTFigure, extract_pages
from core_pdf.impl.spec.s_07_syntax.resolver import ObjectResolver
from core_pdf.impl.spec.s_09_fonts.ligatures import find_companion_font
from core_pdf.impl.types import PdfReference
from tests.helpers.pdf_bytes import one_page_pdf, stream_obj


def internal_resource_pdf(indirect_resources: bool, indirect_category: bool) -> bytes:
    category = b"9 0 R" if indirect_category else b"<< /One 6 0 R /Two 7 0 R >>"
    resources = b"<< /Font << /F1 5 0 R /Unused 10 0 R >> /XObject " + category + b" >>"
    forms = [
        stream_obj(
            b"BT /F1 12 Tf 10 " + y + b" Td (" + name + b") Tj ET",
            b"/Type /XObject /Subtype /Form /BBox [0 0 100 100] "
            b"/Resources << /Font << /F1 5 0 R >> >>",
        )
        for y, name in [(b"40", b"One"), (b"20", b"Two")]
    ]
    return one_page_pdf(
        b"/One Do /Two Do",
        resources=b"8 0 R" if indirect_resources else resources,
        extra_objects=[
            *forms,
            resources,
            b"<< /One 6 0 R /Two 7 0 R >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
        ],
    )


@pytest.mark.parametrize("indirect_resources", [False, True])
@pytest.mark.parametrize("indirect_category", [False, True])
def test_resource_storage_preserves_form_identity_and_separate_figures(
    indirect_resources: bool, indirect_category: bool
) -> None:
    pdf = internal_resource_pdf(indirect_resources, indirect_category)
    with PdfDocument(pdf) as document:
        program = document.pages[0].get_page_program()
        identifiers = [dict(run.provenance)["layout_form_id"] for run in program.runs]
        assert identifiers == [
            ((("ref", number, 0), (0.0, 0.0, 100.0, 100.0)),) for number in (6, 7)
        ]
        assert document.extract().text.split() == ["One", "Two"]

    page = next(extract_pages(io.BytesIO(pdf)))
    figures = [item for item in page if isinstance(item, LTFigure)]
    assert [len(list(figure)) for figure in figures] == [3, 3]


@pytest.mark.parametrize("indirect_resources", [False, True])
@pytest.mark.parametrize("indirect_category", [False, True])
def test_capture_resolves_requested_resources_without_loading_unused_fonts(
    monkeypatch: pytest.MonkeyPatch, indirect_resources: bool, indirect_category: bool
) -> None:
    resolved: list[int] = []
    resolve = ObjectResolver.internal_resolve_reference

    def track(self: ObjectResolver, reference: PdfReference) -> object:
        resolved.append(reference.object_number)
        return resolve(self, reference)

    with PdfDocument(internal_resource_pdf(indirect_resources, indirect_category)) as document:
        monkeypatch.setattr(ObjectResolver, "internal_resolve_reference", track)
        assert [run.text for run in document.pages[0].get_page_program().runs] == ["One", "Two"]

    assert 10 not in resolved


@pytest.mark.parametrize("indirect", [False, True])
@pytest.mark.parametrize("type3", [False, True])
def test_font_decoding_leaves_unused_sibling_and_type3_resources_unresolved(
    monkeypatch: pytest.MonkeyPatch, indirect: bool, type3: bool
) -> None:
    # qpdf --check and Poppler pdftotext -raw also verified these four PDFs:
    # the ordinary font emits Selected, and the Type 3 encoding emits A.
    if type3:
        resources = b"<< /Font << /Unused 8 0 R >> >>"
        font = (
            b"<< /Type /Font /Subtype /Type3 /FontBBox [0 0 500 500] "
            b"/FontMatrix [.001 0 0 .001 0 0] /FirstChar 65 /LastChar 65 /Widths [500] "
            b"/Encoding << /Type /Encoding /Differences [65 /A] >> "
            b"/CharProcs << /A 6 0 R >> /Resources "
            + (b"7 0 R" if indirect else resources)
            + b" >>"
        )
        pdf = one_page_pdf(
            b"BT /F1 12 Tf 10 40 Td (A) Tj ET",
            font=font,
            extra_objects=[
                stream_obj(b"500 0 d0 0 0 500 500 re f"),
                resources,
                b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
            ],
        )
        expected, unused_object = ["A"], 8
    else:
        category = b"<< /F1 5 0 R /Unused 7 0 R >>"
        pdf = one_page_pdf(
            b"BT /F1 12 Tf 10 40 Td (Selected) Tj ET",
            resources=b"<< /Font " + (b"6 0 R" if indirect else category) + b" >>",
            extra_objects=[
                category,
                b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
            ],
        )
        expected, unused_object = ["Selected"], 7

    resolved: list[int] = []
    resolve = ObjectResolver.internal_resolve_reference

    def track(self: ObjectResolver, reference: PdfReference) -> object:
        resolved.append(reference.object_number)
        return resolve(self, reference)

    with PdfDocument(pdf) as document:
        monkeypatch.setattr(ObjectResolver, "internal_resolve_reference", track)
        assert [run.text for run in document.pages[0].get_page_program().runs] == expected

    assert unused_object not in resolved


@pytest.mark.parametrize("indirect", [False, True])
def test_selected_companion_font_resolves_its_own_metrics(indirect: bool) -> None:
    # Both variants pass qpdf --check, and Poppler pdftotext emits f. The font's
    # metadata is identical; indirect storage must not disable width evidence.
    metrics = (
        b"/BaseFont 9 0 R /FirstChar 7 0 R /LastChar 7 0 R /Widths 8 0 R"
        if indirect
        else b"/BaseFont /Helvetica /FirstChar 102 /LastChar 102 /Widths [278]"
    )
    pdf = one_page_pdf(
        b"BT /F1 12 Tf 10 40 Td (f) Tj ET",
        font=b"<< /Type /Font /Subtype /Type1 " + metrics + b" >>",
        resources=b"<< /Font 6 0 R >>",
        extra_objects=[b"<< /F1 5 0 R >>", b"102", b"[10 0 R]", b"/Helvetica", b"278"],
    )
    with PdfDocument(pdf) as document:
        widths, starters, font_data = find_companion_font(
            document, document.pages[0].resources, "Helvetica", {"f"}
        )

    assert widths == {102: 278.0}
    assert starters == {"f": 278.0}
    assert font_data is None
