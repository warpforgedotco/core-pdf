# SPDX-License-Identifier: AGPL-3.0-only
"""Inherited page properties must not replace the referenced source dictionary."""

from __future__ import annotations

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.types import PdfReference
from tests.helpers.pdf_bytes import HELVETICA, assemble_pdf, stream_obj


def inherited_page_pdf(
    *,
    parent: bytes = b"/Parent 7 0 R",
    recover: bool = False,
    explicit: bool = False,
    page_extra: bytes = b"",
) -> bytes:
    media = b" /MediaBox [2 3 82 103] /Resources << /Font << /F1 6 0 R >> >>"
    display = b" /CropBox [5 5 70 90] /Rotate 90"
    return assemble_pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R /AcroForm << /Fields [4 0 R] >> "
            b"/Names << /Dests << /Names [(first) [3 0 R /Fit]] >> >> >>",
            b"<< /Type /Pages /Count 1 /Kids "
            + (b"[]" if recover else b"[7 0 R]")
            + (b"" if explicit else media)
            + b" >>",
            b"<< /Type /Page /Annots [4 0 R] /Contents 5 0 R "
            + parent
            + (media + display if explicit else b"")
            + b" "
            + page_extra
            + b" >>",
            b"<< /Type /Annot /Subtype /Widget /FT /Tx /T (name) /V (Ada) "
            b"/P 3 0 R /Rect [10 10 50 30] >>",
            stream_obj(b"0.2 0.4 0.8 rg 10 10 20 30 re f BT /F1 8 Tf 10 65 Td (Body) Tj ET"),
            HELVETICA,
            b"<< /Type /Pages /Parent 2 0 R /Count 1 /Kids [3 0 R]"
            + (b"" if explicit else display)
            + b" >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
        ]
    )


@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize(
    "parent",
    [b"/Parent 7 0 R", b"", b"/Parent null", b"/Parent 42", b"/Parent 99 0 R", b"/Parent 3 0 R"],
    ids=["valid", "missing", "null", "non-dictionary", "dangling", "cyclic"],
)
def test_inherited_page_keeps_identity_and_resolves_fields_and_destinations(
    parent: bytes, recover: bool
) -> None:
    with PdfDocument(inherited_page_pdf(parent=parent, recover=recover)) as document:
        original = document.resolve(PdfReference(3))
        assert isinstance(original, dict)
        source_values = dict(original)
        page = document.pages[0]

        assert page.page_dict is original
        assert document.build_page_dicts()[0] is original
        assert document.page_index_for(original) == 0
        assert document.named_destinations()["first"].page_index == 0
        assert [field.value_text for field in page.get_fields()] == ["Ada"]
        assert [field.record.value_text for field in document.extract_form_fields()] == ["Ada"]
        assert page.media_box == (2.0, 3.0, 82.0, 103.0)
        assert page.crop_box == (5.0, 5.0, 70.0, 90.0)
        assert page.rotation == 90
        assert "Body" in "".join(run.text for run in page.chars)
        assert original == source_values
        assert "MediaBox" not in original
        assert "Resources" not in original
        assert document.page_tree_was_recovered is recover


@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize("parent", [b"/Parent 7 0 R", b""])
def test_inherited_page_renders_like_the_equivalent_explicit_page(
    parent: bytes, recover: bool
) -> None:
    with (
        PdfDocument(inherited_page_pdf(parent=parent, recover=recover)) as inherited,
        PdfDocument(inherited_page_pdf(explicit=True)) as explicit,
    ):
        actual = inherited.pages[0].render().rasterize()
        expected = explicit.pages[0].render().rasterize()

        assert (actual.width, actual.height) == (expected.width, expected.height)
        assert bytes(actual.pixels) == bytes(expected.pixels)


@pytest.mark.parametrize("recover", [False, True])
def test_explicit_leaf_properties_override_inheritance_without_mutating_source(
    recover: bool,
) -> None:
    with PdfDocument(
        inherited_page_pdf(
            recover=recover,
            page_extra=b"/MediaBox [3 4 63 104] /Rotate 270 /Resources << /Font << /F1 8 0 R >> >>",
        )
    ) as document:
        original = document.resolve(PdfReference(3))
        page = document.pages[0]

        assert page.page_dict is original
        assert page.media_box == (3.0, 4.0, 63.0, 104.0)
        assert page.rotation == 270
        assert page.crop_box == (5.0, 5.0, 70.0, 90.0)
        fonts = document.resolver.resolve_dict(page.resources.get("Font"))
        assert fonts is not None
        font = document.resolver.resolve_dict(fonts["F1"])
        assert font is not None
        assert document.resolver.resolve_name(font.get("BaseFont")) == "Courier"
