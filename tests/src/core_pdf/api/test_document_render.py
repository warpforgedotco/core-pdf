# SPDX-License-Identifier: AGPL-3.0-only

import pytest

from core_pdf.impl.render.model import RenderOptions
from core_pdf.impl.spec.s_07_document.records import RawAnnotation, RawFormField
from tests.helpers.pdf_bytes import one_page_pdf, open_pdf, stream_obj


@pytest.mark.parametrize("include_layers", [False, True])
@pytest.mark.parametrize("include_annotations", [False, True])
def test_render_reuses_resolved_page_records(
    monkeypatch: pytest.MonkeyPatch, include_layers: bool, include_annotations: bool
) -> None:
    with open_pdf(one_page_pdf(b"1 0 0 rg 0 0 4 4 re f", media_box=(0, 0, 4, 4))) as document:
        page = document.pages[0]
        calls = {"fields": 0, "annotations": 0}

        def fields() -> list[RawFormField]:
            calls["fields"] += 1
            return []

        def annotations() -> list[RawAnnotation]:
            calls["annotations"] += 1
            return []

        monkeypatch.setattr(page, "get_fields", fields)
        monkeypatch.setattr(page, "get_annotations", annotations)
        image = page.render(
            RenderOptions(include_layers=include_layers, include_annotations=include_annotations)
        ).rasterize()

    assert calls == {"fields": int(include_layers), "annotations": int(include_annotations)}
    assert tuple(image.array()[1, 1]) == (255, 0, 0, 255)


def test_render_retains_page_content_with_malformed_form_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with open_pdf(one_page_pdf(b"1 0 0 rg 0 0 4 4 re f", media_box=(0, 0, 4, 4))) as document:
        page = document.pages[0]

        def malformed_fields() -> list[RawFormField]:
            raise ValueError("invalid AcroForm field entry")

        monkeypatch.setattr(page, "get_fields", malformed_fields)
        image = page.render().rasterize()

    assert tuple(image.array()[1, 1]) == (255, 0, 0, 255)


def non_array_appearance_pdf() -> bytes:
    return one_page_pdf(
        b"",
        media_box=(0, 0, 4, 4),
        page_extra=b"/Annots 6 0 R",
        extra_objects=(
            b"<< /Type /Annot /Subtype /Square /Rect [0 0 4 4] /AP << /N 7 0 R >> >>",
            stream_obj(
                b"1 0 0 rg 0 0 4 4 re f",
                b"/Type /XObject /Subtype /Form /BBox [0 0 4 4]",
            ),
        ),
    )


def test_render_keeps_tolerant_appearance_when_recovered_metadata_is_empty() -> None:
    pdf = non_array_appearance_pdf().split(b"xref\n", 1)[0]
    pdf += b"trailer << /Root 1 0 R >>\n%%EOF\n"

    with open_pdf(pdf) as document:
        assert document.recovery_enabled
        page = document.pages[0]
        assert page.get_annotations() == []

        image = page.render().rasterize()

    assert tuple(image.array()[1, 1]) == (255, 0, 0, 255)


def test_render_does_not_swallow_strict_enabled_annotation_errors() -> None:
    with open_pdf(non_array_appearance_pdf()) as document:
        page = document.pages[0]
        with pytest.raises(ValueError, match="invalid page Annots array"):
            page.render()

        # Disabling annotations bypasses their strict metadata read and paint.
        image = page.render(RenderOptions(include_annotations=False)).rasterize()

    assert image.array()[:, :, 3].max() == 0
