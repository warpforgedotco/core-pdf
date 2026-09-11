# SPDX-License-Identifier: AGPL-3.0-only
"""Version-specific color constraints retain shared color calculations."""

from __future__ import annotations

import pytest

from core_pdf_spec.exceptions import PdfUnsupportedError
from core_pdf_spec.s_08_graphics.color import indexed_color_components
from core_pdf_spec.s_08_graphics.color_spec import parse_color_space
from core_pdf_spec.standards import PdfVersion, SemanticContext
from core_pdf_spec.types import PdfName


def internal_indexed_tint_space(kind: str) -> list[object]:
    names: object = PdfName.of("Ink") if kind == "Separation" else [PdfName.of("Ink")]
    tint = {"FunctionType": 2, "Domain": [0, 1], "N": 1, "C0": [0], "C1": [1]}
    return ["Indexed", [kind, names, "DeviceGray", tint], 1, b"\0\xff"]


@pytest.mark.parametrize("kind", ["Separation", "DeviceN"])
@pytest.mark.parametrize("version", [PdfVersion(1, 1), PdfVersion(1, 2)])
def test_indexed_rejects_tint_bases_before_pdf_13(kind: str, version: PdfVersion) -> None:
    # Adobe PDF Reference 1.3, 4.5.5, pp. 181-182 explicitly distinguishes
    # permitted Indexed bases from the error prescribed in PDF 1.2.
    with pytest.raises(ValueError, match="Indexed.*bases require PDF 1.3"):
        parse_color_space(internal_indexed_tint_space(kind), context=SemanticContext(version))


@pytest.mark.parametrize("kind", ["Separation", "DeviceN"])
@pytest.mark.parametrize("version", [PdfVersion(1, 3), PdfVersion(1, 7), PdfVersion(2, 0)])
def test_indexed_accepts_tint_bases_from_pdf_13(kind: str, version: PdfVersion) -> None:
    # ISO 32000-2:2020, 8.6.6.3 preserves the same permitted bases from PDF 1.3.
    space = parse_color_space(internal_indexed_tint_space(kind), context=SemanticContext(version))
    assert space.base is not None
    assert space.base.kind == kind
    assert indexed_color_components(space, 0) == (0.0,)
    assert indexed_color_components(space, 1) == (1.0,)


@pytest.mark.parametrize("version", [PdfVersion(1, 1), PdfVersion(1, 2), PdfVersion(2, 0)])
def test_indexed_device_base_has_the_same_palette_semantics(version: PdfVersion) -> None:
    space = parse_color_space(
        ["Indexed", "DeviceRGB", 0, b"\xff\x00\x80"], context=SemanticContext(version)
    )
    assert indexed_color_components(space, 0) == (1.0, 0.0, 128 / 255)


@pytest.mark.parametrize("version", [PdfVersion(1, 2), PdfVersion(2, 0)])
@pytest.mark.parametrize("base", ["Pattern", ["Indexed", "DeviceGray", 0, b"\0"]])
def test_indexed_prohibited_bases_stay_prohibited(base: object, version: PdfVersion) -> None:
    with pytest.raises(ValueError, match="invalid Indexed base color space"):
        parse_color_space(["Indexed", base, 0, b"\0"], context=SemanticContext(version))


def test_nested_pattern_base_keeps_the_document_version() -> None:
    value = ["Pattern", internal_indexed_tint_space("Separation")]
    with pytest.raises(ValueError, match="Indexed.*bases require PDF 1.3"):
        parse_color_space(value, context=SemanticContext(PdfVersion(1, 2)))
    space = parse_color_space(value, context=SemanticContext(PdfVersion(1, 3)))
    assert space.base is not None
    assert space.base.kind == "Indexed"


@pytest.mark.parametrize("kind", ["Separation", "DeviceN"])
def test_no_context_preserves_existing_indexed_base_support(kind: str) -> None:
    space = parse_color_space(internal_indexed_tint_space(kind))
    assert space.base is not None
    assert space.base.kind == kind


@pytest.mark.parametrize("version", [None, PdfVersion(1, 8), PdfVersion(2, 1)])
def test_unknown_context_does_not_guess_color_semantics(version: PdfVersion | None) -> None:
    with pytest.raises(PdfUnsupportedError, match="recognized PDF version"):
        parse_color_space(
            internal_indexed_tint_space("Separation"), context=SemanticContext(version)
        )
