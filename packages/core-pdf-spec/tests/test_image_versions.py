# SPDX-License-Identifier: AGPL-3.0-only
"""JPX Decode semantics changed between ISO 32000-1 and ISO 32000-2."""

import pytest

from core_pdf_spec.exceptions import PdfUnsupportedError
from core_pdf_spec.s_07_syntax.resolver import ObjectResolver
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.image_spec import (
    image_decode_array_applies,
    image_smask_in_data,
    image_source_from_stream,
)
from core_pdf_spec.standards import PdfVersion, SemanticContext


@pytest.mark.parametrize("version", [PdfVersion(1, 5), PdfVersion(1, 7), PdfVersion(2, 0)])
@pytest.mark.parametrize("color_space", [False, True])
def test_jpx_decode_requires_pdf20_and_explicit_color_space(
    version: PdfVersion, color_space: bool
) -> None:
    dictionary: dict[object, object] = {"Filter": ["FlateDecode", "JPXDecode"]}
    if color_space:
        dictionary["ColorSpace"] = "DeviceGray"
    assert image_decode_array_applies(dictionary, context=SemanticContext(version)) is (
        version == PdfVersion(2, 0) and color_space
    )


@pytest.mark.parametrize("version", [PdfVersion(1, 7), PdfVersion(2, 0)])
def test_jpx_stencil_and_non_jpx_decode_always_apply(version: PdfVersion) -> None:
    context = SemanticContext(version)
    assert image_decode_array_applies({"Filter": "JPXDecode", "ImageMask": True}, context=context)
    assert image_decode_array_applies({"Filter": "FlateDecode"}, context=context)


def test_no_context_preserves_iso1_jpx_decode_behavior() -> None:
    assert not image_decode_array_applies({"Filter": "JPXDecode", "ColorSpace": "DeviceGray"})


@pytest.mark.parametrize("version", [None, PdfVersion(2, 1)])
def test_jpx_unknown_version_does_not_guess_decode_semantics(version: PdfVersion | None) -> None:
    with pytest.raises(PdfUnsupportedError, match="recognized PDF version"):
        image_decode_array_applies({"Filter": "JPXDecode"}, context=SemanticContext(version))


def test_passive_image_source_preserves_explicit_context() -> None:
    context = SemanticContext(PdfVersion(1, 7))
    resolver = ObjectResolver(b"", {})
    try:
        source = image_source_from_stream(PdfStream({}), resolver, semantic_context=context)
        assert source.semantic_context is context
        assert image_source_from_stream(PdfStream({}), resolver).semantic_context is None
    finally:
        resolver.close()


@pytest.mark.parametrize("selector", [0, 1, 2])
def test_jpx_opacity_selector_retains_straight_and_premultiplied_modes(selector: int) -> None:
    assert image_smask_in_data({"Filter": "JPXDecode", "SMaskInData": selector}) == selector
    assert image_smask_in_data({"Filter": "FlateDecode", "SMaskInData": selector}) == 0
    assert image_smask_in_data({"Filter": "JPXDecode"}) == 0


@pytest.mark.parametrize("selector", [-1, 3, True, 1.0, "1"])
def test_jpx_opacity_selector_is_a_strict_pdf_integer(selector: object) -> None:
    with pytest.raises(ValueError, match="SMaskInData"):
        image_smask_in_data({"Filter": "JPXDecode", "SMaskInData": selector})
