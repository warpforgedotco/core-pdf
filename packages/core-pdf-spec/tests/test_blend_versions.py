# SPDX-License-Identifier: AGPL-3.0-only
"""ISO 32000-1 Table 136 versus ISO 32000-2 Table 134 singular blend corners."""

from typing import cast

import numpy
import pytest

from core_pdf_spec.exceptions import PdfUnsupportedError
from core_pdf_spec.s_11_transparency.blend import BlendMode, blend_component, blend_components
from core_pdf_spec.standards import PdfExtension, PdfVersion, SemanticContext


@pytest.mark.parametrize("version", [*[PdfVersion(1, n) for n in range(8)], PdfVersion(2, 0)])
def test_version_selects_both_singular_corners(version: PdfVersion) -> None:
    context = SemanticContext(version)
    revised = version == PdfVersion(2, 0)
    assert blend_component(0, 1, "ColorDodge", context=context) == (0 if revised else 1)
    assert blend_component(1, 0, "ColorBurn", context=context) == (1 if revised else 0)


@pytest.mark.parametrize("mode", ["ColorDodge", "ColorBurn"])
@pytest.mark.parametrize("version", [PdfVersion(1, 7), PdfVersion(2, 0)])
def test_scalar_and_arrays_agree_for_corners_interior_and_broadcasting(
    mode: BlendMode, version: PdfVersion
) -> None:
    values = numpy.asarray([0.0, 5e-324, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    context = SemanticContext(version)
    expected = [
        [blend_component(float(cb), float(cs), mode, context=context) for cs in values]
        for cb in values
    ]
    with numpy.errstate(divide="raise", invalid="raise", over="raise"):
        actual = blend_components(values[:, None], values[None, :], mode, context=context)
    numpy.testing.assert_array_equal(actual, expected)
    assert actual.dtype == numpy.float64
    assert values.tolist() == [0.0, 5e-324, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


def test_full_piecewise_equations_and_no_context_latest_default() -> None:
    assert blend_component(0.2, 0.5, "ColorDodge") == pytest.approx(0.4)
    assert blend_component(0.8, 0.5, "ColorDodge") == 1
    assert blend_component(0.8, 0.5, "ColorBurn") == pytest.approx(0.6)
    assert blend_component(0.2, 0.5, "ColorBurn") == 0
    assert blend_component(0, 1, "ColorDodge") == 0
    assert blend_components(1, 0, "ColorBurn").item() == 1


@pytest.mark.parametrize(
    ("extension", "revised"),
    [
        (PdfExtension("ADBE", PdfVersion(1, 7), 5), True),
        (PdfExtension("ADBE", PdfVersion(1, 7), 3), False),
        (PdfExtension("ADBE", PdfVersion(1, 7), 99), False),
        (PdfExtension("OTHER", PdfVersion(1, 7), 5), False),
        (PdfExtension("ADBE", PdfVersion(1, 6), 5), False),
        (PdfExtension("ADBE", PdfVersion(1, 7), 5, extension_revision=":2099"), False),
    ],
)
def test_adobe_supplement_selects_only_audited_extension_identity(
    extension: PdfExtension, revised: bool
) -> None:
    # Adobe Supplement, BaseVersion 1.7/ExtensionLevel 5 (June 2009), 3.1.
    context = SemanticContext(PdfVersion(1, 7), (extension,))
    assert blend_component(0, 1, "ColorDodge", context=context) == (0 if revised else 1)
    assert blend_components(1, 0, "ColorBurn", context=context).item() == (1 if revised else 0)


@pytest.mark.parametrize("version", [None, PdfVersion(1, 9), PdfVersion(3, 0)])
def test_unknown_explicit_context_is_rejected_even_for_empty_arrays(
    version: PdfVersion | None,
) -> None:
    with pytest.raises(PdfUnsupportedError, match="recognized PDF version"):
        blend_component(0.2, 0.5, "ColorBurn", context=SemanticContext(version))
    with pytest.raises(PdfUnsupportedError, match="recognized PDF version"):
        blend_components(numpy.asarray([]), 0.5, "ColorDodge", context=SemanticContext(version))


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
def test_components_must_be_unit_range_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        blend_component(value, 0.5, "ColorDodge")
    with pytest.raises(ValueError, match="finite"):
        blend_components(0.5, numpy.asarray([value]), "ColorBurn")


def test_invalid_blend_mode_and_incompatible_shapes_are_rejected() -> None:
    invalid_mode = cast(BlendMode, "Unknown")
    with pytest.raises(PdfUnsupportedError, match="blend mode"):
        blend_component(0.2, 0.5, invalid_mode)
    with pytest.raises(PdfUnsupportedError, match="blend mode"):
        blend_components(0.2, 0.5, invalid_mode)
    with pytest.raises(ValueError):
        blend_components(numpy.zeros(2), numpy.zeros(3), "ColorBurn")
