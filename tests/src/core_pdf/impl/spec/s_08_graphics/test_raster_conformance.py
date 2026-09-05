# SPDX-License-Identifier: AGPL-3.0-only
"""Rasterizer conformance rules from ISO 32000-1 clauses 8.7, 8.9 and 9.3.

Each test names the clause it pins. The stencil-mask and Type 3 rules were
previously pinned the other way by tests written from the implementation, and
the golden raster corpus rasters only first pages, so neither caught them.
"""

from __future__ import annotations

import zlib
from unittest.mock import patch

import pytest

from core_pdf.impl.spec.s_07_filters import pipeline as stream_pipeline
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_08_graphics.image_decode import ImageSource
from core_pdf.impl.spec.s_08_graphics.pdf_function import internal_evaluate_pdf_function
from core_pdf.impl.spec.s_08_graphics.shading import prepare_shading

# Row of 8 samples, only the first bit set.
ONE_HIGH_BIT = bytes((0b10000000,))
MASK_DICT = {"ImageMask": True, "Width": 8, "Height": 1, "BitsPerComponent": 1}


def internal_alpha(dictionary: dict[str, object]) -> list[int]:
    raster = ImageSource(ONE_HIGH_BIT, dictionary).decode()
    assert raster is not None
    return [int(value) for value in raster.array[0, :, 1]]


def test_default_decode_makes_the_zero_sample_paint() -> None:
    """8.9.6.2: "If the Decode array is [ 0 1 ] (the default for an image mask),
    a sample value of 0 shall mark the page with the current colour, and a 1
    shall leave the previous contents unchanged."

    Alpha is the marking mask, so the 0 samples are the opaque ones. This was
    inverted, so every stencil mask painted its own negative.
    """
    assert internal_alpha(dict(MASK_DICT)) == [0, 255, 255, 255, 255, 255, 255, 255]


def test_reversed_decode_swaps_which_sample_paints() -> None:
    """8.9.6.2: "If the Decode array is [ 1 0 ], these meanings shall be reversed"."""
    assert internal_alpha({**MASK_DICT, "Decode": [1, 0]}) == [255, 0, 0, 0, 0, 0, 0, 0]


def test_explicit_default_decode_matches_the_implicit_one() -> None:
    assert internal_alpha({**MASK_DICT, "Decode": [0, 1]}) == internal_alpha(dict(MASK_DICT))


RED_RAMP_COMPONENTS = [
    {"FunctionType": 2, "Domain": [0, 1], "C0": [0.0], "C1": [1.0], "N": 1},
    {"FunctionType": 2, "Domain": [0, 1], "C0": [0.0], "C1": [0.0], "N": 1},
    {"FunctionType": 2, "Domain": [0, 1], "C0": [0.0], "C1": [0.0], "N": 1},
]
RED_RAMP_SINGLE = {
    "FunctionType": 2,
    "Domain": [0, 1],
    "C0": [0.0, 0.0, 0.0],
    "C1": [1.0, 0.0, 0.0],
    "N": 1,
}


@pytest.mark.parametrize("value", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_shading_function_array_matches_the_equivalent_single_function(value: float) -> None:
    """Table 78: Function is "A 1-in, n-out function or an array of n 1-in,
    1-out functions".

    An array matched no branch and fell through to a scalar return, which the
    renderer painted as a grey ramp -- silently, so the lost colour was
    invisible to everything upstream.
    """
    assert internal_evaluate_pdf_function(RED_RAMP_COMPONENTS, value) == pytest.approx(
        internal_evaluate_pdf_function(RED_RAMP_SINGLE, value)
    )


def test_shading_function_array_produces_all_components() -> None:
    assert internal_evaluate_pdf_function(RED_RAMP_COMPONENTS, 0.5) == pytest.approx(
        (0.5, 0.0, 0.0)
    )


def test_prepared_sampled_shading_decodes_its_function_stream_once() -> None:
    function_dictionary = {
        "FunctionType": 0,
        "BitsPerSample": 8,
        "Size": [2],
        "Domain": [0, 1],
        "Range": [0, 1],
        "Filter": "FlateDecode",
    }
    function = PdfStream(
        function_dictionary,
        zlib.compress(bytes((0, 255))),
        spec=function_dictionary,
    )

    with patch.object(
        stream_pipeline,
        "decode_stream_data",
        wraps=stream_pipeline.decode_stream_data,
    ) as decode_stream_data:
        shading = prepare_shading(
            {
                "ShadingType": 2,
                "Coords": [0, 0, 10, 0],
                "ColorSpace": "DeviceGray",
                "Function": function,
            }
        )
        assert shading is not None
        assert shading.evaluate(0.0) == (0.0,)
        assert shading.evaluate(0.25) == pytest.approx((0.25,))
        assert shading.evaluate(1.0) == (1.0,)

    assert decode_stream_data.call_count == 1


def test_sampled_function_interpolates_across_every_input_dimension() -> None:
    function = PdfStream(
        {
            "FunctionType": 0,
            "BitsPerSample": 8,
            "Size": [2, 2],
            "Domain": [0, 1, 0, 1],
            "Range": [0, 1],
        },
        decoded_data=bytes((0, 255, 0, 255)),
    )

    assert internal_evaluate_pdf_function(function, 0.25, 0.75) == pytest.approx((0.25,))


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-0.25, 0.0), (0.25, 0.5), (1.25, 1.0)],
)
def test_exponential_function_clips_input_to_its_domain(value: float, expected: float) -> None:
    function = {
        "FunctionType": 2,
        "Domain": [0, 1],
        "C0": [0],
        "C1": [1],
        "N": 0.5,
    }

    assert internal_evaluate_pdf_function(function, value) == pytest.approx((expected,))


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1.0, 0.0), (2.5, 0.25), (3.0, 0.5), (3.5, 0.75), (5.0, 1.0)],
)
def test_stitching_function_uses_and_clips_to_its_outer_domain(
    value: float, expected: float
) -> None:
    function = {
        "FunctionType": 3,
        "Domain": [2, 4],
        "Bounds": [3],
        "Encode": [0, 1, 0, 1],
        "Functions": [
            {
                "FunctionType": 2,
                "Domain": [0, 1],
                "C0": [0],
                "C1": [0.5],
                "N": 1,
            },
            {
                "FunctionType": 2,
                "Domain": [0, 1],
                "C0": [0.5],
                "C1": [1],
                "N": 1,
            },
        ],
    }

    assert internal_evaluate_pdf_function(function, value) == pytest.approx((expected,))


def test_prepare_shading_rejects_an_unsupported_function() -> None:
    assert (
        prepare_shading(
            {
                "ShadingType": 2,
                "Coords": [0, 0, 10, 0],
                "ColorSpace": "DeviceGray",
                "Function": {"FunctionType": 4},
            }
        )
        is None
    )
