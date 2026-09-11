"""Sampled and stitching function constraints from ISO 32000-1, 7.10."""

import math

import pytest

from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.pdf_function import compile_pdf_function


def sampled_stream(data: bytes = b"\x00\xff", /, **entries: object) -> PdfStream:
    dictionary: dict[str, object] = {
        "FunctionType": 0,
        "BitsPerSample": 8,
        "Size": [2],
        "Domain": [0, 1],
        "Range": [0, 1],
    }
    dictionary.update(entries)
    return PdfStream(dictionary, data)


@pytest.mark.parametrize("entries", [{}, {"Order": 1}, {"Order": None}])
def test_sampled_linear_order_and_default_interpolate_and_clip(entries: dict[str, object]) -> None:
    # Tables 38/39: clip input to Domain, interpolate, then clip to Range.
    function = compile_pdf_function(
        sampled_stream(bytes([0, 32, 128, 255]), Size=[4], Domain=[0, 3], Range=[0, 255], **entries)
    )
    assert function(-1) == (0,)
    assert function(1.5) == (80,)
    assert function(4) == (255,)


@pytest.mark.parametrize("order", [0, 2, 4, 42, -1, 1.0, "1", True, []])
def test_sampled_rejects_invalid_interpolation_order(order: object) -> None:
    # Table 39 permits only PDF integers 1 and 3, even when Size < 4.
    with pytest.raises(ValueError, match="sampled PDF function"):
        compile_pdf_function(sampled_stream(Order=order))


@pytest.mark.parametrize("sizes", [[4], [5], [2, 4], [4, 2]])
def test_sampled_rejects_unimplemented_cubic_interpolation(sizes: list[int]) -> None:
    # Order 3 must not silently become linear in a dimension with >= 4 samples.
    with pytest.raises(ValueError, match="sampled PDF function"):
        compile_pdf_function(
            sampled_stream(bytes(math.prod(sizes)), Size=sizes, Domain=[0, 1] * len(sizes), Order=3)
        )


@pytest.mark.parametrize(
    ("data", "sizes", "inputs", "expected"),
    [
        (bytes([128]), [1], (0.75,), 128),
        (bytes([0, 200]), [2], (0.25,), 50),
        (bytes([0, 80, 200]), [3], (0.25,), 40),
        (bytes([0, 40, 80, 120, 160, 200]), [2, 3], (0.5, 0.25), 60),
    ],
)
def test_sampled_cubic_order_uses_prescribed_small_dimension_fallback(
    data: bytes, sizes: list[int], inputs: tuple[float, ...], expected: float
) -> None:
    # Section 7.10.2: Size 1 is constant; ignore Order 3 when Size < 4.
    function = compile_pdf_function(
        sampled_stream(data, Size=sizes, Domain=[0, 1] * len(sizes), Range=[0, 255], Order=3)
    )
    assert function(*inputs) == (expected,)


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("Domain", []),
        ("Domain", [0]),
        ("Domain", [0, 1, 7]),
        ("Domain", [0, 1, 7, 9]),
        ("Domain", ["0", 1]),
        ("Range", []),
        ("Range", [0]),
        ("Range", [0, 1, 7]),
        ("Range", [1, 0]),
        ("Range", ["0", 1]),
        ("Range", [0, float("inf")]),
        ("Decode", []),
        ("Decode", [0]),
        ("Decode", [0, 1, 7]),
        ("Decode", [0, 1, 7, 9]),
        ("Encode", []),
        ("Encode", [0]),
        ("Encode", [0, 1, 7]),
        ("Encode", [0, 1, 7, 9]),
    ],
)
def test_sampled_rejects_malformed_numeric_arrays(name: str, values: list[object]) -> None:
    # Tables 38/39: Domain/Encode have 2m entries; Range/Decode have 2n.
    with pytest.raises(ValueError, match="sampled PDF function"):
        compile_pdf_function(sampled_stream(**{name: values}))


def test_sampled_multiple_inputs_and_outputs_preserve_sample_order() -> None:
    # Section 7.10.2: first input dimension varies fastest, output order follows Range.
    function = compile_pdf_function(
        sampled_stream(
            bytes([0, 0, 100, 50, 200, 150, 255, 200]),
            Size=[2, 2],
            Domain=[0, 2, 0, 4],
            Range=[0, 255, -255, 0],
            Decode=[0, 255, 0, -255],
        )
    )
    assert function(1, 2) == pytest.approx((138.75, -100))
    assert function(2, 4) == (255, -200)


def test_sampled_allows_reversed_encode_and_decode_and_constant_range() -> None:
    # Encode/Decode mappings may reverse direction; Range pairs must be ordered.
    function = compile_pdf_function(sampled_stream(Encode=[1, 0], Decode=[1, -1], Range=[0, 1]))
    assert function(0) == (0,)
    assert function(0.75) == (0.5,)
    assert function(1) == (1,)
    constant = compile_pdf_function(sampled_stream(Range=[0.5, 0.5]))
    assert constant(0) == constant(1) == (0.5,)


def stitching_dictionary(bounds: object, count: int) -> dict[str, object]:
    return {
        "FunctionType": 3,
        "Domain": [0, 1],
        "Bounds": bounds,
        "Encode": [0, 1] * count,
        "Functions": [
            {"FunctionType": 2, "Domain": [0, 1], "N": 1, "C0": [i * 10], "C1": [i * 10 + 1]}
            for i in range(count)
        ],
    }


@pytest.mark.parametrize(
    "bounds",
    [[0.8, 0.2], [0.5, 0.5], [0, 0.5], [-0.1, 0.5], [0.5, 1.1], [1, 1], [0.5, "0.8"]],
)
def test_stitching_rejects_unordered_or_out_of_domain_bounds(bounds: list[object]) -> None:
    # Table 41 and 7.10.4: Domain0 < Bounds0 < ... < Bounds(k-2) <= Domain1.
    with pytest.raises(ValueError, match="stitching function"):
        compile_pdf_function(stitching_dictionary(bounds, 3))


@pytest.mark.parametrize("bounds", [None, 0, {}, [float("nan")], ["bad"], [0.5]])
def test_stitching_single_child_requires_an_empty_bounds_array(bounds: object) -> None:
    with pytest.raises(ValueError, match="stitching function"):
        compile_pdf_function(stitching_dictionary(bounds, 1))


def test_stitching_rejects_missing_bounds() -> None:
    dictionary = stitching_dictionary([], 1)
    del dictionary["Bounds"]
    with pytest.raises(ValueError, match="stitching function bounds"):
        compile_pdf_function(dictionary)


def test_stitching_uses_half_open_intervals_and_clips_inputs() -> None:
    function = compile_pdf_function(stitching_dictionary([0.25, 0.75], 3))
    assert function(-1) == (0,)
    assert function(0.125) == (0.5,)
    assert function(0.25) == (10,)
    assert function(0.5) == (10.5,)
    assert function(0.75) == (20,)
    assert function(2) == (21,)


def test_stitching_allows_last_bound_at_upper_endpoint() -> None:
    # Section 7.10.4 explicitly permits the last bound to equal Domain1.
    dictionary = stitching_dictionary([0.5, 1], 3)
    dictionary["Encode"] = [0, 1, 0, 1, 0.25, 0.75]
    function = compile_pdf_function(dictionary)
    assert function(0.75) == (10.5,)
    assert function(1) == function(2) == (20.25,)


def test_stitching_single_child_allows_degenerate_domain_and_reverse_encode() -> None:
    # Section 7.10.4 allows Domain0 == Domain1 only for a single child.
    dictionary = stitching_dictionary([], 1)
    dictionary["Domain"] = [0.5, 0.5]
    dictionary["Encode"] = [0.75, 0.25]
    function = compile_pdf_function(dictionary)
    assert function(0) == function(0.5) == function(1) == (0.75,)


def test_stitching_multiple_children_reject_degenerate_domain() -> None:
    dictionary = stitching_dictionary([0.5], 2)
    dictionary["Domain"] = [0.5, 0.5]
    with pytest.raises(ValueError, match="stitching function bounds"):
        compile_pdf_function(dictionary)
