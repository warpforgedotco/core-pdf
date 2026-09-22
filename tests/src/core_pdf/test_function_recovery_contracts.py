import pytest

from core_pdf.impl.graphics.functions import compile_pdf_function
from core_pdf_spec.exceptions import PdfParseError, PdfUnsupportedError
from core_pdf_spec.s_07_filters.errors import FilterParseError
from core_pdf_spec.s_07_syntax.stream import PdfStream


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        (lambda x: x * 2, (0.5,)),
        (lambda x: [x, 1 - x], (0.25, 0.75)),
        ([2, "3"], (2.0, 3.0)),
        ((lambda x: [x], lambda x: [1 - x, 2]), (0.25, 0.75, 2.0)),
        ((lambda x: [],), (0.25,)),
    ],
)
def test_callable_and_constant_adapters_preserve_output_shape(function, expected):
    assert compile_pdf_function(function)(0.25) == expected


@pytest.mark.parametrize("value", [None, 42, [], [None], [lambda x: x, None]])
def test_unsupported_function_inputs_fail_during_compilation(value):
    with pytest.raises(ValueError, match="invalid PDF function"):
        compile_pdf_function(value)


@pytest.mark.parametrize("x", [0, 0.25, 1])
@pytest.mark.parametrize(
    ("start", "end", "expected_start", "expected_end"),
    [
        ([0, 2], [1], (0, 2), (1, 1)),
        ([0], [1, 3], (0, 0), (1, 3)),
        (None, None, (0,), (1,)),
        (["bad"], [], (0,), (1,)),
    ],
)
def test_exponential_defaults_and_component_extension_are_shared(
    x, start, end, expected_start, expected_end
):
    function = {"FunctionType": "2", "C0": start, "C1": end, "Range": [0, 0.1]}
    actual = compile_pdf_function(function)(x)
    assert actual == pytest.approx(
        tuple(a + x * (b - a) for a, b in zip(expected_start, expected_end, strict=True))
    )
    assert function["Range"] == [0, 0.1]


@pytest.mark.parametrize(
    ("bounds", "x", "expected"),
    [
        ([0.75, 0.25], 0.0, 0.0),
        ([0.75, 0.25], 0.375, 0.5),
        ([0.75, 0.25], 0.75, 20 + 2 / 3),
        ([0.75, 0.25], 1.0, 21.0),
        ([0.5, 0.5], 0.5, 20.0),
        ([1.0, 1.0], 1.0, 20.0),
        ([2.0], 1.0, 0.5),
    ],
)
def test_unordered_stitching_preserves_first_match_and_degenerate_intervals(bounds, x, expected):
    parts = [{"FunctionType": 2, "C0": [offset], "C1": [offset + 1]} for offset in (0, 10, 20)]
    function = {"FunctionType": 3, "Domain": [0, 1], "Bounds": bounds, "Functions": parts}
    evaluate = compile_pdf_function(function)
    assert evaluate(x) == pytest.approx((expected,))
    with pytest.raises(ValueError, match="input count"):
        evaluate(x, x)


@pytest.mark.parametrize("x", [-1, 0.25, 0.75, 2])
def test_stitching_repeats_missing_parts_and_defaults_missing_encode_values(x):
    part = {"FunctionType": 2, "C0": [0], "C1": [1]}
    function = {"FunctionType": 3, "Bounds": [0.5], "Functions": [part], "Encode": [1, 0]}
    expected = 1 - 2 * max(0, x) if x < 0.5 else 2 * min(1, x) - 1
    assert compile_pdf_function(function)(x) == pytest.approx((expected,))
    assert function["Functions"] == [part]
    assert function["Encode"] == [1, 0]


@pytest.mark.parametrize("kind", [0, 4])
@pytest.mark.parametrize(
    "failure_type",
    [TypeError, ValueError, ArithmeticError, FilterParseError, PdfParseError, PdfUnsupportedError],
)
def test_known_stream_failures_are_labeled_and_keep_their_cause(kind, failure_type):
    failure = failure_type("broken input")
    calls = []

    def decode(*args, **kwargs):
        calls.append(1)
        raise failure

    stream = PdfStream({"FunctionType": kind}, b"data", decoder=decode)
    with pytest.raises(ValueError, match="sampled" if kind == 0 else "calculator") as caught:
        compile_pdf_function(stream)
    assert caught.value.__cause__ is failure
    assert calls == [1]


@pytest.mark.parametrize("kind", [0, 4])
def test_unexpected_decoder_defects_are_not_reclassified_as_numeric_errors(kind):
    failure = RuntimeError("decoder defect")

    def decode(*args, **kwargs):
        raise failure

    with pytest.raises(RuntimeError) as caught:
        compile_pdf_function(PdfStream({"FunctionType": kind}, decoder=decode))
    assert caught.value is failure


@pytest.mark.parametrize("domain", [None, "bad", [0, None]])
def test_sampled_functions_reject_an_unusable_input_domain(domain):
    stream = PdfStream(
        {"FunctionType": 0, "Domain": domain, "Range": [0, 1], "Size": [2], "BitsPerSample": 8},
        b"\x00\xff",
    )
    with pytest.raises(ValueError):
        compile_pdf_function(stream)


def test_sampled_functions_default_unusable_encode_pairs_to_sample_extents():
    stream = PdfStream(
        {
            "FunctionType": 0,
            "Domain": [0, 1],
            "Range": [0, 1],
            "Size": [2],
            "BitsPerSample": 8,
            "Encode": ["bad", 1],
        },
        b"\x00\xff",
    )
    evaluate = compile_pdf_function(stream)
    assert [evaluate(x) for x in (0, 0.5, 1)] == [(0.0,), (0.5,), (1.0,)]


def test_sampled_functions_reject_nonfinite_decode_values():
    stream = PdfStream(
        {
            "FunctionType": 0,
            "Domain": [0, 1],
            "Range": [0, 1],
            "Size": [2],
            "BitsPerSample": 8,
            "Decode": [float("nan"), 1],
        },
        b"\x00\xff",
    )
    with pytest.raises(ValueError):
        compile_pdf_function(stream)


@pytest.mark.parametrize("parts", [None, []])
def test_stitching_without_subfunctions_is_not_recoverable(parts):
    with pytest.raises(ValueError):
        compile_pdf_function({"FunctionType": 3, "Functions": parts})
