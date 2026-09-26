"""A tint transform is compiled once per content and remembers its outputs."""

import math

import pytest

from core_pdf.impl import graphics_image_samples as image_samples
from core_pdf.impl.graphics_image_samples import tint_function, tint_function_key
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.types import PdfName, PdfReference


def calculator(program: bytes, **extra: object) -> PdfStream:
    dictionary: dict[str, object] = {
        "FunctionType": 4,
        "Domain": [0, 1],
        "Range": [-1e9, 1e9],
        "Length": len(program),
        **extra,
    }
    return PdfStream(dictionary, program)


@pytest.fixture(autouse=True)
def empty_cache() -> None:
    image_samples.TINT_FUNCTION_CACHE.clear()


def test_equal_streams_share_one_compiled_function() -> None:
    first = calculator(b"{ 2 mul }")
    second = calculator(b"{ 2 mul }")
    assert first is not second
    assert tint_function_key(first) == tint_function_key(second)
    assert tint_function(first) is tint_function(second)
    assert tint_function(first)(0.25) == (0.5,)


def test_different_programs_do_not_share() -> None:
    assert tint_function(calculator(b"{ 2 mul }"))(0.25) == (0.5,)
    assert tint_function(calculator(b"{ 3 mul }"))(0.25) == (0.75,)


def test_a_repeated_input_is_not_evaluated_again(monkeypatch: pytest.MonkeyPatch) -> None:
    runs: list[tuple[float, ...]] = []
    compile_function = image_samples.compile_pdf_function

    def counting(tint_fn: object) -> image_samples.TintFunction:
        compiled = compile_function(tint_fn)

        def run(*inputs: float) -> tuple[float, ...]:
            runs.append(inputs)
            return compiled(*inputs)

        return run

    monkeypatch.setattr(image_samples, "compile_pdf_function", counting)
    function = tint_function(calculator(b"{ 2 mul }"))
    assert function(0.25) == function(0.25) == (0.5,)
    assert runs == [(0.25,)]


def test_negative_zero_is_not_answered_for_zero() -> None:
    # The two zeros are equal keys, and an identity program returns each as is.
    identity = tint_function(calculator(b"{ }", Domain=[-1, 1]))
    assert math.copysign(1.0, identity(0.0)[0]) == 1.0
    assert math.copysign(1.0, identity(-0.0)[0]) == -1.0


def test_a_dictionary_with_references_is_cached_by_object_alone() -> None:
    stream = calculator(b"{ 2 mul }", Extra=PdfReference(4, 0))
    assert tint_function_key(stream) is None
    assert tint_function(stream) is tint_function(stream)
    assert tint_function(calculator(b"{ 2 mul }", Extra=PdfReference(4, 0))) is not (
        tint_function(stream)
    )


def test_names_and_arrays_key_by_content() -> None:
    stream = calculator(b"{ 2 mul }", Name=PdfName(b"X"), Size=[2, 3])
    assert tint_function_key(stream) is not None
