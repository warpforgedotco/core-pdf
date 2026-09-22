# SPDX-License-Identifier: AGPL-3.0-only

import math
import zlib

import pytest

from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_08_graphics.pdf_function import PdfFunctionEvaluator, compile_pdf_function
from core_pdf_spec.types import PdfName


def internal_stream(program: bytes, **entries: object) -> PdfStream:
    dictionary: dict[str, object] = {
        "FunctionType": 4,
        "Domain": [-1e100, 1e100],
        "Range": [-1e100, 1e100],
    }
    dictionary.update(entries)
    return PdfStream(dictionary, program)


def constant(expression: str, outputs: int = 1) -> PdfFunctionEvaluator:
    return compile_pdf_function(
        internal_stream(
            ("{ pop " + expression + " }").encode("ascii"), Range=[-1e100, 1e100] * outputs
        )
    )


def test_calculator_clips_inputs_before_execution_and_outputs_after_execution() -> None:
    function = compile_pdf_function(internal_stream(b"{ sqrt }", Domain=[0, 9], Range=[0, 2]))
    assert function(-4) == (0,)
    assert function(1) == (1,)
    assert function(4) == (2,)
    assert function(16) == (2,)


def test_calculator_multiple_inputs_and_outputs_follow_stack_order() -> None:
    function = compile_pdf_function(
        internal_stream(b"{ exch }", Domain=[0, 2, 3, 5], Range=[0, 4, 1, 2])
    )
    assert function(-10, 10) == (4, 1)
    assert function(1.5, 3.5) == (3.5, 1.5)


def test_calculator_constant_domain_and_range_are_valid() -> None:
    function = compile_pdf_function(internal_stream(b"{ dup mul }", Domain=[2, 2], Range=[3, 3]))
    assert function(99) == (3,)


@pytest.mark.parametrize("name", ["Domain", "Range"])
@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        [0],
        [0, 1, 2],
        [1, 0],
        [False, 1],
        [0, True],
        ["0", 1],
        [0, math.inf],
        [-math.inf, 1],
        [math.nan, 1],
        [0, 10**400],
        "01",
    ],
)
def test_calculator_requires_nonempty_ordered_finite_domain_and_range(
    name: str, value: object
) -> None:
    with pytest.raises(ValueError):
        compile_pdf_function(internal_stream(b"{}", **{name: value}))


@pytest.mark.parametrize("name", ["Domain", "Range"])
def test_calculator_requires_explicit_domain_and_range(name: str) -> None:
    stream = internal_stream(b"{}")
    del stream.dictionary[name]
    with pytest.raises(ValueError):
        compile_pdf_function(stream)


def test_calculator_requires_a_stream_not_a_plain_function_dictionary() -> None:
    with pytest.raises(ValueError):
        compile_pdf_function({"FunctionType": 4, "Domain": [0, 1], "Range": [0, 1]})


def test_calculator_reuses_compiled_program_without_leaking_operand_stack() -> None:
    function = compile_pdf_function(internal_stream(b"{ dup 0 eq { pop -1 sqrt } if 2 mul }"))
    assert function(3) == (6,)
    with pytest.raises(ValueError):
        function(0)
    assert function(4) == (8,)
    assert function(3) == (6,)


def test_calculator_decodes_stream_before_parsing() -> None:
    program = b"{ 1 exch sub }"
    stream = internal_stream(
        zlib.compress(program), Filter=PdfName(b"FlateDecode"), Domain=[0, 1], Range=[0, 1]
    )
    stream.spec = stream.dictionary
    assert compile_pdf_function(stream)(0.25) == (0.75,)


def test_calculator_compiles_decoded_memoryview_stream_once() -> None:
    calls: list[bytes] = []

    def decode(
        data: bytes | memoryview,
        dictionary: object,
        *,
        parent_dictionary: object | None = None,
    ) -> bytes:
        assert dictionary is None
        assert parent_dictionary is stream.dictionary
        calls.append(bytes(data))
        return b"{ dup mul }"

    stream = PdfStream(
        {"FunctionType": 4, "Domain": [0, 3], "Range": [0, 9]},
        memoryview(b"encoded program"),
        decoder=decode,
    )
    function = compile_pdf_function(stream)
    assert calls == [b"encoded program"]
    assert function(2) == (4,)
    assert function(3) == (9,)
    assert calls == [b"encoded program"]
