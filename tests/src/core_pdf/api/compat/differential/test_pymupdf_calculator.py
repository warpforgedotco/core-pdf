from io import BytesIO

import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

real_pypdf = pytest.importorskip("pypdf")
real_pymupdf = pytest.importorskip("pymupdf")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize(
    "expression",
    [
        "dup mul",
        "pop -0.25 abs",
        "pop 3 2 add 10 div",
        "pop 3 2 sub 5 div",
        "pop 2 3 exp 10 div",
        "pop 3 2 idiv 4 div",
        "pop 5 3 mod 4 div",
        "pop 4 sqrt 4 div",
        "pop 1 ln",
        "pop 100 log 4 div",
        "pop 0.5 neg abs",
        "pop 1.2 ceiling 4 div",
        "pop 1.8 floor 4 div",
        "pop -1.5 round abs 4 div",
        "pop -6.5 round abs 10 div",
        "pop 6.5 round 10 div",
        "pop -1.49 round abs 4 div",
        "pop -1.51 round abs 4 div",
        "pop -1.8 truncate abs 4 div",
        "pop 1.8 cvi cvr 4 div",
        "pop 30 sin",
        "pop 60 cos",
        "pop 1 1 atan 90 div",
        "dup 0.5 gt { pop 0.2 } { pop 0.8 } ifelse",
        "dup 0.5 le { pop 0.2 } if",
        "dup 0.5 lt { } if",
        "dup 0.5 lt { pop 0.2 } { pop 0.75 0.5 gt { 0.3 } { 0.4 } ifelse } ifelse",
        "pop true false or { 0.2 } { 0.8 } ifelse",
        "pop true false and not { 0.2 } { 0.8 } ifelse",
        "pop true false xor { 0.2 } { 0.8 } ifelse",
        "pop true 1 eq { 0.2 } { 0.8 } ifelse",
        "pop 3 1 bitshift 10 div",
        "pop -1 -30 bitshift 4 div",
        "pop -1 -31 bitshift 4 div",
        "pop 1 31 bitshift 0 lt { 0.2 } { 0.8 } ifelse",
        "pop 3 1 and 4 div",
        "pop 2 1 or 4 div",
        "pop 3 1 xor 4 div",
        "pop -2 not 4 div",
        "pop 0.2 0.3 exch pop",
        "pop 0.2 0.3 1 index exch pop exch pop",
        "pop 0.2 0.3 2 copy pop pop pop",
        "pop 0.2 0.3 0.4 3 1 roll pop pop",
        "pop 0.2 0.3 0.4 3 -1 roll pop pop",
        "pop -1",
        "pop 2",
        "pop +.25",
        "pop 1. 4 div",
        "pop 1\x00 4 div",
        "pop\x00 .25",
        "pop 1_0 20 div",
        "pop 1e-1",
        "pop 1\x0b 4 div",
        "pop 0 1 gt { bogus } if 0.2",
        "pop 0 1 lt { 0.2 } { bogus } ifelse",
        "pop { 3 } dup pop pop 1",
        "pop { 3 } pop 1",
        "pop 0 1 lt 0.2 if",
        "pop 0 1 lt { 0.2 } ifelse",
    ],
)
@pytest.mark.parametrize("tint", [0.25, 0.75])
def test_calculator_tint_function(expression: str, tint: float) -> None:
    internal_assert_calculator_color(expression, tint)


def internal_assert_calculator_color(
    expression: str, tint: float, *, stitched: bool = False
) -> None:
    g = real_pypdf.generic
    name = g.NameObject
    number = g.NumberObject
    writer = real_pypdf.PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    function = g.DecodedStreamObject()
    function.set_data(("{ " + expression + " 0 0 0 }").encode())
    function[name("/FunctionType")] = number(4)
    function[name("/Domain")] = g.ArrayObject([number(0), number(1)])
    function[name("/Range")] = g.ArrayObject([number(0), number(1)] * 4)
    tint_transform = writer._add_object(function)
    if stitched:
        tint_transform = writer._add_object(
            g.DictionaryObject(
                {
                    name("/FunctionType"): number(3),
                    name("/Domain"): g.ArrayObject([number(0), number(1)]),
                    name("/Functions"): g.ArrayObject([tint_transform]),
                    name("/Bounds"): g.ArrayObject(),
                    name("/Encode"): g.ArrayObject([number(0), number(1)]),
                }
            )
        )
    font = g.DictionaryObject(
        {
            name("/Type"): name("/Font"),
            name("/Subtype"): name("/Type1"),
            name("/BaseFont"): name("/Helvetica"),
        }
    )
    page[name("/Resources")] = g.DictionaryObject(
        {
            name("/Font"): g.DictionaryObject({name("/F1"): writer._add_object(font)}),
            name("/ColorSpace"): g.DictionaryObject(
                {
                    name("/Ink"): g.ArrayObject(
                        [
                            name("/Separation"),
                            name("/Example"),
                            name("/DeviceCMYK"),
                            tint_transform,
                        ]
                    )
                }
            ),
        }
    )
    stream = g.DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 20 100 Td /Ink cs {tint} scn (Visible text) Tj ET".encode())
    page[name("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    with (
        real_pymupdf.open(stream=output.getvalue()) as reference,
        compat_pymupdf.open(stream=output.getvalue()) as actual,
    ):
        expected = reference[0].get_text("dict")["blocks"][0]["lines"][0]["spans"][0]
        got = actual[0].get_text("dict")["blocks"][0]["lines"][0]["spans"][0]
        assert got["text"] == expected["text"] == "Visible text"
        assert got["color"] == expected["color"]


@pytest.mark.parametrize("expression", ["pop -1.5 round abs 4 div", "pop -6.5 round abs 10 div"])
@pytest.mark.parametrize("tint", [0.25, 0.75])
def test_stitched_calculator_retains_reader_rounding(expression: str, tint: float) -> None:
    internal_assert_calculator_color(expression, tint, stitched=True)
