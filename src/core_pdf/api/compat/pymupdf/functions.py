"""Function loading details selected by the PyMuPDF compatibility facade."""

from __future__ import annotations

import math
import re

from core_pdf.api.compat._shared import float32
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_08_graphics.pdf_calculator import Number, compile_calculator_function
from core_pdf.impl.spec.s_08_graphics.pdf_function import (
    internal_compile_pdf_function,
    internal_compile_sampled_function,
    internal_number_array,
    internal_PdfFunctionEvaluator,
)


def internal_reader_round(value: Number) -> int:
    # MuPDF chooses the integer away from zero at negative half ties. Keep
    # that reader behavior separate from the PostScript-defined native round.
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def compile_color_function(function: object) -> internal_PdfFunctionEvaluator:
    if isinstance(function, PdfStream) and function.dictionary.get("FunctionType") == 0:
        return internal_compile_sampled_function(function, coordinate_rounding=float32)
    if isinstance(function, PdfStream) and function.dictionary.get("FunctionType") == 4:
        # MuPDF's calculator loader receives true/false as PDF Boolean tokens,
        # but accepts only numbers, keywords and procedure delimiters. Its
        # failed function load supplies zero components. Preserve this reader
        # behavior here; the native calculator supports the specified literals.
        data = function.data
        code = re.sub(rb"%[^\r\n]*", b"", data)
        if re.search(rb"(?:^|[\x00\t\n\f\r {}])(?:true|false)(?=$|[\x00\t\n\f\r {}])", code):
            ranges = function.dictionary.get("Range")
            count = len(ranges) // 2 if isinstance(ranges, (list, tuple)) else 0
            return lambda *_inputs: (0.0,) * count
        return compile_calculator_function(
            data,
            internal_number_array(function.dictionary.get("Domain")),
            internal_number_array(function.dictionary.get("Range")),
            unary_operators={"round": internal_reader_round},
        )
    return internal_compile_pdf_function(function, compile_nested=compile_color_function)
