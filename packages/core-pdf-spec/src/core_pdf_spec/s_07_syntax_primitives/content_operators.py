# SPDX-License-Identifier: AGPL-3.0-only
"""The content-stream operator vocabulary.

The single table of PDF 7.8.2 operators. It lives at the spec floor because two
layers need it and neither may import the other: `s_07_filters` recognizes
content streams before parsing them, and `s_07_content` binds each operator to
the interpreter method that implements it. Add operators here and implement
their handlers in `s_07_content/state.py`.
"""

from __future__ import annotations

#: Operator name -> the `TextState` method that implements it.
CONTENT_OPERATOR_HANDLERS = {
    "BT": "op_BT",
    "ET": "op_ET",
    "T*": "op_T_star",
    "Td": "op_Td",
    "TD": "op_TD",
    "Tj": "op_Tj",
    "TJ": "op_TJ",
    "Tm": "op_Tm",
    "Tf": "op_Tf",
    "TL": "op_TL",
    "Tc": "op_Tc",
    "Tw": "op_Tw",
    "Tz": "op_Tz",
    "Tr": "op_Tr",
    "Ts": "op_Ts",
    "'": "op_quote",
    '"': "op_double_quote",
    "Do": "op_Do",
    "BI": "op_BI",
    "BDC": "op_BDC",
    "BMC": "op_BMC",
    "EMC": "op_EMC",
    "q": "op_q",
    "Q": "op_Q",
    "cm": "op_cm",
    "g": "op_g",
    "rg": "op_rg",
    "k": "op_k",
    "G": "op_G",
    "RG": "op_RG",
    "K": "op_K",
    "CS": "op_CS",
    "cs": "op_cs",
    "SC": "op_SC",
    "SCN": "op_SCN",
    "sc": "op_sc",
    "scn": "op_scN",
    "sh": "op_sh",
    "i": "op_i",
    "ri": "op_ri",
    "MP": "op_MP",
    "DP": "op_DP",
    "BX": "op_BX",
    "EX": "op_EX",
    "d0": "op_d0",
    "d1": "op_d1",
    "w": "op_w",
    "J": "op_J",
    "j": "op_j",
    "M": "op_M",
    "d": "op_d",
    "gs": "op_gs",
    "m": "op_m",
    "l": "op_l",
    "re": "op_re",
    "h": "op_h",
    "c": "op_c",
    "v": "op_v",
    "y": "op_y",
    "W": "op_W",
    "W*": "op_W_star",
    "S": "op_paint_stroke",
    "s": "op_paint_close_stroke",
    "f": "op_paint_fill",
    "F": "op_paint_fill",
    "f*": "op_paint_fill_evenodd",
    "B": "op_paint_fillstroke",
    "b": "op_paint_close_fillstroke",
    "B*": "op_paint_fillstroke_evenodd",
    "b*": "op_paint_close_fillstroke_evenodd",
    "n": "op_paint_clear",
}

#: `ID` and `EI` delimit inline-image data rather than invoking a handler, so
#: they carry no entry above but are still part of the lexical vocabulary.
INLINE_IMAGE_DATA_OPERATORS = frozenset({b"ID", b"EI"})

#: Lexical vocabulary used to recognize content streams before they are parsed.
PDF_CONTENT_OPERATOR_BYTES = (
    frozenset(name.encode("latin-1") for name in CONTENT_OPERATOR_HANDLERS)
    | INLINE_IMAGE_DATA_OPERATORS
)


__all__: tuple[str, ...] = (
    "CONTENT_OPERATOR_HANDLERS",
    "INLINE_IMAGE_DATA_OPERATORS",
    "PDF_CONTENT_OPERATOR_BYTES",
)

# Operand categories in ISO 32000-1 Tables 51 and 57–59 and Annex A.
# Variable-length color operands are checked against the selected color space.
CONTENT_OPERATOR_SIGNATURES = {
    "BT": "",
    "ET": "",
    "T*": "",
    "Td": "nn",
    "TD": "nn",
    "Tj": "s",
    "TJ": "a",
    "Tm": "nnnnnn",
    "Tf": "/n",
    "TL": "n",
    "Tc": "n",
    "Tw": "n",
    "Tz": "n",
    "Tr": "i",
    "Ts": "n",
    "'": "s",
    '"': "nns",
    "Do": "/",
    "BI": "I",
    "BDC": "/p",
    "BMC": "/",
    "EMC": "",
    "q": "",
    "Q": "",
    "cm": "nnnnnn",
    "g": "n",
    "rg": "nnn",
    "k": "nnnn",
    "G": "n",
    "RG": "nnn",
    "K": "nnnn",
    "CS": "/",
    "cs": "/",
    "sh": "/",
    "i": "n",
    "ri": "/",
    "MP": "/",
    "DP": "/p",
    "BX": "",
    "EX": "",
    "d0": "nn",
    "d1": "nnnnnn",
    "w": "n",
    "J": "i",
    "j": "i",
    "M": "n",
    "d": "an",
    "gs": "/",
    "m": "nn",
    "l": "nn",
    "re": "nnnn",
    "h": "",
    "c": "nnnnnn",
    "v": "nnnn",
    "y": "nnnn",
    "W": "",
    "W*": "",
    "S": "",
    "s": "",
    "f": "",
    "F": "",
    "f*": "",
    "B": "",
    "b": "",
    "B*": "",
    "b*": "",
    "n": "",
}

__all__ += ("CONTENT_OPERATOR_SIGNATURES",)
