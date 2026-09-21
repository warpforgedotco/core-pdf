# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

internal_CONTENT_OPERATORS: dict[str, tuple[str, str | None]] = {
    "BT": ("op_BT", ""),
    "ET": ("op_ET", ""),
    "T*": ("op_T_star", ""),
    "Td": ("op_Td", "nn"),
    "TD": ("op_TD", "nn"),
    "Tj": ("op_Tj", "s"),
    "TJ": ("op_TJ", "a"),
    "Tm": ("op_Tm", "nnnnnn"),
    "Tf": ("op_Tf", "/n"),
    "TL": ("op_TL", "n"),
    "Tc": ("op_Tc", "n"),
    "Tw": ("op_Tw", "n"),
    "Tz": ("op_Tz", "n"),
    "Tr": ("op_Tr", "i"),
    "Ts": ("op_Ts", "n"),
    "'": ("op_quote", "s"),
    '"': ("op_double_quote", "nns"),
    "Do": ("op_Do", "/"),
    "BI": ("op_BI", "I"),
    "BDC": ("op_BDC", "/p"),
    "BMC": ("op_BMC", "/"),
    "EMC": ("op_EMC", ""),
    "q": ("op_q", ""),
    "Q": ("op_Q", ""),
    "cm": ("op_cm", "nnnnnn"),
    "g": ("op_g", "n"),
    "rg": ("op_rg", "nnn"),
    "k": ("op_k", "nnnn"),
    "G": ("op_G", "n"),
    "RG": ("op_RG", "nnn"),
    "K": ("op_K", "nnnn"),
    "CS": ("op_CS", "/"),
    "cs": ("op_cs", "/"),
    "SC": ("op_SC", None),
    "SCN": ("op_SCN", None),
    "sc": ("op_sc", None),
    "scn": ("op_scN", None),
    "sh": ("op_sh", "/"),
    "i": ("op_i", "n"),
    "ri": ("op_ri", "/"),
    "MP": ("op_MP", "/"),
    "DP": ("op_DP", "/p"),
    "BX": ("op_BX", ""),
    "EX": ("op_EX", ""),
    "d0": ("op_d0", "nn"),
    "d1": ("op_d1", "nnnnnn"),
    "w": ("op_w", "n"),
    "J": ("op_J", "i"),
    "j": ("op_j", "i"),
    "M": ("op_M", "n"),
    "d": ("op_d", "an"),
    "gs": ("op_gs", "/"),
    "m": ("op_m", "nn"),
    "l": ("op_l", "nn"),
    "re": ("op_re", "nnnn"),
    "h": ("op_h", ""),
    "c": ("op_c", "nnnnnn"),
    "v": ("op_v", "nnnn"),
    "y": ("op_y", "nnnn"),
    "W": ("op_W", ""),
    "W*": ("op_W_star", ""),
    "S": ("op_paint_stroke", ""),
    "s": ("op_paint_close_stroke", ""),
    "f": ("op_paint_fill", ""),
    "F": ("op_paint_fill", ""),
    "f*": ("op_paint_fill_evenodd", ""),
    "B": ("op_paint_fillstroke", ""),
    "b": ("op_paint_close_fillstroke", ""),
    "B*": ("op_paint_fillstroke_evenodd", ""),
    "b*": ("op_paint_close_fillstroke_evenodd", ""),
    "n": ("op_paint_clear", ""),
}

CONTENT_OPERATOR_HANDLERS = {
    name: handler for name, (handler, _) in internal_CONTENT_OPERATORS.items()
}

CONTENT_OPERATOR_SIGNATURES = {
    name: signature
    for name, (_, signature) in internal_CONTENT_OPERATORS.items()
    if signature is not None
}

INLINE_IMAGE_DATA_OPERATORS = frozenset({b"ID", b"EI"})

PDF_CONTENT_OPERATOR_BYTES = (
    frozenset(name.encode("latin-1") for name in CONTENT_OPERATOR_HANDLERS)
    | INLINE_IMAGE_DATA_OPERATORS
)


__all__: tuple[str, ...] = (
    "CONTENT_OPERATOR_HANDLERS",
    "INLINE_IMAGE_DATA_OPERATORS",
    "PDF_CONTENT_OPERATOR_BYTES",
    "CONTENT_OPERATOR_SIGNATURES",
)
