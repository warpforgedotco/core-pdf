# SPDX-License-Identifier: AGPL-3.0-only
"""Operator metadata and dispatch tables for PDF content streams."""

from __future__ import annotations

from typing import Any

#: Content-stream operator name -> the ``TextState`` method that implements it.
OPERATOR_SPECS = {
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

__all__ = (
    "OPERATOR_SPECS",
    "build_operator_handlers",
)


def build_operator_handlers(target: Any) -> dict[str, Any]:
    """Bind each content operator name to its target method."""
    return {name: getattr(target, handler) for name, handler in OPERATOR_SPECS.items()}
