# SPDX-License-Identifier: AGPL-3.0-only
"""Operator metadata and dispatch tables for PDF content streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    handler: str
    text_only_skip: bool = False


OPERATOR_SPECS = {
    "BT": OperatorSpec("op_BT"),
    "ET": OperatorSpec("op_ET"),
    "T*": OperatorSpec("op_T_star"),
    "Td": OperatorSpec("op_Td"),
    "TD": OperatorSpec("op_TD"),
    "Tj": OperatorSpec("op_Tj"),
    "TJ": OperatorSpec("op_TJ"),
    "Tm": OperatorSpec("op_Tm"),
    "Tf": OperatorSpec("op_Tf"),
    "TL": OperatorSpec("op_TL"),
    "Tc": OperatorSpec("op_Tc"),
    "Tw": OperatorSpec("op_Tw"),
    "Tz": OperatorSpec("op_Tz"),
    "Tr": OperatorSpec("op_Tr"),
    "Ts": OperatorSpec("op_Ts"),
    "'": OperatorSpec("op_quote"),
    '"': OperatorSpec("op_double_quote"),
    "Do": OperatorSpec("op_Do"),
    "BI": OperatorSpec("op_BI"),
    "BDC": OperatorSpec("op_BDC"),
    "BMC": OperatorSpec("op_BMC"),
    "EMC": OperatorSpec("op_EMC"),
    "q": OperatorSpec("op_q"),
    "Q": OperatorSpec("op_Q"),
    "cm": OperatorSpec("op_cm"),
    "g": OperatorSpec("op_g", text_only_skip=True),
    "rg": OperatorSpec("op_rg"),
    "k": OperatorSpec("op_k"),
    "G": OperatorSpec("op_G", text_only_skip=True),
    "RG": OperatorSpec("op_RG"),
    "K": OperatorSpec("op_K"),
    "CS": OperatorSpec("op_CS"),
    "cs": OperatorSpec("op_cs"),
    "SC": OperatorSpec("op_SC"),
    "SCN": OperatorSpec("op_SCN"),
    "sc": OperatorSpec("op_sc"),
    "scn": OperatorSpec("op_scN"),
    "sh": OperatorSpec("op_sh"),
    "i": OperatorSpec("op_i", text_only_skip=True),
    "ri": OperatorSpec("op_ri"),
    "MP": OperatorSpec("op_MP", text_only_skip=True),
    "DP": OperatorSpec("op_DP", text_only_skip=True),
    "BX": OperatorSpec("op_BX", text_only_skip=True),
    "EX": OperatorSpec("op_EX", text_only_skip=True),
    "d0": OperatorSpec("op_d0"),
    "d1": OperatorSpec("op_d1"),
    "w": OperatorSpec("op_w", text_only_skip=True),
    "J": OperatorSpec("op_J", text_only_skip=True),
    "j": OperatorSpec("op_j", text_only_skip=True),
    "M": OperatorSpec("op_M", text_only_skip=True),
    "d": OperatorSpec("op_d", text_only_skip=True),
    "gs": OperatorSpec("op_gs"),
    "m": OperatorSpec("op_m", text_only_skip=True),
    "l": OperatorSpec("op_l", text_only_skip=True),
    "re": OperatorSpec("op_re", text_only_skip=True),
    "h": OperatorSpec("op_h", text_only_skip=True),
    "c": OperatorSpec("op_c", text_only_skip=True),
    "v": OperatorSpec("op_v", text_only_skip=True),
    "y": OperatorSpec("op_y", text_only_skip=True),
    "W": OperatorSpec("op_W", text_only_skip=True),
    "W*": OperatorSpec("op_W_star", text_only_skip=True),
    "S": OperatorSpec("op_paint_stroke", text_only_skip=True),
    "s": OperatorSpec("op_paint_close_stroke", text_only_skip=True),
    "f": OperatorSpec("op_paint_fill", text_only_skip=True),
    "F": OperatorSpec("op_paint_fill", text_only_skip=True),
    "f*": OperatorSpec("op_paint_fill_evenodd", text_only_skip=True),
    "B": OperatorSpec("op_paint_fillstroke", text_only_skip=True),
    "b": OperatorSpec("op_paint_close_fillstroke", text_only_skip=True),
    "B*": OperatorSpec("op_paint_fillstroke_evenodd", text_only_skip=True),
    "b*": OperatorSpec("op_paint_close_fillstroke_evenodd", text_only_skip=True),
    "n": OperatorSpec("op_paint_clear", text_only_skip=True),
}

# Some damaged producers emit `N` as a path no-op. It has no normal handler,
# but the text-only scanner historically skips it rather than parsing a keyword.
TEXT_ONLY_SKIP_OPERATORS = frozenset(
    {b"N"}
    | {name.encode("latin-1") for name, spec in OPERATOR_SPECS.items() if spec.text_only_skip}
)


__all__ = (
    "OPERATOR_SPECS",
    "TEXT_ONLY_SKIP_OPERATORS",
    "build_operator_handlers",
)


def build_operator_handlers(target: Any) -> dict[str, Any]:
    """Bind each content operator name to its target method."""
    return {name: getattr(target, spec.handler) for name, spec in OPERATOR_SPECS.items()}
