# SPDX-License-Identifier: AGPL-3.0-only
"""Operator names and dispatch-table inputs for PDF content streams."""

from __future__ import annotations

from typing import Any

TEXT_OP = {
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
    "s": "op_paint_stroke",
    "f": "op_paint_fill",
    "F": "op_paint_fill",
    "f*": "op_paint_fill",
    "B": "op_paint_fillstroke",
    "b": "op_paint_fillstroke",
    "B*": "op_paint_fillstroke",
    "b*": "op_paint_fillstroke",
    "n": "op_paint_clear",
}
TEXT_ONLY_NOOP_OPS = frozenset(
    {
        "m",
        "l",
        "h",
        "v",
        "y",
        "c",
        "re",
        "W",
        "W*",
        "S",
        "s",
        "f",
        "F",
        "f*",
        "B",
        "b",
        "B*",
        "b*",
        "n",
        "w",
        "J",
        "j",
        "M",
        "d",
        "i",
        "BX",
        "EX",
        "MP",
        "DP",
    }
)
TEXT_ONLY_OP = TEXT_OP.copy()
for internal_op in TEXT_ONLY_NOOP_OPS:
    TEXT_ONLY_OP[internal_op] = "op_noop"
del internal_op


def build_operator_tables(
    target: type[Any],
    *,
    capture_graphics: bool,
    capture_clipping: bool,
) -> tuple[
    dict[str, Any],
    dict[bytes, Any],
    list[Any | None],
    dict[int, Any],
]:
    operator_map = TEXT_OP if capture_graphics or capture_clipping else TEXT_ONLY_OP
    handlers = {name: getattr(target, method) for name, method in operator_map.items()}
    byte_handlers = {name.encode("latin-1"): handler for name, handler in handlers.items()}
    single_handlers: list[Any | None] = [None] * 256
    double_handlers: dict[int, Any] = {}
    for name, handler in handlers.items():
        if len(name) == 1:
            single_handlers[ord(name)] = handler
        elif len(name) == 2:
            double_handlers[(ord(name[0]) << 8) | ord(name[1])] = handler
    return handlers, byte_handlers, single_handlers, double_handlers
