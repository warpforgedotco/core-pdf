# SPDX-License-Identifier: AGPL-3.0-only
"""Canonical metadata for PDF content-stream operator tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

OperatorCategory: TypeAlias = Literal[
    "text", "image", "vector_path", "vector_paint", "graphics_state"
]


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    handler: str
    category: OperatorCategory | None = None
    text_only_noop: bool = False
    text_only_skip: bool = False
    type3_replay: bool = False
    cache_keyword: bool = True


OPERATOR_SPECS = {
    "BT": OperatorSpec("op_BT", category="text"),
    "ET": OperatorSpec("op_ET", category="text"),
    "T*": OperatorSpec("op_T_star", category="text"),
    "Td": OperatorSpec("op_Td", category="text"),
    "TD": OperatorSpec("op_TD", category="text"),
    "Tj": OperatorSpec("op_Tj", category="text"),
    "TJ": OperatorSpec("op_TJ", category="text"),
    "Tm": OperatorSpec("op_Tm", category="text"),
    "Tf": OperatorSpec("op_Tf", category="text"),
    "TL": OperatorSpec("op_TL", category="text"),
    "Tc": OperatorSpec("op_Tc", category="text"),
    "Tw": OperatorSpec("op_Tw", category="text"),
    "Tz": OperatorSpec("op_Tz", category="text"),
    "Tr": OperatorSpec("op_Tr", category="text"),
    "Ts": OperatorSpec("op_Ts", category="text"),
    "'": OperatorSpec("op_quote", category="text"),
    '"': OperatorSpec("op_double_quote", category="text"),
    "Do": OperatorSpec("op_Do", category="image"),
    "BI": OperatorSpec("op_BI", category="image"),
    "BDC": OperatorSpec("op_BDC"),
    "BMC": OperatorSpec("op_BMC"),
    "EMC": OperatorSpec("op_EMC"),
    "q": OperatorSpec("op_q", category="graphics_state", type3_replay=True),
    "Q": OperatorSpec("op_Q", category="graphics_state", type3_replay=True),
    "cm": OperatorSpec("op_cm", category="graphics_state", type3_replay=True),
    "g": OperatorSpec("op_g", text_only_skip=True, type3_replay=True),
    "rg": OperatorSpec("op_rg", type3_replay=True),
    "k": OperatorSpec("op_k", type3_replay=True),
    "G": OperatorSpec("op_G", text_only_skip=True, type3_replay=True),
    "RG": OperatorSpec("op_RG", type3_replay=True),
    "K": OperatorSpec("op_K", type3_replay=True),
    "CS": OperatorSpec("op_CS", type3_replay=True),
    "cs": OperatorSpec("op_cs", type3_replay=True),
    "SC": OperatorSpec("op_SC", type3_replay=True),
    "SCN": OperatorSpec("op_SCN", type3_replay=True),
    "sc": OperatorSpec("op_sc", type3_replay=True),
    "scn": OperatorSpec("op_scN", type3_replay=True),
    "sh": OperatorSpec("op_sh", category="vector_paint", type3_replay=True),
    "i": OperatorSpec("op_i", text_only_noop=True, type3_replay=True),
    "ri": OperatorSpec("op_ri", type3_replay=True),
    "MP": OperatorSpec("op_MP", text_only_noop=True),
    "DP": OperatorSpec("op_DP", text_only_noop=True),
    "BX": OperatorSpec("op_BX", text_only_noop=True),
    "EX": OperatorSpec("op_EX", text_only_noop=True),
    "d0": OperatorSpec("op_d0", type3_replay=True, cache_keyword=False),
    "d1": OperatorSpec("op_d1", type3_replay=True, cache_keyword=False),
    "w": OperatorSpec(
        "op_w",
        category="graphics_state",
        text_only_noop=True,
        type3_replay=True,
        cache_keyword=False,
    ),
    "J": OperatorSpec(
        "op_J",
        category="graphics_state",
        text_only_noop=True,
        type3_replay=True,
        cache_keyword=False,
    ),
    "j": OperatorSpec(
        "op_j",
        category="graphics_state",
        text_only_noop=True,
        type3_replay=True,
        cache_keyword=False,
    ),
    "M": OperatorSpec(
        "op_M",
        category="graphics_state",
        text_only_noop=True,
        type3_replay=True,
        cache_keyword=False,
    ),
    "d": OperatorSpec(
        "op_d",
        category="graphics_state",
        text_only_noop=True,
        cache_keyword=False,
    ),
    "gs": OperatorSpec("op_gs", category="graphics_state", type3_replay=True, cache_keyword=False),
    "m": OperatorSpec(
        "op_m",
        category="vector_path",
        text_only_noop=True,
        type3_replay=True,
        cache_keyword=False,
    ),
    "l": OperatorSpec(
        "op_l",
        category="vector_path",
        text_only_noop=True,
        type3_replay=True,
        cache_keyword=False,
    ),
    "re": OperatorSpec(
        "op_re",
        category="vector_path",
        text_only_noop=True,
        type3_replay=True,
        cache_keyword=False,
    ),
    "h": OperatorSpec(
        "op_h",
        category="vector_path",
        text_only_noop=True,
        type3_replay=True,
        cache_keyword=False,
    ),
    "c": OperatorSpec(
        "op_c",
        category="vector_path",
        text_only_noop=True,
        type3_replay=True,
        cache_keyword=False,
    ),
    "v": OperatorSpec(
        "op_v",
        category="vector_path",
        text_only_noop=True,
        type3_replay=True,
        cache_keyword=False,
    ),
    "y": OperatorSpec(
        "op_y",
        category="vector_path",
        text_only_noop=True,
        type3_replay=True,
        cache_keyword=False,
    ),
    "W": OperatorSpec("op_W", text_only_noop=True, type3_replay=True, cache_keyword=False),
    "W*": OperatorSpec("op_W_star", text_only_noop=True, type3_replay=True, cache_keyword=False),
    "S": OperatorSpec(
        "op_paint_stroke",
        category="vector_paint",
        text_only_noop=True,
        type3_replay=True,
        cache_keyword=False,
    ),
    "s": OperatorSpec(
        "op_paint_stroke",
        category="vector_paint",
        text_only_noop=True,
        type3_replay=True,
        cache_keyword=False,
    ),
    "f": OperatorSpec(
        "op_paint_fill",
        category="vector_paint",
        text_only_noop=True,
        type3_replay=True,
        cache_keyword=False,
    ),
    "F": OperatorSpec(
        "op_paint_fill",
        category="vector_paint",
        text_only_noop=True,
        type3_replay=True,
        cache_keyword=False,
    ),
    "f*": OperatorSpec(
        "op_paint_fill",
        category="vector_paint",
        text_only_noop=True,
        type3_replay=True,
        cache_keyword=False,
    ),
    "B": OperatorSpec(
        "op_paint_fillstroke",
        category="vector_paint",
        text_only_noop=True,
        type3_replay=True,
        cache_keyword=False,
    ),
    "b": OperatorSpec(
        "op_paint_fillstroke",
        category="vector_paint",
        text_only_noop=True,
        type3_replay=True,
        cache_keyword=False,
    ),
    "B*": OperatorSpec(
        "op_paint_fillstroke",
        category="vector_paint",
        text_only_noop=True,
        type3_replay=True,
        cache_keyword=False,
    ),
    "b*": OperatorSpec(
        "op_paint_fillstroke",
        category="vector_paint",
        text_only_noop=True,
        type3_replay=True,
        cache_keyword=False,
    ),
    "n": OperatorSpec(
        "op_paint_clear", text_only_noop=True, type3_replay=True, cache_keyword=False
    ),
}

TEXT_OP = {name: spec.handler for name, spec in OPERATOR_SPECS.items()}
TEXT_ONLY_NOOP_OPS = frozenset(name for name, spec in OPERATOR_SPECS.items() if spec.text_only_noop)
TEXT_ONLY_OP = TEXT_OP.copy()
for internal_op in TEXT_ONLY_NOOP_OPS:
    TEXT_ONLY_OP[internal_op] = "op_noop"
del internal_op

TYPE3_REPLAY_OPERATORS = frozenset(
    name for name, spec in OPERATOR_SPECS.items() if spec.type3_replay
)
CACHED_OPERATOR_KEYWORDS = tuple(
    name.encode("latin-1") for name, spec in OPERATOR_SPECS.items() if spec.cache_keyword
)

TEXT_OPERATORS = frozenset(name for name, spec in OPERATOR_SPECS.items() if spec.category == "text")
IMAGE_OPERATORS = frozenset(
    {name for name, spec in OPERATOR_SPECS.items() if spec.category == "image"} | {"ID", "EI"}
)
VECTOR_PATH_OPERATORS = frozenset(
    name for name, spec in OPERATOR_SPECS.items() if spec.category == "vector_path"
)
VECTOR_PAINT_OPERATORS = frozenset(
    name for name, spec in OPERATOR_SPECS.items() if spec.category == "vector_paint"
)
GRAPHICS_STATE_OPERATORS = frozenset(
    name for name, spec in OPERATOR_SPECS.items() if spec.category == "graphics_state"
)

internal_TEXT_ONLY_SKIP_OPS = {
    name for name, spec in OPERATOR_SPECS.items() if spec.text_only_noop or spec.text_only_skip
}
# Some damaged producers emit `N` as a path no-op. It has no normal handler,
# but the text-only scanner historically skips it rather than parsing a keyword.
internal_TEXT_ONLY_SKIP_OPS.add("N")
TEXT_ONLY_SKIP_SINGLE = bytes(
    [1 if chr(value) in internal_TEXT_ONLY_SKIP_OPS else 0 for value in range(256)]
)
TEXT_ONLY_SKIP_DOUBLE = bytearray(65536)
for internal_op in internal_TEXT_ONLY_SKIP_OPS:
    if len(internal_op) == 2:
        TEXT_ONLY_SKIP_DOUBLE[(ord(internal_op[0]) << 8) | ord(internal_op[1])] = 1
del internal_op
