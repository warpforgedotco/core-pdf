# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import cast

from core_pdf.impl.engine.spec.s_07_objects.pdfdict import (
    collect_inherited_values,
    lookup_dict_key,
)
from core_pdf.impl.types import PdfDict

FAST_PATH_DICT = {
    "Type": "Page",
    "MediaBox": [0, 0, 612, 792],
    "Resources": {"Font": {"F1": "5 0 R"}},
    "Contents": "7 0 R",
}

# Bytes-keyed dict defeats both direct-hit lookups and forces the linear
# fallback scan lookup_dict_key falls back to for legacy/malformed key encodings.
SLOW_PATH_DICT = {
    b"Type": "Page",
    b"MediaBox": [0, 0, 612, 792],
    b"Resources": {"Font": {"F1": "5 0 R"}},
    b"Contents": "7 0 R",
}


def build_inheritance_chain(depth: int) -> PdfDict:
    node: dict[str, object] = {"Type": "Pages"}
    for level in range(depth):
        node = {
            "Type": "Pages",
            "Parent": node,
            "Resources": {"Font": {f"F{level}": f"{level} 0 R"}},
        }
    node["Type"] = "Page"
    return cast(PdfDict, node)


PAGE_NODE = build_inheritance_chain(depth=6)
INHERITED_KEYS = ("Resources", "MediaBox", "Rotate", "CropBox")


def identity_resolve(value: object) -> object:
    return value


def test_lookup_dict_key_fast_path_benchmark(benchmark) -> None:
    result = benchmark(lookup_dict_key, FAST_PATH_DICT, "MediaBox")
    assert result == [0, 0, 612, 792]


def test_lookup_dict_key_scan_fallback_benchmark(benchmark) -> None:
    result = benchmark(lookup_dict_key, SLOW_PATH_DICT, "MediaBox")
    assert result == [0, 0, 612, 792]


def test_collect_inherited_values_benchmark(benchmark) -> None:
    result = benchmark(collect_inherited_values, PAGE_NODE, INHERITED_KEYS, identity_resolve)
    assert "Resources" in result
