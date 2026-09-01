# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import cast

import pytest

from core_pdf.impl.spec.s_07_syntax.inherited_values import collect_inherited_values
from core_pdf.impl.spec.s_07_syntax.types import PdfDict


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


@pytest.mark.benchmark_high_impact
def test_collect_inherited_values_benchmark(benchmark) -> None:
    result = benchmark(collect_inherited_values, PAGE_NODE, INHERITED_KEYS, identity_resolve)
    assert "Resources" in result
