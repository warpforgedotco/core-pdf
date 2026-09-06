# SPDX-License-Identifier: AGPL-3.0-only
"""Keep reader policy out of the PDF semantics, including annotation imports."""

from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path

SPEC_ROOT = Path(__file__).resolve().parents[1] / "src/core_pdf/impl/spec"
NEUTRAL_IMPORTS = {
    "core_pdf.impl._impl.model.geometry": {"transform_bbox", "points_bbox"},
    "core_pdf.impl._impl.runtime.array_views": {"uint8_view"},
}


def test_spec_does_not_import_reader_policy_even_for_type_checking() -> None:
    violations: list[str] = []
    for path in sorted(SPEC_ROOT.rglob("*.py")):
        package = "core_pdf.impl.spec." + ".".join(path.parent.relative_to(SPEC_ROOT).parts)
        package = package.rstrip(".")
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                modules = [(alias.name, None) for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = (
                    resolve_name("." * node.level + (node.module or ""), package)
                    if node.level
                    else node.module or ""
                )
                modules = [(module, alias.name) for alias in node.names]
            else:
                continue
            for module, symbol in modules:
                target = f"{module}.{symbol}" if symbol is not None else module
                if not target.startswith("core_pdf.impl._impl"):
                    continue
                if symbol is not None and symbol in NEUTRAL_IMPORTS.get(module, set()):
                    continue
                violations.append(f"{path.relative_to(SPEC_ROOT)}:{node.lineno}: {target}")
    assert not violations, "Reader policy imported by spec:\n" + "\n".join(violations)
