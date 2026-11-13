# SPDX-License-Identifier: AGPL-3.0-only

"""Check that core_pdf.impl's subpackages import only the layers below them.

LAYERS lists the subpackages from the bottom up; each line is one layer. A module may
import its own package and any package on a lower layer, never a package on its own layer
or above, so the import graph between packages has no cycles. The floor packages form one
layer and may import each other. Imports under ``if TYPE_CHECKING:`` are exempt: they name
types for annotations and load nothing. ``lazy`` imports count like any other.

uv run --no-project python scripts/check_impl_layers.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "src"
IMPL = ROOT / "core_pdf/impl"

LAYERS = (
    # The floor, which also works on its own; its packages may import each other.
    ("impl", "spec", "adobe_fonts", "jbig2", "crypto", "postscript", "data"),
    ("base",),
    ("output",),
    ("fonts", "graphics", "layout"),
    ("recovery",),
    ("capture",),
    ("render", "extract"),
    ("document",),
    ("export",),
)
FLOOR = 0
LAYER = {package: index for index, layer in enumerate(LAYERS) for package in layer}
LAZY = re.compile(r"^(\s*)lazy (from|import) ", re.MULTILINE)


def package_of(module: str) -> str:
    parts = module.split(".")
    return parts[2] if len(parts) > 2 else "impl"


def is_type_checking(test: ast.expr) -> bool:
    """Exactly ``TYPE_CHECKING`` or ``typing.TYPE_CHECKING``: a condition that only
    mentions it (``not TYPE_CHECKING``, ``TYPE_CHECKING or x``) can be true at runtime."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return (
        isinstance(test, ast.Attribute)
        and test.attr == "TYPE_CHECKING"
        and isinstance(test.value, ast.Name)
        and test.value.id == "typing"
    )


def type_checking_nodes(tree: ast.AST) -> set[int]:
    """Nodes in the body of an ``if TYPE_CHECKING:`` block; its ``else`` runs."""
    nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and is_type_checking(node.test):
            nodes.update(id(child) for statement in node.body for child in ast.walk(statement))
    return nodes


def imported_modules(node: ast.AST) -> list[str]:
    if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("core_pdf.impl"):
        if node.module == "core_pdf.impl":
            return [f"core_pdf.impl.{alias.name}" for alias in node.names]
        return [node.module]
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names if alias.name.startswith("core_pdf.impl.")]
    return []


def main() -> int:
    problems: list[str] = []
    unknown: set[str] = set()
    for path in sorted(IMPL.rglob("*.py")):
        module = ".".join(path.relative_to(ROOT).with_suffix("").parts).removesuffix(".__init__")
        package = package_of(module)
        if package not in LAYER:
            unknown.add(package)
            continue
        tree = ast.parse(LAZY.sub(r"\1\2 ", path.read_text(encoding="utf-8")))
        exempt = type_checking_nodes(tree)
        for node in ast.walk(tree):
            if id(node) in exempt:
                continue
            for target in imported_modules(node):
                imported = package_of(target)
                if imported == package:
                    continue
                if imported not in LAYER:
                    unknown.add(imported)
                    continue
                here, there = LAYER[package], LAYER[imported]
                if there > here or (there == here and here != FLOOR):
                    line = getattr(node, "lineno", 0)
                    problems.append(
                        f"{path.relative_to(ROOT.parent)}:{line}: {package} imports {target}"
                    )
    if unknown:
        problems.append(f"packages missing from LAYERS: {', '.join(sorted(unknown))}")
    if problems:
        print("\n".join(problems))
        print(f"{len(problems)} imports break the layer order")
        return 1
    print(f"{len(LAYERS)} layers, every import points down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
