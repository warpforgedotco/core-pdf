"""Cross-distribution identities and dependency/API contracts."""

import ast
import importlib
from pathlib import Path

import core_pdf
import core_pdf.impl.exceptions as core_errors
import core_pdf.impl.types as core_types
import core_pdf_spec.exceptions as spec_errors
import core_pdf_spec.types as spec_types

ROOT = Path(__file__).resolve().parents[4]
SPEC_ROOT = ROOT / "packages/core-pdf-spec/src/core_pdf_spec"


def test_core_reexports_spec_object_identities() -> None:
    for name in spec_types.__all__:
        assert getattr(core_types, name) is getattr(spec_types, name)
    for name in spec_errors.__all__:
        assert getattr(core_errors, name) is getattr(spec_errors, name)
        assert getattr(core_pdf, name) is getattr(spec_errors, name)
    assert issubclass(core_errors.PdfDocumentClosedError, spec_errors.PdfError)
    assert issubclass(core_errors.PdfContractError, spec_errors.PdfError)


def test_spec_has_no_core_or_backend_imports_even_for_typing() -> None:
    violations = []
    forbidden = {"core_pdf", "core_pdf_ocr", "fontTools", "imagecodecs", "tesserocr"}
    for path in SPEC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            violations.extend(
                f"{path.relative_to(ROOT)}:{getattr(node, 'lineno', 0)}: {name}"
                for name in names
                if name.split(".")[0] in forbidden
            )
    assert not violations, "\n".join(violations)


def test_consumers_use_supported_spec_exports() -> None:
    violations = []
    roots = (ROOT / "src/core_pdf", ROOT / "packages/core-pdf-ocr/src/core_pdf_ocr")
    for root in roots:
        for path in root.rglob("*.py"):
            if "_vendor" in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.ImportFrom) or not node.module:
                    continue
                if not node.module.startswith("core_pdf_spec."):
                    continue
                module = importlib.import_module(node.module)
                exports = getattr(module, "__all__", ())
                for alias in node.names:
                    if alias.name.startswith("internal_") or alias.name not in exports:
                        violations.append(f"{path.relative_to(ROOT)}: {node.module}.{alias.name}")
    assert not violations, "\n".join(violations)
