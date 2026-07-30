from __future__ import annotations

import ast
from pathlib import Path

ENGINE_ROOT = Path(__file__).parents[1] / "src" / "core_pdf" / "impl" / "engine"
VENDOR_ROOT = Path(__file__).parents[1] / "src" / "core_pdf" / "_vendor"
SOURCE_ROOT = Path(__file__).parents[1] / "src" / "core_pdf"


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_shared_execution_is_owned_by_the_engine() -> None:
    assert not (ENGINE_ROOT / "parse" / "runtime.py").exists()
    assert not (ENGINE_ROOT / "parse").is_dir()
    assert (ENGINE_ROOT / "execution.py").exists()
    for path in ENGINE_ROOT.rglob("*.py"):
        assert "core_pdf.impl.engine.parse.runtime" not in imported_modules(path)


def test_execution_does_not_depend_on_downstream_engine_packages() -> None:
    imports = imported_modules(ENGINE_ROOT / "execution.py")
    assert not {
        module
        for module in imports
        if module.startswith(
            (
                "core_pdf.impl.engine.parse",
                "core_pdf.impl.engine.rendering",
                "core_pdf.impl.engine.writing",
            )
        )
    }


def test_rendering_layout_separates_display_records_from_rasterization() -> None:
    assert (ENGINE_ROOT / "rendering.py").exists()
    assert not (ENGINE_ROOT / "rendering").is_dir()


def test_parse_does_not_depend_on_private_rendering_kernels() -> None:
    imports = imported_modules(ENGINE_ROOT / "parse.py")
    assert "core_pdf.impl.engine.rendering" in imports
    assert not {
        module for module in imports if module.startswith("core_pdf.impl.engine.rendering.")
    }


def test_public_page_uses_pipeline_instead_of_emitter_details() -> None:
    imports = imported_modules(ENGINE_ROOT / "page.py")
    assert "core_pdf.impl.engine.parse" in imports
    assert "core_pdf.impl.engine.rendering" in imports


def test_vendor_namespace_contains_only_third_party_code() -> None:
    assert {
        path.name for path in VENDOR_ROOT.iterdir() if path.is_dir() and path.name != "__pycache__"
    } == {"fontTools"}


def test_first_party_vendor_imports_are_absent() -> None:
    for path in VENDOR_ROOT.parent.rglob("*.py"):
        if "_vendor/fontTools" in path.as_posix():
            continue
        assert "core_pdf._vendor.core_" not in path.read_text(encoding="utf-8")


def test_compile_namespaces_are_removed() -> None:
    assert not list(SOURCE_ROOT.rglob("compile"))
    for path in SOURCE_ROOT.rglob("*.py"):
        if "_vendor/fontTools" in path.as_posix():
            continue
        assert not {module for module in imported_modules(path) if ".compile" in module}
