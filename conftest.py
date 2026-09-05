"""Workspace-wide test guards and benchmark defaults."""

from __future__ import annotations

import pathlib

import pytest

internal_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent
internal_SOURCE_ROOTS = (
    internal_REPOSITORY_ROOT / "src",
    internal_REPOSITORY_ROOT / "packages/core-pdf-ocr/src",
)
internal_EXTENSION_SUFFIXES = (".so", ".pyd", ".dylib")


def internal_shadowed_modules() -> list[tuple[pathlib.Path, pathlib.Path]]:
    """Find compiled extensions that shadow a same-named source module."""
    shadowed: list[tuple[pathlib.Path, pathlib.Path]] = []
    for root in internal_SOURCE_ROOTS:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in internal_EXTENSION_SUFFIXES:
                continue
            source = path.parent / f"{path.name.split('.')[0]}.py"
            if source.is_file():
                shadowed.append((path, source))
    return shadowed


def internal_disable_benchmarks_by_default(config: pytest.Config) -> None:
    """Run benchmarks once as ordinary tests unless measurement was requested."""
    if not config.pluginmanager.hasplugin("benchmark"):
        return
    if config.getoption("benchmark_enable") or config.getoption("benchmark_only"):
        return
    config.option.benchmark_disable = True


def pytest_configure(config: pytest.Config) -> None:
    internal_disable_benchmarks_by_default(config)
    shadowed = internal_shadowed_modules()
    if not shadowed:
        return
    lines = [
        "Compiled extension modules are shadowing their Python sources.",
        "Tests would run a compiled snapshot instead of the workspace sources.",
        "",
    ]
    lines += [
        f"  {extension.relative_to(internal_REPOSITORY_ROOT)} shadows "
        f"{source.relative_to(internal_REPOSITORY_ROOT)}"
        for extension, source in shadowed
    ]
    lines += ["", "Delete the extension modules and re-run."]
    raise pytest.UsageError("\n".join(lines))
