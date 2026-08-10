"""Session-wide guards that run before any test is collected."""

from __future__ import annotations

import pathlib

import pytest

internal_SOURCE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "src"
internal_EXTENSION_SUFFIXES = (".so", ".pyd", ".dylib")


def internal_shadowed_modules() -> list[tuple[pathlib.Path, pathlib.Path]]:
    """Find compiled extensions that shadow a same-named source module.

    Python's ``FileFinder`` consults ``ExtensionFileLoader`` before
    ``SourceFileLoader``, so a stale ``lexer.cpython-313-darwin.so`` left in the
    tree by a Nuitka module build wins over ``lexer.py`` and keeps reporting the
    ``.py`` path as ``__file__``. Edits to the source then silently do nothing.
    """
    shadowed: list[tuple[pathlib.Path, pathlib.Path]] = []
    for path in internal_SOURCE_ROOT.rglob("*"):
        if not path.is_file():
            continue
        # Extension modules are named "<module>.<abi-tag>.<suffix>"; the module
        # name is everything before the first dot.
        suffix = "".join(path.suffixes[-1:])
        if suffix not in internal_EXTENSION_SUFFIXES:
            continue
        source = path.parent / f"{path.name.split('.')[0]}.py"
        if source.is_file():
            shadowed.append((path, source))
    return shadowed


def pytest_configure(config: pytest.Config) -> None:
    shadowed = internal_shadowed_modules()
    if not shadowed:
        return
    lines = [
        "Compiled extension modules are shadowing their Python sources.",
        "Tests would run the compiled snapshot, not the code in src/.",
        "",
    ]
    lines += [
        f"  {extension.relative_to(internal_SOURCE_ROOT)} shadows "
        f"{source.relative_to(internal_SOURCE_ROOT)}"
        for extension, source in shadowed
    ]
    lines += ["", "Delete the extension modules and re-run."]
    raise pytest.UsageError("\n".join(lines))
