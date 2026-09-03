"""Session-wide guards that run before any test is collected."""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess

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


def internal_disable_benchmarks_by_default(config: pytest.Config) -> None:
    """Make ``--benchmark-disable`` the default for plain ``pytest`` runs.

    Benchmarked tests then execute once as ordinary tests with no timing
    rounds or stats. Pass ``--benchmark-enable`` or ``--benchmark-only`` to
    opt back in. Done here rather than in ``addopts`` because pytest-benchmark
    lives in the optional ``benchmark`` dependency group: an ``addopts`` flag
    would fail as unrecognised whenever the plugin is not installed.
    """
    if not config.pluginmanager.hasplugin("benchmark"):
        return
    if config.getoption("benchmark_enable") or config.getoption("benchmark_only"):
        return
    config.option.benchmark_disable = True


def internal_export_tessdata_prefix() -> None:
    """Resolve the Tesseract data directory once for the whole session.

    ``ocr_tesseract.internal_resolve_tessdata_path`` honours ``TESSDATA_PREFIX``
    before anything else, and without it every xdist worker re-derives the
    directory through ``tesserocr.get_languages()`` (about 2.4 s each). One
    ``tesseract --list-langs`` here, before workers fork, makes that lookup free.
    The tessdata tests set or clear the variable themselves.
    """
    if os.environ.get("TESSDATA_PREFIX"):
        return
    executable = shutil.which("tesseract")
    if executable is None:
        return
    try:
        completed = subprocess.run(
            [executable, "--list-langs"], capture_output=True, check=False, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return
    match = re.search(
        r'List of available languages in "([^"]+)"', completed.stdout + completed.stderr
    )
    if match is None:
        return
    tessdata = pathlib.Path(match.group(1)).expanduser()
    if (tessdata / "eng.traineddata").is_file():
        os.environ["TESSDATA_PREFIX"] = str(tessdata.resolve())


def pytest_configure(config: pytest.Config) -> None:
    internal_disable_benchmarks_by_default(config)
    internal_export_tessdata_prefix()
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
