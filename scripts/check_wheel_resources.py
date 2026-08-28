#!/usr/bin/env python3
"""Smoke-check resource files from an isolated core-pdf wheel installation."""

from __future__ import annotations

import struct
import sys
import sysconfig
from importlib import metadata, resources
from pathlib import Path


def internal_resource_bytes(package: str, *parts: str) -> bytes:
    resource = resources.files(package).joinpath(*parts)
    if not resource.is_file():
        joined = "/".join(parts)
        raise AssertionError(f"missing wheel resource: {package}/{joined}")
    return resource.read_bytes()


def internal_check_installed_import() -> None:
    import core_pdf

    if sys.prefix == sys.base_prefix:
        raise AssertionError("wheel smoke check must run in an isolated virtual environment")

    package_path = Path(core_pdf.__file__).resolve()
    site_roots = {
        Path(path).resolve()
        for key in ("purelib", "platlib")
        if (path := sysconfig.get_path(key)) is not None
    }
    if not any(package_path.is_relative_to(root) for root in site_roots):
        raise AssertionError(f"core_pdf was not imported from the isolated install: {package_path}")

    version = metadata.version("core-pdf")
    if not core_pdf.__all__:
        raise AssertionError("core_pdf public exports are empty")
    print(f"Imported core-pdf {version} from {package_path}")


def internal_check_wheel_inventory() -> None:
    files = metadata.files("core-pdf")
    if files is None:
        raise AssertionError("installed wheel does not expose a file inventory")
    paths = {entry.as_posix() for entry in files}
    expected_paths = {
        "core_pdf/impl/capture_model/__init__.py",
        "core_pdf/impl/capture_model/runs.py",
        "core_pdf/impl/engine/parse/block_layout.py",
    }
    missing = sorted(expected_paths - paths)
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise AssertionError(f"expected files missing from installed wheel:\n{formatted}")
    legacy_paths = sorted(
        path
        for path in paths
        if path.startswith("core_pdf/impl/engine/model/")
        or path == "core_pdf/impl/engine/parse/layout.py"
    )
    generated_paths = sorted(
        path for path in paths if "/__pycache__/" in path or path.endswith(".pyc")
    )
    unexpected = [*legacy_paths, *generated_paths]
    if unexpected:
        formatted = "\n".join(f"- {path}" for path in unexpected)
        raise AssertionError(f"unexpected files in installed wheel:\n{formatted}")
    print(f"Verified clean wheel inventory ({len(paths):,} files)")


def internal_check_cmap() -> None:
    data = internal_resource_bytes(
        "core_pdf.impl.spec.s_09_fonts.data",
        "cmaps",
        "Adobe-Japan1-7",
        "CMap",
        "90ms-RKSJ-H",
    )
    if not data.startswith(b"%!PS-Adobe-") or b"/CMapName /90ms-RKSJ-H def" not in data:
        raise AssertionError("packaged 90ms-RKSJ-H CMap has unexpected contents")
    print(f"Verified CMap resource ({len(data):,} bytes)")


def internal_check_raster_font() -> None:
    data = internal_resource_bytes(
        "core_pdf.impl.spec.s_09_fonts",
        "data",
        "raster_fonts",
        "LiberationSans-Regular.ttf",
    )
    if len(data) < 100_000 or data[:4] != b"\x00\x01\x00\x00":
        raise AssertionError("packaged raster font is missing or malformed")
    print(f"Verified raster font resource ({len(data):,} bytes)")


def internal_check_word_ranks() -> None:
    data = internal_resource_bytes(
        "core_pdf.impl.engine.layout.data.wordlists",
        "english_word_ranks.bin",
    )
    if len(data) < 1_000_000:
        raise AssertionError("packaged word-rank index is unexpectedly small")
    magic, count = struct.unpack_from("<8sI", data)
    if magic != b"CPWRANK1" or count < 100_000:
        raise AssertionError("packaged word-rank index has an invalid header")
    print(f"Verified word-rank resource ({count:,} entries)")


def internal_check_icc_profile() -> None:
    data = internal_resource_bytes(
        "core_pdf._vendor.icc",
        "SWOP2006_Coated5v2.icc",
    )
    declared_size = int.from_bytes(data[:4], "big")
    if declared_size != len(data) or data[16:20] != b"CMYK" or data[36:40] != b"acsp":
        raise AssertionError("packaged ICC profile has an invalid header")
    print(f"Verified ICC profile resource ({len(data):,} bytes)")


def internal_check_license_resources() -> None:
    checks = (
        (
            "core_pdf.impl.spec.s_07_filters.jbig2",
            b"GNU Affero General Public License version 3 only",
        ),
        (
            "core_pdf.impl.spec.s_07_security",
            b"GNU Affero General Public License v3.0 or later",
        ),
    )
    for package, expected_text in checks:
        data = internal_resource_bytes(package, "LICENSE.txt")
        if expected_text not in data:
            raise AssertionError(f"packaged license has unexpected contents: {package}")
    print("Verified JBIG2 and security license resources")


def main() -> None:
    internal_check_installed_import()
    internal_check_wheel_inventory()
    internal_check_cmap()
    internal_check_raster_font()
    internal_check_word_ranks()
    internal_check_icc_profile()
    internal_check_license_resources()


if __name__ == "__main__":
    main()
