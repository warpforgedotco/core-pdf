from __future__ import annotations

import pathlib
import tomllib

import pytest

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent


def source_roots() -> tuple[pathlib.Path, ...]:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as handle:
        sources = tomllib.load(handle)["tool"]["coverage"]["run"]["source"]
    return tuple(REPOSITORY_ROOT / source for source in sources)


SOURCE_ROOTS = source_roots()
EXTENSION_SUFFIXES = (".so", ".pyd", ".dylib")


@pytest.fixture
def text_pdf_bytes() -> bytes:
    content = b"BT /F1 12 Tf 20 100 Td (Hello maintenance) Tj ET"
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
    )
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, value in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode() + value + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(data)


def shadowed_modules() -> list[tuple[pathlib.Path, pathlib.Path]]:
    shadowed: list[tuple[pathlib.Path, pathlib.Path]] = []
    for root in SOURCE_ROOTS:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in EXTENSION_SUFFIXES:
                continue
            source = path.parent / f"{path.name.split('.')[0]}.py"
            if source.is_file():
                shadowed.append((path, source))
    return shadowed


def pytest_configure() -> None:
    shadowed = shadowed_modules()
    if not shadowed:
        return
    lines = [
        "Compiled extension modules are shadowing their Python sources.",
        "Tests would run a compiled snapshot instead of the workspace sources.",
        "",
    ]
    lines += [
        f"  {extension.relative_to(REPOSITORY_ROOT)} shadows {source.relative_to(REPOSITORY_ROOT)}"
        for extension, source in shadowed
    ]
    lines += ["", "Delete the extension modules and re-run."]
    raise pytest.UsageError("\n".join(lines))
