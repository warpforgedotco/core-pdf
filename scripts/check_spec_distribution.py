"""Verify built workspace wheels in an isolated installation outside the checkout."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

SPEC_SMOKE = """
import importlib
import importlib.util
import pkgutil
import zlib
from importlib import resources

import core_pdf_spec
from core_pdf_spec.s_07_filters.pipeline import decode_stream_data
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.s_09_fonts.cmap_resources import resolve_cmap_decoder
from core_pdf_spec.s_09_fonts.glyphs import glyph_name_to_unicode

for module in pkgutil.walk_packages(core_pdf_spec.__path__, core_pdf_spec.__name__ + "."):
    importlib.import_module(module.name)
assert importlib.util.find_spec("core_pdf") is None
assert importlib.util.find_spec("core_pdf_ocr") is None
assert importlib.util.find_spec("fontTools") is None
assert importlib.util.find_spec("imagecodecs") is None
lexer = PdfLexer(b"<< /Type /Page >>")
try:
    assert str(lexer.parse_dictionary_or_stream()["Type"]) == "Page"
finally:
    lexer.close()
assert decode_stream_data(zlib.compress(b"PDF"), {"Filter": "FlateDecode"}) == b"PDF"
assert resolve_cmap_decoder("UniJIS-UTF16-V") is not None
assert glyph_name_to_unicode("uni00410042") == "AB"
assert resources.files("core_pdf_spec").joinpath("py.typed").is_file()
data = resources.files("core_pdf_spec._vendor.font_data")
assert data.joinpath("LICENSE.txt").is_file()
assert data.joinpath("LICENSE.external.txt").is_file()
print("Standalone spec wheel: imports, parsing, decoding, CMaps and notices passed")
"""

CORE_SMOKE = """
import sys
from pathlib import Path
import core_pdf
import core_pdf_spec.exceptions as errors
from core_pdf import PdfDocument
assert core_pdf.PdfError is errors.PdfError
assert core_pdf.PdfParseError is errors.PdfParseError
with PdfDocument(Path(sys.argv[1])) as document:
    assert document.page_count() > 0
assert not any(name == "core_pdf_ocr" or name.startswith("core_pdf_ocr.") for name in sys.modules)
print("Core wheel: public API, parsing and exception identities passed")
"""

OCR_SMOKE = """
import core_pdf
import core_pdf_ocr
assert issubclass(core_pdf_ocr.PdfDocument, core_pdf.PdfDocument)
assert issubclass(core_pdf_ocr.PdfPage, core_pdf.PdfPage)
print("OCR wheel: document/page integration passed")
"""


def run(*arguments: str, cwd: Path) -> None:
    subprocess.run(arguments, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing the three built wheels")
    parser.add_argument("--fixture", type=Path, required=True, help="A valid one-page PDF")
    args = parser.parse_args()
    wheel_dir = args.directory.resolve()
    wheels: dict[str, Path] = {}
    for name in ("core_pdf_spec", "core_pdf", "core_pdf_ocr"):
        matches = sorted(wheel_dir.glob(f"{name}-[0-9]*.whl"))
        if len(matches) != 1:
            parser.error(f"expected one {name} wheel in {wheel_dir}; found {len(matches)}")
        wheels[name] = matches[0]
    with ZipFile(wheels["core_pdf_spec"]) as archive:
        assert not any(
            "__pycache__" in name or name.endswith((".pyc", ".so")) for name in archive.namelist()
        ), "the spec wheel contains cached or compiled source artifacts"
    with tempfile.TemporaryDirectory(prefix="core-pdf-spec-wheel-") as temporary:
        work = Path(temporary)
        environment = work / "environment"
        run("uv", "venv", "--python", sys.executable, str(environment), cwd=work)
        python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        install = ("uv", "pip", "install", "--python", str(python), "--find-links", str(wheel_dir))
        run(*install, str(wheels["core_pdf_spec"]), cwd=work)
        run(str(python), "-I", "-c", SPEC_SMOKE, cwd=work)
        run(*install, str(wheels["core_pdf"]), cwd=work)
        run(str(python), "-I", "-c", CORE_SMOKE, str(args.fixture.resolve()), cwd=work)
        run(str(python), "-I", "-m", "core_pdf", "--help", cwd=work)
        run(*install, str(wheels["core_pdf_ocr"]), cwd=work)
        run(str(python), "-I", "-c", OCR_SMOKE, cwd=work)


if __name__ == "__main__":
    main()
