# SPDX-License-Identifier: AGPL-3.0-only

"""Check every cited version rule on both sides of its boundary.

A `cites` marker with since or until (core_pdf.impl.spec.citations) says a function behaves
one way before a PDF version and another from it. This holds one case per such marker, keyed
by the function and the marker's rule, which runs the function at the last version before the
boundary and at the boundary and checks both results. It fails when a marker has no case, a
case names a marker that is gone, or a case does not hold.

uv run --all-packages python scripts/check_version_rules.py
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from core_pdf import EditableDocument, PdfDocument
from core_pdf.impl.document.navigation import legacy_name_context
from core_pdf.impl.document.standards import extension_diagnostics
from core_pdf.impl.spec.s_07_security.output import NewEncryption
from core_pdf.impl.spec.s_07_syntax_primitives.text_string import decode_pdf_text_string
from core_pdf.impl.spec.s_07_syntax_primitives.tokens import lexical_rules
from core_pdf.impl.spec.s_08_graphics.color_spec import ColorSpace, parse_color_space
from core_pdf.impl.spec.s_08_graphics.image_spec import image_decode_array_applies
from core_pdf.impl.spec.s_09_fonts.helpers import get_base_encoding_glyph_names
from core_pdf.impl.spec.s_11_transparency.blend import revised_blending
from core_pdf.impl.spec.s_14_structure.roles import resolve_structure_role
from core_pdf.impl.spec.standards import (
    DocumentStandards,
    PdfExtension,
    PdfVersion,
    SemanticContext,
)
from core_pdf.impl.spec.types import PdfName

ROOT = Path(__file__).resolve().parent.parent


def load_spec_index() -> Any:
    """scripts/spec_index.py, which finds the cites markers."""
    location = importlib.util.spec_from_file_location(
        "spec_index", ROOT / "scripts" / "spec_index.py"
    )
    assert location is not None and location.loader is not None
    module = importlib.util.module_from_spec(location)
    location.loader.exec_module(module)
    return module


# A PDF 1.4 file, small enough to save in every way.
PDF_1_4 = ROOT / "tests/upstream/pypdf/resources/toy.pdf"


def context(version: str) -> SemanticContext:
    return SemanticContext(PdfVersion.parse(version))


def raises(call: Callable[[], object]) -> bool:
    try:
        call()
    except ValueError:
        return True
    return False


def indexed_separation(version: str) -> ColorSpace:
    tint = {"FunctionType": 2, "Domain": [0, 1], "C0": [0.0], "C1": [1.0], "N": 1}
    separation = [PdfName(b"Separation"), PdfName(b"Spot"), PdfName(b"DeviceGray"), tint]
    return parse_color_space(
        [PdfName(b"Indexed"), separation, 1, b"\x00\xff"], context=context(version)
    )


def mapped_role(version: str) -> str:
    return resolve_structure_role(
        PdfName(b"P"), role_map={"P": PdfName(b"H1")}, context=context(version)
    ).name


def missing_url_diagnosed(version: str) -> bool:
    standards = DocumentStandards(
        effective_version=PdfVersion.parse(version),
        extensions=(PdfExtension("XYZW", PdfVersion(1, 7), 1),),
    )
    return any(
        diagnostic.code == "missing-extension-url"
        for diagnostic in extension_diagnostics(standards).diagnostics
    )


def saved_header(
    *, xref_stream: bool = False, encryption: Literal["keep"] | NewEncryption = "keep"
) -> bytes:
    with PdfDocument.open(PDF_1_4.read_bytes()) as document:
        return EditableDocument(document).save(encryption=encryption, xref_stream=xref_stream)[:8]


JPX_IMAGE: dict[object, object] = {
    "Filter": PdfName(b"JPXDecode"),
    "ColorSpace": PdfName(b"DeviceRGB"),
}

# (module.function, rule) -> whether the rule holds before its boundary and from it.
CASES: dict[tuple[str, str], Callable[[], bool]] = {
    (
        "core_pdf.impl.document.navigation.legacy_name_context",
        "a # in a name is a literal character",
    ): lambda: legacy_name_context(context("1.1")) and not legacy_name_context(context("1.2")),
    (
        "core_pdf.impl.spec.s_07_syntax_primitives.text_string.decode_pdf_text_string",
        "UTF-16BE text strings, after the byte order mark FE FF",
    ): lambda: (
        decode_pdf_text_string(b"\xfe\xff\x00A", context=context("1.1")) != "A"
        and decode_pdf_text_string(b"\xfe\xff\x00A", context=context("1.2")) == "A"
    ),
    (
        "core_pdf.impl.spec.s_07_syntax_primitives.text_string.decode_pdf_text_string",
        "UTF-8 text strings, after the byte order mark EF BB BF",
    ): lambda: (
        decode_pdf_text_string(b"\xef\xbb\xbfA", context=context("1.7")) != "A"
        and decode_pdf_text_string(b"\xef\xbb\xbfA", context=context("2.0")) == "A"
    ),
    (
        "core_pdf.impl.spec.s_07_syntax_primitives.text_string.decode_pdf_text_string",
        "PDFDocEncoding defines byte A0 (Euro sign)",
    ): lambda: (
        raises(lambda: decode_pdf_text_string(b"\xa0", context=context("1.2")))
        and decode_pdf_text_string(b"\xa0", context=context("1.3")) == "€"
    ),
    (
        "core_pdf.impl.spec.s_07_syntax_primitives.tokens.lexical_rules",
        "a # in a name starts a hexadecimal escape",
    ): lambda: (
        not lexical_rules(context("1.1")).name_escapes
        and lexical_rules(context("1.2")).name_escapes
    ),
    (
        "core_pdf.impl.spec.s_07_syntax_primitives.tokens.lexical_rules",
        "NUL is a white-space character",
    ): lambda: (
        b"\x00" not in lexical_rules(context("1.2")).whitespace
        and b"\x00" in lexical_rules(context("1.3")).whitespace
    ),
    (
        "core_pdf.impl.spec.s_07_syntax_primitives.tokens.lexical_rules",
        "{ and } delimit only PostScript calculator functions",
    ): lambda: (
        b"{" in lexical_rules(context("1.7")).delimiters
        and b"{" not in lexical_rules(context("2.0")).delimiters
    ),
    (
        "core_pdf.impl.spec.s_07_syntax_primitives.tokens.lexical_rules",
        "object identifiers are written in canonical form",
    ): lambda: (
        not lexical_rules(context("1.7")).canonical_identifiers
        and lexical_rules(context("2.0")).canonical_identifiers
    ),
    (
        "core_pdf.impl.spec.s_08_graphics.color_spec.parse_color_space_versioned",
        "an Indexed colour space may have a Separation or DeviceN base",
    ): lambda: (
        raises(lambda: indexed_separation("1.2")) and indexed_separation("1.3").kind == "Indexed"
    ),
    (
        "core_pdf.impl.spec.s_08_graphics.image_spec.image_decode_array_applies",
        "a JPX image's Decode array applies when it has a ColorSpace",
    ): lambda: (
        not image_decode_array_applies(JPX_IMAGE, context=context("1.7"))
        and image_decode_array_applies(JPX_IMAGE, context=context("2.0"))
    ),
    (
        "core_pdf.impl.spec.s_09_fonts.helpers.get_base_encoding_glyph_names",
        "WinAnsiEncoding 80, 8E and 9E are Euro, Zcaron and zcaron, not bullet",
    ): lambda: (
        get_base_encoding_glyph_names("WinAnsiEncoding", context=context("1.2"))[0x80] == "bullet"
        and get_base_encoding_glyph_names("WinAnsiEncoding", context=context("1.3"))[0x80] == "Euro"
    ),
    (
        "core_pdf.impl.spec.s_11_transparency.blend.revised_blending",
        "revised ColorDodge and ColorBurn",
    ): lambda: not revised_blending(context("1.7")) and revised_blending(context("2.0")),
    (
        "core_pdf.impl.spec.s_14_structure.roles.resolve_structure_role",
        "a role map can remap standard structure types",
    ): lambda: mapped_role("1.4") == "P" and mapped_role("1.5") == "H1",
    (
        "core_pdf.impl.document.standards.extension_diagnostics",
        "a developer extension declares its URL",
    ): lambda: not missing_url_diagnosed("1.7") and missing_url_diagnosed("2.0"),
    ("core_pdf.impl.document.editing.save", "cross-reference streams"): lambda: (
        saved_header() == b"%PDF-1.4" and saved_header(xref_stream=True) == b"%PDF-1.5"
    ),
    (
        "core_pdf.impl.document.editing.save",
        "AES-256 encryption (PDF 1.7 with an extension, or 2.0)",
    ): lambda: saved_header(encryption=NewEncryption("user")) == b"%PDF-1.7",
}


def main() -> int:
    markers = {
        (f"{marked.module}.{marked.name}", citation.rule)
        for marked in load_spec_index().survey().functions
        for citation in marked.citations
        if citation.since or citation.until
    }
    problems = [
        f"no case for the version rule {name}: {rule!r}"
        for name, rule in sorted(markers - CASES.keys())
    ]
    problems += [
        f"a case names no version rule: {name}: {rule!r}"
        for name, rule in sorted(CASES.keys() - markers)
    ]
    for (name, rule), case in CASES.items():
        if (name, rule) in markers and not case():
            problems.append(f"{name}: {rule!r} does not hold on both sides of its boundary")
    for problem in problems:
        print(problem)
    if problems:
        return 1
    print(f"{len(CASES)} version rules hold on both sides of their boundaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
