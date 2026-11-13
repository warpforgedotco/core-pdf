# SPDX-License-Identifier: AGPL-3.0-only

"""Check core's font program readers against fontTools, which they replace.

core reads TrueType, OpenType CFF/CFF2 and Type 1 font programs itself, matching fontTools
4.63.0's results, damaged fonts included. This compares the two on every font embedded in
the upstream corpora, and on OpenType CFF and CFF2 fonts built from the corpora's CFF fonts
(the corpora embed almost none), plus a variable CFF2 font that uses blend:

- TrueType: the glyph names, the Unicode and code cmaps, the metrics and loca table, each
  composite's components, and the drawing of sampled glyphs, pen operation for operation;
- OpenType: the glyph names, and each sampled glyph's decomposed drawing and bounds;
- Type 1: each sampled glyph's drawing, and its bounds where they are drawn in Python.

Drawings and bounds must be equal to the last bit; a read that fails must fail on both
sides. Needs fontTools and the upstream submodules:

uv run --all-packages --with fonttools==4.63.0 python scripts/check_font_programs.py
"""

from __future__ import annotations

import hashlib
import io
import sys
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from fontTools.cffLib.CFFToCFF2 import convertCFFToCFF2  # ty: ignore[unresolved-import]
from fontTools.fontBuilder import FontBuilder  # ty: ignore[unresolved-import]
from fontTools.misc.psCharStrings import T1CharString, T2CharString  # ty: ignore[unresolved-import]
from fontTools.pens.boundsPen import BoundsPen  # ty: ignore[unresolved-import]
from fontTools.pens.recordingPen import DecomposingRecordingPen, RecordingPen  # ty: ignore[unresolved-import]
from fontTools.pens.transformPen import TransformPen  # ty: ignore[unresolved-import]
from fontTools.ttLib import TTFont, newTable  # ty: ignore[unresolved-import]

from core_pdf import PdfDocument
from core_pdf.impl.base.types import PdfReference
from core_pdf.impl.fonts import program_truetype as core_truetype
from core_pdf.impl.fonts.outline_pens import BoundsSink, Type1Outline, draw_decomposed
from core_pdf.impl.fonts.program_truetype import OpenTypeFontProgram, TrueTypeFontProgram
from core_pdf.impl.fonts.program_type1 import Type1FontProgram
from core_pdf.impl.fonts.sfnt import GlyphOutlines, placeholder_glyph_id, placeholder_glyph_order
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream

ROOT = Path(__file__).resolve().parent.parent
CORPORA = ROOT / "tests/upstream"
GLYPHS_PER_FONT = 96
MAX_PDF_SIZE = 20_000_000


def outcome(function: Callable[[], Any]) -> Any:
    try:
        return ("ok", function())
    except Exception as exc:  # noqa: BLE001
        return ("error", type(exc).__name__)


def failed(result: Any) -> bool:
    return result[0] == "error"


def same(a: Any, b: Any) -> bool:
    """Equal results, or both failures of whatever kind."""
    return (failed(a) and failed(b)) or a == b


def sampled(count: int) -> list[int]:
    if count <= GLYPHS_PER_FONT:
        return list(range(count))
    step = count / GLYPHS_PER_FONT
    return sorted({int(index * step) for index in range(GLYPHS_PER_FONT)} | {count - 1})


def embedded_fonts() -> Iterator[tuple[str, str, bytes, int | None]]:
    """(source, kind, data, Length1) of each distinct font program in the corpora."""
    seen: set[bytes] = set()
    for path in sorted(CORPORA.glob("**/*.pdf")):
        if not path.is_file() or path.stat().st_size > MAX_PDF_SIZE:
            continue
        try:
            document = PdfDocument(path)
        except Exception:  # noqa: BLE001
            continue
        with document:
            for key in list(document.xref):
                try:
                    obj = document.resolve(PdfReference(key >> 16, key & 0xFFFF))
                except Exception:  # noqa: BLE001
                    continue
                if not isinstance(obj, dict):
                    continue
                for name in ("FontFile", "FontFile2", "FontFile3"):
                    reference = obj.get(name)
                    if reference is None:
                        continue
                    try:
                        stream = document.resolve(reference)
                        data = bytes(stream.data) if isinstance(stream, PdfStream) else None
                    except Exception:  # noqa: BLE001
                        continue
                    if data is None:
                        continue
                    digest = hashlib.sha256(data).digest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    assert isinstance(stream, PdfStream)
                    subtype = str(stream.dictionary.get("Subtype", "")).lstrip("/")
                    length1 = stream.dictionary.get("Length1")
                    kind = name if name != "FontFile3" else f"FontFile3/{subtype}"
                    yield (
                        str(path.relative_to(ROOT)),
                        kind,
                        data,
                        length1 if isinstance(length1, int) else None,
                    )


# TrueType: fontTools with the glyph naming core keeps.


def reference_truetype(data: bytes) -> TTFont:
    font = TTFont(io.BytesIO(data), lazy=True)
    if not {"maxp", "glyf", "loca", "head"} <= set(font.keys()):
        raise ValueError("invalid TrueType glyph tables")
    placeholder = False

    def placeholder_names() -> None:
        nonlocal placeholder
        order = placeholder_glyph_order(int(font["maxp"].numGlyphs))
        order[0] = ".notdef"
        font.glyphOrder = order
        placeholder = True
        if "cmap" in font:
            font["cmap"].buildReversedMin()

    font.__dict__["_getGlyphNamesFromCmap"] = placeholder_names
    try:
        font.getGlyphOrder()
    except Exception:  # noqa: BLE001
        count = int(font["maxp"].numGlyphs)
        if count <= 0:
            raise ValueError("invalid TrueType glyph order") from None
        font.setGlyphOrder(placeholder_glyph_order(count))
        placeholder = True
    finally:
        font.__dict__.pop("_getGlyphNamesFromCmap", None)
    font.__dict__["placeholder"] = placeholder
    return font


def reference_gid_for(font: TTFont) -> Callable[[str], int | None]:
    if font.__dict__["placeholder"]:
        return placeholder_glyph_id
    reverse = font.getReverseGlyphMap()
    return lambda name: reverse[name] if name in reverse else placeholder_glyph_id(name)


def reference_unicode_cmap(font: TTFont) -> dict[int, int]:
    symbol = False
    try:
        cmap = font["cmap"]
        names = cmap.getBestCmap()
        if names is None:
            table = cmap.getcmap(3, 0)
            names = table.cmap if table is not None else {}
            symbol = bool(names)
        gid_for = reference_gid_for(font)
    except Exception:  # noqa: BLE001
        return {}
    mapping: dict[int, int] = {}
    for code, name in names.items():
        if not core_truetype.is_unicode_scalar(code):
            continue
        gid = gid_for(name)
        if gid is not None and gid > 0:
            mapping[code] = gid
            if symbol and 0xF000 <= code <= 0xF2FF:
                mapping.setdefault(core_truetype.symbol_character_code(code), gid)
    return mapping


def reference_code_cmap(font: TTFont) -> dict[int, int]:
    try:
        cmap = font["cmap"]
        gid_for = reference_gid_for(font)
    except Exception:  # noqa: BLE001
        return {}
    for platform, encoding in ((3, 0), (1, 0)):
        try:
            table = cmap.getcmap(platform, encoding)
        except Exception:  # noqa: BLE001
            continue
        if table is None:
            continue
        mapping: dict[int, int] = {}
        for code, name in table.cmap.items():
            gid = gid_for(name)
            if gid is None or gid <= 0:
                continue
            mapping.setdefault(code, gid)
            if platform == 3 and 0xF000 <= code <= 0xF0FF:
                mapping.setdefault(code - 0xF000, gid)
        if mapping:
            return mapping
    return {}


def reference_drawing(font: TTFont, name: str, glyph_set: Any) -> list[Any]:
    pen = DecomposingRecordingPen(glyph_set, skipMissingComponents=True)
    glyph_set[name].draw(pen)
    return [(operator, tuple(points)) for operator, points in pen.value]


def check_truetype(data: bytes) -> list[str]:
    reference = outcome(lambda: reference_truetype(data))
    core = outcome(lambda: TrueTypeFontProgram(data))
    if failed(reference) or failed(core):
        return [] if failed(reference) and failed(core) else [f"build: {reference} vs {core}"]
    font, program = reference[1], core[1]
    sfnt = program.font
    problems = []
    if list(font.getGlyphOrder()) != sfnt.glyph_order:
        problems.append("glyph order")
    if font.__dict__["placeholder"] != sfnt.placeholder:
        problems.append("placeholder names")
    upem = outcome(lambda: float(getattr(font["head"], "unitsPerEm", 1000) or 1000))
    if not same(upem, ("ok", program.units_per_em)):
        problems.append("unitsPerEm")
    if reference_unicode_cmap(font) != program.unicode_cmap:
        problems.append("Unicode cmap")
    if not same(
        outcome(lambda: reference_code_cmap(font)),
        outcome(lambda: core_truetype.code_gid_cmap(sfnt)),
    ):
        problems.append("code cmap")
    for tag in ("hmtx", "vmtx"):
        if tag in font or tag in sfnt:
            ours = outcome(lambda tag=tag: dict(sfnt.table(tag).metrics))
            theirs = outcome(lambda tag=tag: {k: tuple(v) for k, v in font[tag].metrics.items()})
            if not same(theirs, ours):
                problems.append(tag)
    if not same(
        outcome(lambda: list(font["loca"].locations)),
        outcome(lambda: list(sfnt.table("loca").locations)),
    ):
        problems.append("loca")
    glyph_set = outcome(font.getGlyphSet)
    outlines = outcome(lambda: GlyphOutlines(sfnt))
    if failed(glyph_set) != failed(outlines):
        problems.append(f"glyph set: {glyph_set} vs {outlines}")
    count = len(sfnt.glyph_order)
    for gid in sampled(count):
        name = sfnt.glyph_name(gid)
        components = outcome(
            lambda name=name: (
                [c.getComponentInfo() for c in font["glyf"][name].components]
                if font["glyf"][name].isComposite()
                else None
            )
        )
        ours_components = outcome(
            lambda name=name: (
                [
                    (c.name, c.transform())
                    for c in core_truetype.glyf_glyph(sfnt.table("glyf"), name).components
                ]
                if core_truetype.glyf_glyph(sfnt.table("glyf"), name).is_composite()
                else None
            )
        )
        if not same(components, ours_components):
            problems.append(f"components of glyph {gid}")
        if not failed(glyph_set) and not failed(outlines):
            theirs = outcome(lambda name=name: reference_drawing(font, name, glyph_set[1]))
            ours = outcome(lambda name=name: outlines[1].record(name))
            if not same(theirs, ours):
                problems.append(f"drawing of glyph {gid}")
    return problems


# OpenType CFF and CFF2.


def opentype_drawing(font: TTFont, glyph_id: int, transform: tuple[float, ...] | None) -> Any:
    glyph_set = font.getGlyphSet()
    name = font.getGlyphName(glyph_id)
    if transform is None:
        return reference_drawing(font, name, glyph_set)
    pen = BoundsPen(glyph_set)
    glyph_set[name].draw(TransformPen(pen, transform))
    return None if pen.bounds is None else tuple(float(value) for value in pen.bounds)


def check_opentype(data: bytes) -> list[str]:
    def reference_font() -> TTFont:
        font = TTFont(io.BytesIO(data), lazy=True, recalcBBoxes=False, recalcTimestamp=False)
        if not {"CFF ", "CFF2"} & set(font.keys()):
            raise ValueError("no CFF outline table")
        font.getGlyphOrder()
        float(getattr(font["head"], "unitsPerEm", 1000) or 1000)
        if "CFF2" in font and "fvar" in font:
            del font["fvar"]
        return font

    reference = outcome(reference_font)
    core = outcome(lambda: OpenTypeFontProgram(data))
    if failed(reference) or failed(core):
        return [] if failed(reference) and failed(core) else [f"build: {reference} vs {core}"]
    font, program = reference[1], core[1]
    problems = []
    if list(font.getGlyphOrder()) != program.font.glyph_order:
        problems.append("glyph order")
    scale = (program.scale, 0.0, 0.0, program.scale, 0.0, 0.0)
    for gid in sampled(program.glyph_count):
        theirs = outcome(lambda gid=gid: opentype_drawing(font, gid, None))
        recording: list[Any] = []
        ours = outcome(
            lambda gid=gid, recording=recording: (
                program.draw(gid, None, recording.append) or recording
            )
        )
        if not same(theirs, ours):
            problems.append(f"drawing of glyph {gid}")
        theirs = outcome(lambda gid=gid: opentype_drawing(font, gid, scale))
        sink = BoundsSink()
        ours = outcome(
            lambda gid=gid, sink=sink: (
                program.draw(gid, scale, sink)
                or (None if sink.bounds is None else tuple(float(v) for v in sink.bounds))
            )
        )
        if not same(theirs, ours):
            problems.append(f"bounds of glyph {gid}")
    return problems


def opentype_from_cff(data: bytes, *, cff2: bool, named: bool) -> bytes:
    """An OpenType font around a bare CFF program; with cff2, its CFF2 conversion. A font
    whose post table does not name its glyphs gets names from a cmap of its first glyphs."""
    font = TTFont(sfntVersion="OTTO")
    cff = newTable("CFF ")
    cff.decompile(data, font)
    font["CFF "] = cff
    order = cff.getGlyphOrder()
    font.setGlyphOrder(order)
    builder = FontBuilder(font=font, isTTF=False)
    builder.setupHead(unitsPerEm=1000)
    builder.setupHorizontalMetrics(dict.fromkeys(order, (500, 0)))
    builder.setupHorizontalHeader()
    builder.setupCharacterMap({0x41 + index: name for index, name in enumerate(order[1:30])})
    builder.setupMaxp()
    builder.setupPost(keepGlyphNames=named)
    builder.setupOS2()
    builder.setupNameTable({"familyName": "Check", "styleName": "Regular"})
    if cff2:
        convertCFFToCFF2(font)
    buffer = io.BytesIO()
    font.save(buffer)
    return buffer.getvalue()


def variable_cff2() -> bytes:
    """A variable CFF2 font whose outlines use vsindex and blend (three regions, so each
    blended value carries three deltas)."""
    builder = FontBuilder(1000, isTTF=False)
    order = [".notdef", "A", "B", "C"]
    builder.setupGlyphOrder(order)
    builder.setupCharacterMap({0x41: "A", 0x42: "B", 0x43: "C"})
    builder.setupNameTable({"familyName": "Check", "styleName": "Regular"})
    builder.setupFvar([("wght", 100, 400, 900, "Weight")], [])
    charstrings = {
        ".notdef": [100, 0, "rmoveto", 300, 0, 0, 500, -300, 0, "rlineto"],
        "A": [
            100,
            10,
            20,
            30,
            1,
            "blend",
            0,
            "rmoveto",
            400,
            0,
            5,
            6,
            7,
            1,
            1,
            1,
            2,
            "blend",
            "rlineto",
            -200,
            600,
            "rlineto",
        ],
        "B": [
            0,
            "vsindex",
            50,
            50,
            1,
            2,
            3,
            4,
            5,
            6,
            2,
            "blend",
            "rmoveto",
            100,
            200,
            300,
            400,
            500,
            600,
            "rrcurveto",
        ],
        "C": [10, 20, "rmoveto", 30, 40, 50, 60, 70, 80, 9, 8, 7, 1, "blend", "rrcurveto"],
    }
    builder.setupCFF2(
        {name: T2CharString(program=program) for name, program in charstrings.items()},
        regions=[{"wght": (0, 1, 1)}, {"wght": (0, 0.5, 1)}, {"wght": (0.5, 1, 1)}],
    )
    builder.setupHorizontalMetrics(dict.fromkeys(order, (500, 0)))
    builder.setupHorizontalHeader()
    builder.setupPost()
    builder.setupOS2()
    buffer = io.BytesIO()
    builder.save(buffer)
    return buffer.getvalue()


def cubic_glyf() -> bytes:
    """A TrueType font with cubic glyf contours (flag bit 0x80), which no corpus font has:
    on-curve points between cubic segments, a run of four cubic off-curve points (with
    implied on-curve midpoints), a contour of cubic off-curve points only, a quadratic
    contour beside a cubic one, and a composite placing the cubic glyph scaled and moved."""
    from fontTools.ttLib.tables import ttProgram  # noqa: PLC0415  # ty: ignore[unresolved-import]
    from fontTools.ttLib.tables._g_l_y_f import Glyph, GlyphComponent, GlyphCoordinates  # noqa: PLC0415  # ty: ignore[unresolved-import]

    on, off, cubic = 0x01, 0x00, 0x80

    def simple(*contours: list[tuple[int, int, int]]) -> Any:
        glyph = Glyph()
        points = [point for contour in contours for point in contour]
        glyph.coordinates = GlyphCoordinates([(x, y) for x, y, _ in points])
        glyph.flags = bytearray(flag for _, _, flag in points)
        ends, total = [], 0
        for contour in contours:
            total += len(contour)
            ends.append(total - 1)
        glyph.endPtsOfContours = ends
        glyph.numberOfContours = len(contours)
        glyph.program = ttProgram.Program()
        glyph.program.fromBytecode(b"")
        return glyph

    glyphs = {
        ".notdef": simple([(50, 0, on), (450, 0, on), (450, 700, on), (50, 700, on)]),
        "cubic": simple(
            [
                (100, 0, on),
                (150, 200, cubic),
                (350, 200, cubic),
                (400, 0, on),
                (420, 100, cubic),
                (440, 300, cubic),
                (420, 500, cubic),
                (380, 650, cubic),
                (250, 700, on),
            ],
            [(100, 300, cubic), (100, 500, cubic), (300, 500, cubic), (300, 300, cubic)],
        ),
        "mixed": simple(
            [(0, 0, on), (250, 400, off), (500, 0, on)],
            [(100, 450, on), (150, 650, cubic), (350, 650, cubic), (400, 450, on)],
        ),
    }
    composite = Glyph()
    composite.numberOfContours = -1
    component = GlyphComponent()
    component.glyphName = "cubic"
    component.x, component.y = 37, -11
    component.flags = 0x0004
    component.transform = [[0.75, 0], [0, 0.5]]
    composite.components = [component]
    glyphs["composite"] = composite
    order = list(glyphs)
    builder = FontBuilder(1000, isTTF=True)
    builder.setupHead(unitsPerEm=1000, glyphDataFormat=1)  # 1: the glyf table has cubics
    builder.setupGlyphOrder(order)
    builder.setupCharacterMap({0x41 + index: name for index, name in enumerate(order[1:])})
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics(dict.fromkeys(order, (500, 0)))
    builder.setupHorizontalHeader()
    builder.setupPost()
    builder.setupOS2()
    builder.setupNameTable({"familyName": "Check", "styleName": "Regular"})
    buffer = io.BytesIO()
    builder.save(buffer)
    return buffer.getvalue()


def check_program_contours(data: bytes) -> list[str]:
    """What the TrueType program returns, compiled kernel first, against fontTools."""
    font = TTFont(io.BytesIO(data), lazy=True)
    glyph_set = font.getGlyphSet()
    program = TrueTypeFontProgram(data)
    scale = 1000.0 / font["head"].unitsPerEm
    problems = []
    for gid, name in enumerate(font.getGlyphOrder()):
        recording = reference_drawing(font, name, glyph_set)
        theirs = tuple(tuple(c) for c in core_truetype.recording_to_contours(recording))
        if scale != 1.0:
            theirs = core_truetype.scale_contours(theirs, scale)
        if program.normalized_glyph_contours(gid) != theirs:
            problems.append(f"contours of {name}")
    return problems


# Type 1.


def check_type1(data: bytes, length1: int | None) -> list[str]:
    core = outcome(lambda: Type1FontProgram(data, length1=length1))
    if failed(core):
        return []
    program = core[1]
    subrs: list[T1CharString] = []
    subrs.extend(T1CharString(subr, subrs=subrs) for subr in program.subr_data)
    charstrings = {
        name: T1CharString(charstring, subrs=subrs)
        for name, charstring in program.charstring_data.items()
    }
    problems = []
    a, b, c, d, e, f = program.font_matrix
    normalizing = (a * 1000.0, b * 1000.0, c * 1000.0, d * 1000.0, e * 1000.0, f * 1000.0)
    for gid in sampled(len(program.glyph_names)):
        name = program.glyph_names[gid]

        def reference_recording(name: str = name) -> list[Any]:
            pen = RecordingPen()
            charstrings[name].draw(pen)
            return [(operator, tuple(points)) for operator, points in pen.value]

        def reference_bounds(name: str = name) -> Any:
            pen = BoundsPen(charstrings)
            charstrings[name].draw(TransformPen(pen, normalizing))
            return pen.bounds

        def core_bounds(name: str = name) -> Any:
            sink = BoundsSink()
            draw_decomposed(
                program.record, name, normalizing, sink, program.charstring_data.__contains__
            )
            return sink.bounds

        ours = outcome(
            lambda name=name: Type1Outline(program.subr_data).record(program.charstring_data[name])
        )
        if not same(outcome(reference_recording), ours):
            problems.append(f"drawing of {name}")
        if not same(outcome(reference_bounds), outcome(core_bounds)):
            problems.append(f"bounds of {name}")
    return problems


def main() -> int:
    results: Counter[str] = Counter()
    failures: list[str] = []

    def record(kind: str, source: str, problems: list[str]) -> None:
        results[kind] += 1
        if problems:
            failures.append(f"{kind} {source}: {', '.join(sorted(set(problems))[:6])}")

    cff_programs: list[tuple[str, bytes]] = []
    for source, kind, data, length1 in embedded_fonts():
        if kind == "FontFile2":
            record("TrueType", source, check_truetype(data))
        elif kind == "FontFile3/OpenType":
            record("OpenType", source, check_opentype(data))
        elif kind == "FontFile":
            record("Type 1", source, check_type1(data, length1))
        elif kind in ("FontFile3/Type1C", "FontFile3/CIDFontType0C"):
            cff_programs.append((source, data))
    built = Counter()
    for index, (source, data) in enumerate(cff_programs):
        for cff2 in (False, True):
            named = index % 2 == 0
            try:
                font = opentype_from_cff(data, cff2=cff2, named=named)
            except Exception:  # noqa: BLE001
                built["not built"] += 1
                continue
            kind = "built CFF2" if cff2 else "built OpenType CFF"
            record(kind, source, check_opentype(font))
    record("variable CFF2", "built", check_opentype(variable_cff2()))
    cubic = cubic_glyf()
    record("cubic glyf", "built", check_truetype(cubic) + check_program_contours(cubic))
    print("; ".join(f"{count} {kind}" for kind, count in results.items()), end="")
    print(f"; {built['not built']} CFF programs fontTools could not wrap" if built else "")
    if failures:
        print(f"{len(failures)} fonts differ:")
        print("\n".join(failures[:40]))
        return 1
    print("every font reads the same as with fontTools")
    return 0


if __name__ == "__main__":
    sys.exit(main())
