from __future__ import annotations

import re
from collections import OrderedDict
from functools import lru_cache
from typing import Any

from core_pdf._vendor.fontTools.agl import LEGACY_AGL2UV, toUnicode
from core_pdf.impl.exceptions import PdfError
from core_pdf.impl.fonts_cmap import resolve_cmap_decoder
from core_pdf.impl.fonts_decoder import FontDecoder
from core_pdf.impl.fonts_metrics import FONT_DATA, LIGATURE_TEXT_TO_CHAR
from core_pdf.impl.geometry import bbox_union
from core_pdf.impl.pdf_values import recover_pdf_name
from core_pdf.impl.types import PdfName, PdfString, RecordType, ReplaceFields, ReprFields
from core_pdf_spec.s_09_fonts.data.base_encodings import (
    MAC_ROMAN_ENCODING,
    STANDARD_ENCODING,
    WIN_ANSI_ENCODING,
)

from ._zapf_dingbats import ZAPF_DINGBATS_WIDTHS


def decoder_base_encoding(decoder: object) -> str | None:
    return decoder.encoding.base_encoding if isinstance(decoder, FontDecoder) else None


def decoder_encoding_differences(decoder: object) -> dict[int, str]:
    return decoder.encoding.encoding_differences if isinstance(decoder, FontDecoder) else {}


def pdfminer_encoding_differences(decoder: object) -> dict[int, str]:
    """The glyph names pdfminer's encoding gives codes, over its base encoding.

    pdfminer reads the /Differences of an /Encoding dictionary, and only without /Encoding
    the encoding a Type 1 font program (/FontFile) declares; the engine also merges in the
    encodings of other font programs, which pdfminer never reads.
    """
    if not isinstance(decoder, FontDecoder):
        return {}
    known = PDFMINER_DIFFERENCES.get(id(decoder))
    if known is not None and known[0] is decoder:
        return known[1]
    differences = decoder.encoding.encoding_differences
    font = decoder.font
    encoding = _font_value(font, "Encoding")
    if isinstance(encoding, dict):
        differences = {}
        code: int | None = None
        array = _mapping_value(encoding, "Differences")
        for item in array if isinstance(array, (list, tuple)) else ():
            if isinstance(item, bool):
                continue
            if isinstance(item, (int, float)):
                code = int(item)
            elif isinstance(item, PdfName) and code is not None:
                differences[code] = item.value
                code += 1
    elif (
        encoding is not None
        or _mapping_value(_font_value(font, "FontDescriptor"), "FontFile") is None
    ):
        if not getattr(decoder, "is_type3", False) and not getattr(decoder, "is_cid_font", False):
            differences = {}
    PDFMINER_DIFFERENCES[id(decoder)] = (decoder, differences)
    if len(PDFMINER_DIFFERENCES) > 256:
        PDFMINER_DIFFERENCES.pop(next(iter(PDFMINER_DIFFERENCES)))
    return differences


PDFMINER_DIFFERENCES: dict[int, tuple[object, dict[int, str]]] = {}


def decoder_to_unicode(decoder: object) -> Any:
    return decoder.unicode.to_unicode if isinstance(decoder, FontDecoder) else None


@lru_cache(maxsize=4096)
def legacy_glyph_name_text(name: str) -> str:
    codepoints = LEGACY_AGL2UV.get(name)
    if codepoints is not None:
        return chr(codepoints[0]) if len(codepoints) == 1 else "".join(map(chr, codepoints))
    return toUnicode(name)


def _mapping_value(mapping: object, name: str) -> object | None:
    if not isinstance(mapping, dict):
        return None
    return next((value for key, value in mapping.items() if str(key) == name), None)


class _FontProjection(ReprFields, ReplaceFields, metaclass=RecordType, frozen=False):
    font: dict[Any, Any]
    values: dict[str, Any]
    has_widths: bool
    legacy_widths: list[float] | None
    first_char: int
    recovered_malformed_token: bool

    __hash__ = None  # type: ignore[assignment]


FONT_PROJECTION_CACHE_MAX_ENTRIES = 128

FONT_PROJECTION_CACHE: OrderedDict[int, _FontProjection] = OrderedDict()


LAST_FONT_PROJECTION: list[_FontProjection | None] = [None]


def _font_projection(font: dict[Any, Any]) -> _FontProjection:
    last = LAST_FONT_PROJECTION[0]
    if last is not None and last.font is font:
        return last
    cache_key = id(font)
    entry = FONT_PROJECTION_CACHE.get(cache_key)
    if entry is not None and entry.font is font:
        FONT_PROJECTION_CACHE.move_to_end(cache_key)
        LAST_FONT_PROJECTION[0] = entry
        return entry
    values: dict[str, Any] = {}
    for raw_key, value in font.items():
        values.setdefault(str(raw_key), value)
    font_widths = values.get("Widths")
    first_char = values.get("FirstChar")
    legacy_widths: list[float] | None = None
    recovered_malformed_token = False
    if isinstance(font_widths, (list, tuple)) and isinstance(first_char, int):
        legacy_widths = []
        for width_value in font_widths:
            if isinstance(width_value, str):
                match = re.fullmatch(r"[^0-9+.-]+([+-]?(?:\d+(?:\.\d*)?|\.\d+))", width_value)
                if match is not None:
                    legacy_widths.extend((0.0, float(match.group(1))))
                    recovered_malformed_token = True
                    continue
            try:
                legacy_widths.append(float(width_value))
            except TypeError, ValueError:
                legacy_widths.append(0.0)
    entry = _FontProjection(
        font,
        values,
        font_widths is not None,
        legacy_widths,
        first_char if isinstance(first_char, int) else 0,
        recovered_malformed_token,
    )
    FONT_PROJECTION_CACHE[cache_key] = entry
    while len(FONT_PROJECTION_CACHE) > FONT_PROJECTION_CACHE_MAX_ENTRIES:
        FONT_PROJECTION_CACHE.popitem(last=False)
    LAST_FONT_PROJECTION[0] = entry
    return entry


def _font_value(font: object, name: str) -> object | None:
    if not isinstance(font, dict):
        return None
    return _font_projection(font).values.get(name)


BASE_ENCODING_TABLES = {
    "MacRomanEncoding": MAC_ROMAN_ENCODING,
    "WinAnsiEncoding": WIN_ANSI_ENCODING,
}
WIN_ANSI_BLANK_CODES = frozenset({127, 129, 141, 143, 144, 157})
MAC_ROMAN_BLANK_CODES = frozenset(
    {173, 176, 178, 179, 182, 183, 184, 185, 186, 189, 195, 197, 198, 215}
)


def base_encoding_table(base_encoding: object) -> tuple[str, ...]:
    return BASE_ENCODING_TABLES.get(
        base_encoding if isinstance(base_encoding, str) else "", STANDARD_ENCODING
    )


def _pdfminer_base_encoding_text(
    decoder: Any,
    char_code: int,
    base_encoding: str | None = None,
) -> str:
    if base_encoding is None:
        base_encoding = decoder_base_encoding(decoder)
    table = base_encoding_table(base_encoding)
    if base_encoding == "WinAnsiEncoding":
        if char_code == 173:
            return " "
        if char_code in WIN_ANSI_BLANK_CODES:
            return ""
    if base_encoding == "MacRomanEncoding" and char_code in MAC_ROMAN_BLANK_CODES:
        return ""
    return table[char_code]


def _pdfminer_to_unicode_text(glyph: Any, to_unicode: Any) -> str | None:
    mappings = getattr(to_unicode, "mappings", {})
    decoder = glyph.font_decoder
    if not getattr(decoder, "is_cid_font", False):
        return mappings.get(glyph.code_bytes)
    cid = glyph.cid
    if cid is None:
        return None
    for length in getattr(to_unicode, "decode_lengths", (1,)):
        if cid >= 1 << (length * 8):
            continue
        mapped = mappings.get(cid.to_bytes(length, "big"))
        if mapped is not None:
            return mapped
    return None


# pdfminer makes a character of every code, even one that maps to no text; the engine records
# a glyph only for text, so an empty mapping stands in as this until the projection.
EMPTY_GLYPH_TEXT = "\x00"


def pdfminer_glyph_text(glyph: Any) -> str:
    text = _pdfminer_glyph_text(glyph)
    return "" if text == EMPTY_GLYPH_TEXT else text


def _pdfminer_glyph_text(glyph: Any) -> str:
    if glyph.unicode_source == "actual_text" and glyph.text == "\ufeff":
        return ""
    if glyph.unicode_source == "actual_text" and glyph.alternates:
        return glyph.alternates[0]
    to_unicode = decoder_to_unicode(glyph.font_decoder)
    decoder = glyph.font_decoder
    glyph_name = pdfminer_encoding_differences(decoder).get(glyph.char_code)
    if glyph_name and glyph_name.isdecimal():
        glyph_name = None
    glyph_name_text = legacy_glyph_name_text(glyph_name) if glyph_name else ""
    if to_unicode is not None and glyph.code_bytes:
        mapped = _pdfminer_to_unicode_text(glyph, to_unicode)
        if mapped is not None and len(mapped) <= 1:
            return mapped
        if glyph_name_text and (
            glyph.unicode_source == "encoding" or not getattr(decoder, "is_cid_font", False)
        ):
            return glyph_name_text
        if (
            glyph.unicode_source != "identity"
            and not getattr(decoder, "is_cid_font", False)
            and glyph.char_code is not None
            and (not glyph_name or decoder_base_encoding(decoder) == "WinAnsiEncoding")
        ):
            base_encoding = decoder_base_encoding(decoder)
            if base_encoding is None and str(_font_value(decoder.font, "Subtype")) == "TrueType":
                base_encoding = "WinAnsiEncoding"
            encoded = _pdfminer_base_encoding_text(decoder, glyph.char_code, base_encoding)
            if encoded:
                return encoded
        if getattr(decoder, "is_cid_font", False):
            return f"(cid:{glyph.cid})"
        if glyph_name and not glyph_name_text and not getattr(decoder, "is_type3", False):
            return f"(cid:{glyph.cid})"
        if glyph.unicode_source == "identity":
            return f"(cid:{glyph.cid})"
    elif glyph_name_text and (
        glyph.unicode_source == "encoding" or not getattr(decoder, "is_cid_font", False)
    ):
        return glyph_name_text
    if (
        glyph_name
        and getattr(decoder, "is_type3", False)
        and glyph.char_code is not None
        and (base_text := _pdfminer_base_encoding_text(decoder, glyph.char_code))
    ):
        return base_text
    if glyph_name:
        return f"(cid:{glyph.cid})"
    if glyph.cid is None:
        return glyph.text
    if glyph.unicode_source == "identity" and (
        to_unicode is None or getattr(decoder, "is_cid_font", False)
    ):
        return f"(cid:{glyph.cid})"
    if glyph.unicode_source in {"fallback_nul", "undefined"}:
        base_encoding = decoder_base_encoding(decoder)
        if base_encoding in {"MacRomanEncoding", "StandardEncoding", "WinAnsiEncoding"} and (
            glyph.char_code is not None
        ):
            base_text = _pdfminer_base_encoding_text(decoder, glyph.char_code)
            if base_text:
                return base_text
        return f"(cid:{glyph.cid})"
    if glyph.unicode_source == "encoding" and not glyph_name and glyph.char_code is not None:
        base_text = _pdfminer_base_encoding_text(decoder, glyph.char_code)
        return base_text or f"(cid:{glyph.cid})"
    if glyph.unicode_source == "truetype_cmap" and decoder_to_unicode(decoder) is None:
        descendants = _font_value(getattr(decoder, "font", None), "DescendantFonts")
        descendant = descendants[0] if isinstance(descendants, list) and descendants else None
        system_info = _mapping_value(descendant, "CIDSystemInfo")
        registry = _mapping_value(system_info, "Registry")
        registry_data = getattr(registry, "data", b"")
        if isinstance(registry_data, bytes) and registry_data.strip() == b"PDFAUTOCAD":
            return f"(cid:{glyph.cid})"
    return glyph.text


EMBEDDED_CMAP_UNUSABLE: dict[int, tuple[object, bool]] = {}
EMBEDDED_CMAP_UNUSABLE_LIMIT = 256
PREDEFINED_CMAP_RESOLVES: dict[str, bool] = {}


def predefined_cmap_resolves(name: str) -> bool:
    resolves = PREDEFINED_CMAP_RESOLVES.get(name)
    if resolves is None:
        resolves = PREDEFINED_CMAP_RESOLVES[name] = resolve_cmap_decoder(name) is not None
    return resolves


def pdfminer_embedded_cmap_is_unusable(glyph: Any) -> bool:
    decoder = glyph.font_decoder
    if not getattr(decoder, "is_cid_font", False):
        return False
    known = EMBEDDED_CMAP_UNUSABLE.get(id(decoder))
    if known is not None and known[0] is decoder:
        return known[1]
    unusable = embedded_cmap_is_unusable(decoder)
    if len(EMBEDDED_CMAP_UNUSABLE) >= EMBEDDED_CMAP_UNUSABLE_LIMIT:
        EMBEDDED_CMAP_UNUSABLE.clear()
    EMBEDDED_CMAP_UNUSABLE[id(decoder)] = (decoder, unusable)
    return unusable


def embedded_cmap_is_unusable(decoder: Any) -> bool:
    descendants = _font_value(decoder.font, "DescendantFonts")
    if not isinstance(descendants, list) or not descendants:
        raise PdfError("Type0 font is missing /DescendantFonts")
    # pdfminer's PDFCIDFont._get_cmap_name never reads an embedded CMap's data: it takes
    # the /CMapName entry of the stream's dictionary, or "unknown" without one, and looks
    # that name up among the predefined CMaps. A name it does not find gives an empty CMap,
    # which decodes no character.
    encoding = _font_value(decoder.font, "Encoding")
    dictionary = getattr(encoding, "dictionary", None)
    if not isinstance(dictionary, dict):
        return False
    cmap_name = dictionary.get("CMapName")
    if cmap_name is None:
        name = "unknown"
    elif isinstance(cmap_name, PdfName):
        name = recover_pdf_name(cmap_name) or ""
    else:
        return True
    name = {"DLIdent-H": "Identity-H", "DLIdent-V": "Identity-V"}.get(name, name)
    return not predefined_cmap_resolves(name)


def pdfminer_ligature_overrides(
    glyphs: tuple[Any, ...],
) -> tuple[
    dict[
        int,
        tuple[
            str,
            tuple[float, float, float, float],
            tuple[float, float, float, float] | None,
        ],
    ],
    set[int],
]:
    overrides: dict[
        int,
        tuple[
            str,
            tuple[float, float, float, float],
            tuple[float, float, float, float] | None,
        ],
    ] = {}
    skipped: set[int] = set()
    # What a glyph can start depends only on its decoder and codes, which a page repeats.
    # The glyphs keep their decoders alive for the call, so a decoder's id is a safe key.
    plans: dict[tuple[Any, ...], tuple[str, str] | None] = {}
    for index, glyph in enumerate(glyphs):
        plan_key = (id(glyph.font_decoder), glyph.code_bytes, glyph.cid, glyph.char_code)
        if plan_key in plans:
            plan = plans[plan_key]
        else:
            plan = plans[plan_key] = _ligature_plan(glyph)
        if plan is None:
            continue
        kind, planned_text = plan
        if kind == "mapped":
            cluster_id = glyph.cluster_key
            if (
                cluster_id is not None
                and index + 1 < len(glyphs)
                and glyphs[index + 1].cluster_key == cluster_id
            ):
                cluster = [glyph, glyphs[index + 1]]
                next_index = index + 2
                while next_index < len(glyphs):
                    item = glyphs[next_index]
                    if item.cluster_key != cluster_id:
                        break
                    cluster.append(item)
                    next_index += 1
                box = bbox_union(item.advance_bbox for item in cluster) or glyph.advance_bbox
                skipped.update(id(item) for item in cluster[1:])
            else:
                box = glyph.advance_bbox
            overrides[id(glyph)] = (planned_text, box, glyph.baseline)
            continue
        if kind == "difference":
            glyph_name_text = planned_text
            difference_cluster = glyphs[index : index + len(glyph_name_text)]
            if len(difference_cluster) == len(glyph_name_text) and all(
                item.seqno == glyph.seqno
                and item.code_bytes == glyph.code_bytes
                and item.char_code == glyph.char_code
                for item in difference_cluster
            ):
                box = (
                    bbox_union(item.advance_bbox for item in difference_cluster)
                    or glyph.advance_bbox
                )
                overrides[id(glyph)] = (glyph_name_text, box, glyph.baseline)
                skipped.update(id(item) for item in difference_cluster[1:])
                continue
            # Not a run of the name's characters: what its encoding names may still be one.
            difference_ligature = _expected_ligature(glyph)
            if difference_ligature is None:
                continue
            expected_ligature = difference_ligature
        else:
            expected_ligature = planned_text
        ligature_cluster: tuple[Any, ...] | None = None
        legacy_text: str | None = None
        for cluster_size in (3, 2):
            candidate = glyphs[index : index + cluster_size]
            if len(candidate) != cluster_size:
                continue
            decomposition = "".join(item.text for item in candidate)
            candidate_text = LIGATURE_TEXT_TO_CHAR.get(decomposition)
            if candidate_text is None or candidate_text != expected_ligature:
                continue
            if any(
                item.seqno != glyph.seqno
                or item.code_bytes != glyph.code_bytes
                or item.char_code != glyph.char_code
                for item in candidate[1:]
            ):
                continue
            ligature_cluster = candidate
            legacy_text = candidate_text
            break
        if ligature_cluster is None or legacy_text is None:
            continue
        box = bbox_union(item.advance_bbox for item in ligature_cluster) or glyph.advance_bbox
        first_baseline = ligature_cluster[0].baseline
        last_baseline = ligature_cluster[-1].baseline
        baseline = (
            (
                first_baseline[0],
                first_baseline[1],
                last_baseline[2],
                last_baseline[3],
            )
            if first_baseline is not None and last_baseline is not None
            else None
        )
        overrides[id(glyph)] = (legacy_text, box, baseline)
        skipped.update(id(item) for item in ligature_cluster[1:])
    return overrides, skipped


def _ligature_plan(glyph: Any) -> tuple[str, str] | None:
    """What a glyph with these codes can start, before its neighbours are looked at.

    "mapped" and its /ToUnicode text; "difference" and the several characters its
    /Differences name stands for; "ligature" and the ligature its encoding names; or None.
    """
    to_unicode = decoder_to_unicode(glyph.font_decoder)
    if to_unicode is not None and glyph.code_bytes:
        mapped = _pdfminer_to_unicode_text(glyph, to_unicode)
        if mapped is not None:
            return ("mapped", mapped)
    if glyph.char_code is None:
        return None
    glyph_name = decoder_encoding_differences(glyph.font_decoder).get(glyph.char_code)
    glyph_name_text = legacy_glyph_name_text(glyph_name) if glyph_name else ""
    if len(glyph_name_text) > 1:
        return ("difference", glyph_name_text)
    expected = _expected_ligature(glyph)
    return None if expected is None else ("ligature", expected)


LIGATURE_CHARS = frozenset(LIGATURE_TEXT_TO_CHAR.values())


def _expected_ligature(glyph: Any) -> str | None:
    """The ligature character the glyph's encoding names for its code, or None if none does.

    The /Differences name comes first, then the base encoding's entry, as in pdfminer.
    """
    decoder = glyph.font_decoder
    glyph_name = decoder_encoding_differences(decoder).get(glyph.char_code)
    base_table = base_encoding_table(decoder_base_encoding(decoder))
    encoded_ligature = base_table[glyph.char_code] if 0 <= glyph.char_code < len(base_table) else ""
    expected_ligature = (
        LIGATURE_TEXT_TO_CHAR.get(str(glyph_name), encoded_ligature)
        if glyph_name
        else encoded_ligature
    )
    return expected_ligature if expected_ligature and expected_ligature in LIGATURE_CHARS else None


def _pdfminer_builtin_width(glyph: Any, projected_text: str | None = None) -> float | None:
    decoder = glyph.font_decoder
    if projected_text is None:
        projected_text = pdfminer_glyph_text(glyph)
    font = decoder.font
    projection = _font_projection(font) if isinstance(font, dict) else None
    values = projection.values if projection is not None else {}
    legacy_widths = projection.legacy_widths if projection is not None else None
    width_index = -1
    if projection is not None and legacy_widths is not None:
        width_index = glyph.char_code - projection.first_char if glyph.char_code is not None else -1
        if projection.recovered_malformed_token and 0 <= width_index < len(legacy_widths):
            return legacy_widths[width_index]
    glyph_name = decoder_encoding_differences(decoder).get(glyph.char_code)
    if (
        not decoder.is_cid_font
        and len(projected_text) == 1
        and ord(projected_text) < 32
        and glyph_name is not None
        and legacy_glyph_name_text(glyph_name).isspace()
        and legacy_glyph_name_text(glyph_name) != projected_text
    ):
        return 0.0
    if decoder.is_cid_font or decoder.is_type3:
        return None
    base_font = str(values.get("BaseFont") or "")
    entry = FONT_DATA.get(base_font)
    if isinstance(entry, dict):
        widths = entry.get("widths")
        if not isinstance(widths, dict):
            return None
        if base_font == "ZapfDingbats":
            # pdfminer's table keys these glyphs apart from their text; see _zapf_dingbats.
            width = (
                ZAPF_DINGBATS_WIDTHS.get(ord(projected_text)) if len(projected_text) == 1 else None
            )
        else:
            width = widths.get(projected_text)
        return 0.0 if width is None else float(width)
    if legacy_widths is not None:
        if 0 <= width_index < len(legacy_widths):
            return legacy_widths[width_index]
        descriptor = values.get("FontDescriptor")
        missing_width = _mapping_value(descriptor, "MissingWidth")
        return float(missing_width) if isinstance(missing_width, (int, float)) else 0.0
    if projection is not None and projection.has_widths:
        return None
    descriptor = values.get("FontDescriptor")
    missing_width = _mapping_value(descriptor, "MissingWidth")
    return float(missing_width) if isinstance(missing_width, (int, float)) else 0.0


def pdfminer_normalized_width(glyph: Any, projected_text: str | None = None) -> float:
    width_code = (
        glyph.cid
        if getattr(glyph.font_decoder, "is_cid_font", False)
        else glyph.char_code
        if glyph.char_code is not None
        else glyph.cid
    )
    width_lookup = getattr(glyph.font_decoder, "glyph_width", None)
    if width_code is None or not callable(width_lookup):
        return 0.0
    width_scale = 0.001
    if getattr(glyph.font_decoder, "is_type3", False):
        font_matrix = _font_value(glyph.font_decoder.font, "FontMatrix")
        if isinstance(font_matrix, (tuple, list)) and len(font_matrix) >= 4:
            width_scale = float(font_matrix[0]) + float(font_matrix[2])
        raw_widths = _font_value(glyph.font_decoder.font, "Widths")
        first_char = _font_value(glyph.font_decoder.font, "FirstChar")
        if isinstance(raw_widths, (tuple, list)) and isinstance(first_char, int):
            index = width_code - first_char
            if 0 <= index < len(raw_widths):
                try:
                    return float(raw_widths[index]) * width_scale
                except TypeError, ValueError:
                    return 0.0
            return 0.0
    width = float(width_lookup(width_code)) * width_scale
    builtin_width = _pdfminer_builtin_width(glyph, projected_text)
    if builtin_width is not None:
        width = builtin_width * 0.001
    base_font = recover_pdf_name(_font_value(glyph.font_decoder.font, "BaseFont"))
    glyph_name = decoder_encoding_differences(glyph.font_decoder).get(glyph.char_code)
    if (
        base_font in {"Symbol", "ZapfDingbats"}
        and glyph_name
        and not legacy_glyph_name_text(glyph_name)
    ):
        return 0.0
    return width


def pdfminer_literal_name(value: object) -> Any:
    """A name as pdfminer's literal_name gives it: UTF-8 text, or the bytes' repr."""
    if not isinstance(value, PdfName):
        return value
    try:
        return value.value_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return str(value.value_bytes)


def pdfminer_font_descriptor(font: object) -> object:
    """The descriptor pdfminer reads a font's name and metrics from.

    Follows pdfminer.six's PDFResourceManager.get_font and its font classes, Copyright (c)
    2004-2016 Yusuke Shinyama, under the MIT License.
    """
    subtype = _font_value(font, "Subtype")
    subtype_name = "Type1" if subtype is None else pdfminer_literal_name(subtype)
    if subtype_name == "Type0":
        descendants = _font_value(font, "DescendantFonts")
        if isinstance(descendants, (list, tuple)) and descendants:
            return _font_value(descendants[0], "FontDescriptor") or {}
        return {}
    if subtype_name in {"CIDFontType0", "CIDFontType2", "Type3"}:
        return _font_value(font, "FontDescriptor") or {}
    base_font = _font_value(font, "BaseFont")
    builtin = FONT_DATA.get(pdfminer_literal_name(base_font)) if base_font is not None else None
    if isinstance(builtin, dict) and isinstance(builtin.get("props"), dict):
        return builtin["props"]
    return _font_value(font, "FontDescriptor") or {}


def pdfminer_font_name(glyph: Any) -> Any:
    decoder = glyph.font_decoder
    known = PDFMINER_FONT_NAMES.get(id(decoder))
    if known is not None and known[0] is decoder:
        return known[1]
    descriptor = pdfminer_font_descriptor(decoder.font)
    name = _mapping_value(descriptor, "FontName") if isinstance(descriptor, dict) else None
    if name is None and isinstance(descriptor, dict) and "FontName" in descriptor:
        name = descriptor["FontName"]
    name = "unknown" if name is None else pdfminer_literal_name(name)
    if isinstance(name, PdfString):
        name = name.data
    PDFMINER_FONT_NAMES[id(decoder)] = (decoder, name)
    if len(PDFMINER_FONT_NAMES) > FONT_PROJECTION_CACHE_MAX_ENTRIES:
        PDFMINER_FONT_NAMES.pop(next(iter(PDFMINER_FONT_NAMES)))
    return name


PDFMINER_FONT_NAMES: dict[int, tuple[Any, Any]] = {}


def pdfminer_descent(glyph: Any) -> float:
    decoder = glyph.font_decoder
    descent_scale = 0.001
    descent_value = float(getattr(decoder, "descent", -200.0))
    base_font = str(_font_value(decoder.font, "BaseFont") or "")
    if (
        not getattr(decoder, "is_cid_font", False)
        and not getattr(decoder, "is_type3", False)
        and _font_value(decoder.font, "FontDescriptor") is None
    ):
        descent_value = 0.0
    builtin_metrics = FONT_DATA.get(base_font)
    if (
        not getattr(decoder, "is_cid_font", False)
        and not getattr(decoder, "is_type3", False)
        and isinstance(builtin_metrics, dict)
        and isinstance(builtin_metrics.get("props"), dict)
    ):
        builtin_descent = builtin_metrics["props"].get("Descent")
        descent_value = float(builtin_descent) if isinstance(builtin_descent, (int, float)) else 0.0
    if getattr(decoder, "is_type3", False):
        font_matrix = _font_value(decoder.font, "FontMatrix")
        if isinstance(font_matrix, (tuple, list)) and len(font_matrix) == 6:
            descent_scale = float(font_matrix[1]) + float(font_matrix[3])
        descriptor = _font_value(decoder.font, "FontDescriptor")
        descriptor_bbox = _mapping_value(descriptor, "FontBBox")
        if descriptor is not None:
            descent_value = (
                float(descriptor_bbox[1])
                if isinstance(descriptor_bbox, (tuple, list)) and len(descriptor_bbox) == 4
                else 0.0
            )
        else:
            font_bbox = _font_value(decoder.font, "FontBBox")
            if isinstance(font_bbox, (tuple, list)) and len(font_bbox) == 4:
                descent_value = float(font_bbox[1])
    return descent_value * descent_scale
