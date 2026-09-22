from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, cast

from core_pdf._vendor.fontTools.agl import LEGACY_AGL2UV, toUnicode
from core_pdf.impl._impl.fonts.cmap_resources import resolve_cmap_decoder
from core_pdf.impl._impl.fonts.data.metrics import FONT_DATA
from core_pdf.impl._impl.model.geometry import bbox_union
from core_pdf.impl._impl.pdf_names import recover_pdf_name
from core_pdf.impl.exceptions import PdfError
from core_pdf_spec.s_09_fonts.data.base_encodings import (
    MAC_ROMAN_ENCODING,
    STANDARD_ENCODING,
    WIN_ANSI_ENCODING,
)

from .._shared import LIGATURES


def internal_glyph_name_text(name: str) -> str:
    codepoints = LEGACY_AGL2UV.get(name)
    if codepoints is not None:
        return chr(codepoints[0]) if len(codepoints) == 1 else "".join(map(chr, codepoints))
    return toUnicode(name)


def _mapping_value(mapping: object, name: str) -> object | None:
    if not isinstance(mapping, dict):
        return None
    return next((value for key, value in mapping.items() if str(key) == name), None)


@dataclass(slots=True)
class _FontProjection:
    font: dict[Any, Any]
    values: dict[str, Any]
    has_widths: bool
    legacy_widths: list[float] | None
    first_char: int
    recovered_malformed_token: bool


internal_FONT_PROJECTION_CACHE_MAX_ENTRIES = 128

internal_FONT_PROJECTION_CACHE: OrderedDict[int, _FontProjection] = OrderedDict()


def _font_projection(font: dict[Any, Any]) -> _FontProjection:
    cache_key = id(font)
    entry = internal_FONT_PROJECTION_CACHE.get(cache_key)
    if entry is not None and entry.font is font:
        internal_FONT_PROJECTION_CACHE.move_to_end(cache_key)
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
    internal_FONT_PROJECTION_CACHE[cache_key] = entry
    while len(internal_FONT_PROJECTION_CACHE) > internal_FONT_PROJECTION_CACHE_MAX_ENTRIES:
        internal_FONT_PROJECTION_CACHE.popitem(last=False)
    return entry


def _font_value(font: object, name: str) -> object | None:
    if not isinstance(font, dict):
        return None
    return _font_projection(font).values.get(name)


def _pdfminer_base_encoding_text(
    decoder: Any,
    char_code: int,
    base_encoding: str | None = None,
) -> str:
    if base_encoding is None:
        base_encoding = getattr(decoder, "base_encoding", None)
    table = {
        "MacRomanEncoding": MAC_ROMAN_ENCODING,
        "WinAnsiEncoding": WIN_ANSI_ENCODING,
    }.get(base_encoding if isinstance(base_encoding, str) else "", STANDARD_ENCODING)
    if base_encoding == "WinAnsiEncoding":
        if char_code == 173:
            return " "
        if char_code in {127, 129, 141, 143, 144, 157}:
            return ""
    if base_encoding == "MacRomanEncoding" and char_code in {
        173,
        176,
        178,
        179,
        182,
        183,
        184,
        185,
        186,
        189,
        195,
        197,
        198,
        215,
    }:
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


def internal_pdfminer_glyph_text(glyph: Any) -> str:
    if glyph.unicode_source == "actual_text" and glyph.text == "\ufeff":
        return ""
    if glyph.unicode_source == "actual_text" and glyph.alternates:
        return glyph.alternates[0]
    to_unicode = getattr(glyph.font_decoder, "to_unicode", None)
    decoder = glyph.font_decoder
    glyph_name = getattr(decoder, "encoding_differences", {}).get(glyph.char_code)
    if glyph_name and glyph_name.isdecimal():
        glyph_name = None
    glyph_name_text = internal_glyph_name_text(glyph_name) if glyph_name else ""
    if to_unicode is not None and glyph.code_bytes:
        mapped = _pdfminer_to_unicode_text(glyph, to_unicode)
        if mapped is not None and len(mapped) <= 1:
            return mapped or f"(cid:{glyph.cid})"
        if glyph_name_text and (
            glyph.unicode_source == "encoding" or not getattr(decoder, "is_cid_font", False)
        ):
            return glyph_name_text
        if (
            glyph.unicode_source != "identity"
            and not getattr(decoder, "is_cid_font", False)
            and glyph.char_code is not None
            and (not glyph_name or getattr(decoder, "base_encoding", None) == "WinAnsiEncoding")
        ):
            base_encoding = getattr(decoder, "base_encoding", None)
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
        base_encoding = getattr(decoder, "base_encoding", None)
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
    if glyph.unicode_source == "truetype_cmap" and getattr(decoder, "to_unicode", None) is None:
        descendants = _font_value(getattr(decoder, "font", None), "DescendantFonts")
        descendant = descendants[0] if isinstance(descendants, list) and descendants else None
        system_info = _mapping_value(descendant, "CIDSystemInfo")
        registry = _mapping_value(system_info, "Registry")
        registry_data = getattr(registry, "data", b"")
        if isinstance(registry_data, bytes) and registry_data.strip() == b"PDFAUTOCAD":
            return f"(cid:{glyph.cid})"
    return glyph.text


def internal_pdfminer_embedded_cmap_is_unusable(glyph: Any) -> bool:
    decoder = glyph.font_decoder
    if not getattr(decoder, "is_cid_font", False):
        return False
    descendants = _font_value(decoder.font, "DescendantFonts")
    if not isinstance(descendants, list) or not descendants:
        raise PdfError("Type0 font is missing /DescendantFonts")
    encoding = _font_value(decoder.font, "Encoding")
    try:
        data = bytes(getattr(encoding, "decoded_data"))
    except AttributeError, TypeError, ValueError:
        return False
    cmap_name_match = re.search(rb"/CMapName\s*/([^\s<>\[\]()/%]+)", data)
    if cmap_name_match is not None:
        cmap_name = cmap_name_match.group(1).decode("latin-1")
        if resolve_cmap_decoder(cmap_name) is not None:
            return False
    return re.search(rb"/[!-~]+\s+usecmap\b", data) is None


def internal_pdfminer_ligature_overrides(
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
    for index, glyph in enumerate(glyphs):
        decoder = glyph.font_decoder
        to_unicode = getattr(decoder, "to_unicode", None)
        if to_unicode is not None and glyph.code_bytes:
            mapped = _pdfminer_to_unicode_text(glyph, to_unicode)
            if mapped is not None:
                cluster_id = glyph.cluster_key
                cluster = [glyph]
                if cluster_id is not None:
                    next_index = index + 1
                    while next_index < len(glyphs):
                        item = glyphs[next_index]
                        if item.cluster_key != cluster_id:
                            break
                        cluster.append(item)
                        next_index += 1
                box = bbox_union(item.advance_bbox for item in cluster) or glyph.advance_bbox
                overrides[id(glyph)] = (mapped, box, glyph.baseline)
                skipped.update(id(item) for item in cluster[1:])
                continue
        if glyph.char_code is None:
            continue
        glyph_name = getattr(decoder, "encoding_differences", {}).get(glyph.char_code)
        glyph_name_text = internal_glyph_name_text(glyph_name) if glyph_name else ""
        if len(glyph_name_text) > 1:
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
        base_encoding = getattr(decoder, "base_encoding", None)
        base_table = {
            "MacRomanEncoding": MAC_ROMAN_ENCODING,
            "WinAnsiEncoding": WIN_ANSI_ENCODING,
        }.get(base_encoding if isinstance(base_encoding, str) else "", STANDARD_ENCODING)
        encoded_ligature = (
            base_table[glyph.char_code] if 0 <= glyph.char_code < len(base_table) else ""
        )
        expected_ligature = LIGATURES.get(str(glyph_name) if glyph_name else "", encoded_ligature)
        if expected_ligature not in LIGATURES.values():
            continue
        ligature_cluster: tuple[Any, ...] | None = None
        legacy_text: str | None = None
        for cluster_size in (3, 2):
            candidate = glyphs[index : index + cluster_size]
            if len(candidate) != cluster_size:
                continue
            decomposition = "".join(item.text for item in candidate)
            candidate_text = LIGATURES.get(decomposition)
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


def _pdfminer_builtin_width(glyph: Any) -> float | None:
    decoder = glyph.font_decoder
    projected_text = internal_pdfminer_glyph_text(glyph)
    font = decoder.font
    projection = _font_projection(font) if isinstance(font, dict) else None
    values = projection.values if projection is not None else {}
    legacy_widths = projection.legacy_widths if projection is not None else None
    width_index = -1
    if projection is not None and legacy_widths is not None:
        width_index = glyph.char_code - projection.first_char if glyph.char_code is not None else -1
        if projection.recovered_malformed_token and 0 <= width_index < len(legacy_widths):
            return legacy_widths[width_index]
    glyph_name = getattr(decoder, "encoding_differences", {}).get(glyph.char_code)
    if (
        not decoder.is_cid_font
        and len(projected_text) == 1
        and ord(projected_text) < 32
        and glyph_name is not None
        and internal_glyph_name_text(glyph_name).isspace()
        and internal_glyph_name_text(glyph_name) != projected_text
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


def internal_pdfminer_normalized_width(glyph: Any) -> float:
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
            width_scale = float(cast(Any, font_matrix[0])) + float(cast(Any, font_matrix[2]))
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
    builtin_width = _pdfminer_builtin_width(glyph)
    if builtin_width is not None:
        width = builtin_width * 0.001
    base_font = recover_pdf_name(_font_value(glyph.font_decoder.font, "BaseFont"))
    glyph_name = getattr(glyph.font_decoder, "encoding_differences", {}).get(glyph.char_code)
    if (
        base_font in {"Symbol", "ZapfDingbats"}
        and glyph_name
        and not internal_glyph_name_text(glyph_name)
    ):
        return 0.0
    return width


def internal_pdfminer_font_name(glyph: Any) -> str:
    base_font = str(_font_value(glyph.font_decoder.font, "BaseFont") or "")
    builtin_metrics = FONT_DATA.get(base_font)
    if isinstance(builtin_metrics, dict):
        props = builtin_metrics.get("props")
        if isinstance(props, dict):
            font_name = props.get("FontName")
            if isinstance(font_name, str):
                return font_name
    return str(glyph.font_name)


def internal_pdfminer_descent(glyph: Any) -> float:
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
            descent_scale = float(cast(Any, font_matrix[1])) + float(cast(Any, font_matrix[3]))
        descriptor = _font_value(decoder.font, "FontDescriptor")
        descriptor_bbox = _mapping_value(descriptor, "FontBBox")
        if descriptor is not None:
            descent_value = (
                float(cast(Any, descriptor_bbox[1]))
                if isinstance(descriptor_bbox, (tuple, list)) and len(descriptor_bbox) == 4
                else 0.0
            )
        else:
            font_bbox = _font_value(decoder.font, "FontBBox")
            if isinstance(font_bbox, (tuple, list)) and len(font_bbox) == 4:
                descent_value = float(cast(Any, font_bbox[1]))
    return descent_value * descent_scale
