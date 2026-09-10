"""Reader Unicode and marked-content replacement over captured glyphs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import replace
from itertools import groupby

from core_pdf.api.document import PdfPage
from core_pdf.impl._impl.document.structure import PageStructure
from core_pdf.impl._impl.fonts.decoder import FontDecoder
from core_pdf.impl._impl.model.glyphs import GlyphObservation
from core_pdf.impl.spec.s_07_content.marked_content import MarkedContentEntry
from core_pdf.impl.spec.s_09_fonts.glyphs import glyph_name_to_unicode

internal_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}


def internal_native_glyphs(glyphs: Iterable[GlyphObservation]) -> Iterator[GlyphObservation]:
    # Native capture may split one PDF glyph into multiple Unicode observations.
    # Rejoin only observations carrying the same source cluster, never adjacent letters.
    for _, members in groupby(glyphs, key=lambda g: g.cluster_key or id(g)):
        cluster = list(members)
        first, last = cluster[0], cluster[-1]
        if first.source_glyphs:
            sources = list(internal_native_glyphs(first.source_glyphs))
            yield from internal_replace_glyphs(sources, first.text)
            continue
        value = "".join(g.text for g in cluster)
        decoder = first.font_decoder
        if isinstance(decoder, FontDecoder):
            raw = None
            if decoder.to_unicode is not None:
                raw = decoder.to_unicode.mappings.get(first.code_bytes)
            code = first.char_code
            if (
                raw is None
                and code is not None
                and 0 <= code < len(decoder.simple_encoding_glyph_names)
            ):
                raw = glyph_name_to_unicode(decoder.simple_encoding_glyph_names[code])
                if (
                    not raw
                    and not decoder.is_cid_font
                    and first.unicode_source in ("encoding", "glyph_name")
                ):
                    # The native reader also recognizes TeX and other extended
                    # glyph names. MuPDF retains the original code when AGL
                    # has no mapping for the simple font's encoded name.
                    value = chr(code)
                    first = replace(first, unicode_source="identity")
            if raw and not decoder.is_cid_font and first.unicode_source == "identity":
                value = raw
                first = replace(first, unicode_source="glyph_name")
            if raw is not None and raw in internal_LIGATURES:
                value = raw
            if (
                len(value) == 1
                and (ord(value) < 8 or 14 <= ord(value) < 32 or 127 <= ord(value) < 160)
                and code is not None
                and 0 <= code < len(decoder.simple_encoding_glyph_names)
            ):
                fallback = glyph_name_to_unicode(decoder.simple_encoding_glyph_names[code])
                if fallback:
                    value = fallback
        baseline = first.baseline
        if baseline is not None and last.baseline is not None:
            baseline = (*baseline[:2], *last.baseline[2:])
        yield replace(first, text=value, baseline=baseline)


def internal_replacement_tail(
    source: GlyphObservation | None, text: str, *, at_endpoint: bool
) -> Iterator[GlyphObservation]:
    if source is None or source.baseline is None or not text:
        return
    point = source.baseline[2:] if at_endpoint else source.baseline[:2]
    yield replace(
        source,
        text=text,
        baseline=(*point, *point),
        char_code=None,
        cid=None,
        gid=None,
        code_bytes=b"",
        unicode_source="actual_text",
    )


def internal_text_span_key(glyph: GlyphObservation) -> tuple[object, ...]:
    path = dict(glyph.provenance).get("marked_content_path")
    scopes = tuple(id(entry) for entry in path) if isinstance(path, tuple) else ()
    return (
        scopes,
        glyph.text_object_id,
        id(glyph.font_decoder),
        glyph.font_size,
        glyph.text_render_mode,
        glyph.fill,
        glyph.stroke_color,
        glyph.glyph_transform[:4] if glyph.glyph_transform is not None else None,
    )


def internal_replace_glyphs(
    sources: list[GlyphObservation], replacement: str
) -> Iterator[GlyphObservation]:
    """Preserve matching edges while consuming a replacement across text spans."""
    remaining = replacement
    last_source = None
    for _, members in groupby(sources, key=internal_text_span_key):
        span = list(members)
        if not remaining:
            continue
        prefix = 0
        while prefix < min(len(span), len(remaining)) and span[prefix].text == remaining[prefix]:
            prefix += 1
        yield from span[:prefix]
        remaining = remaining[prefix:]
        if prefix:
            last_source = span[prefix - 1]
        if prefix == len(span):
            continue
        suffix = 0
        while (
            suffix < min(len(span) - prefix, len(remaining))
            and span[-suffix - 1].text == remaining[-suffix - 1]
        ):
            suffix += 1
        end = len(span) - suffix
        text_end = len(remaining) - suffix
        middle = remaining[:text_end]
        available = span[prefix:end]
        for source, character in zip(available, middle):
            yield replace(source, text=character, unicode_source="actual_text")
        remaining = remaining[min(len(available), len(middle)) :]
        if available:
            last_source = available[-1]
        if suffix:
            yield from internal_replacement_tail(
                last_source, remaining[:-suffix], at_endpoint=False
            )
            yield from span[end:]
            remaining = ""
    yield from internal_replacement_tail(last_source, remaining, at_endpoint=True)


class internal_StructureText:
    def __init__(self, page: PdfPage) -> None:
        self.page = page
        self.structure: PageStructure | None = None
        self.checked = False

    def get(self, mcid: int) -> str | None:
        if not self.checked:
            self.checked = True
            try:
                self.structure = self.page.structure
            except ValueError:
                # Optional marked-content text cannot require a usable tree.
                return None
        if self.structure is None or not 0 <= mcid < len(self.structure):
            return None
        try:
            element = self.structure[mcid]
            return element.actual_text if element is not None else None
        except ValueError:
            return None


def capture_glyphs(page: PdfPage, glyphs: Iterable[GlyphObservation]) -> Iterator[GlyphObservation]:
    structure = internal_StructureText(page)

    def scope_key(glyph: GlyphObservation) -> int | None:
        scope = dict(glyph.provenance).get("marked_content_scope")
        return id(scope) if scope is not None else None

    for scope, members in groupby(internal_native_glyphs(glyphs), key=scope_key):
        if scope is None:
            yield from members
            continue
        sources = list(members)
        first = sources[0]
        provenance = dict(first.provenance)
        mcid = provenance.get("mcid")
        path = provenance.get("marked_content_path")
        inline_replacement = isinstance(path, tuple) and any(
            isinstance(entry, MarkedContentEntry) and entry.actual_text is not None
            for entry in path
        )
        replacement = (
            structure.get(mcid)
            if type(mcid) is int
            and not provenance.get("xobject_depth")
            and provenance.get("source") == "native_text"
            and not inline_replacement
            and not any(glyph.unicode_source == "actual_text" for glyph in sources)
            else None
        )
        if replacement is None:
            yield from sources
        else:
            yield from internal_replace_glyphs(sources, replacement)
