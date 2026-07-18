# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from core_filters.impl.decode_spec import (
    StreamDecodeSpec,
    normalize_stream_decode_spec,
)

from core_pdf.impl.engine.extraction.cache import ExtractionCache
from core_pdf.impl.engine.spec.s_07_content.inline_images import parse_inline_image
from core_pdf.impl.engine.spec.s_07_content.operations import (
    content_stream_may_show_text,
)
from core_pdf.impl.engine.spec.s_07_objects.coercion import (
    normalize_pdf_name,
)
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.engine.spec.s_07_syntax.lexer_helpers import (
    full_source_bytes,
    skip_pdf_ignored,
)
from core_pdf.impl.engine.spec.s_07_syntax.scanning import (
    skip_hex_string,
    skip_literal_string,
    skip_name,
)
from core_pdf.impl.engine.spec.s_07_syntax.tokens import SEPARATOR_TABLE, WS_TABLE
from core_pdf.impl.exceptions import PdfParseError
from core_pdf.impl.objects import PdfStream
from core_pdf.impl.types import PdfDict

TEXT_SHOW_OPS = frozenset({"Tj", "TJ", "'", '"'})
TEXT_STATE_OPS = frozenset(
    {"BT", "ET", "Tc", "Tw", "Tz", "TL", "Tf", "Tr", "Ts", "Td", "TD", "Tm", "T*"}
)
PATH_OPS = frozenset(
    {
        "m",
        "l",
        "c",
        "v",
        "y",
        "h",
        "re",
        "S",
        "s",
        "f",
        "F",
        "f*",
        "B",
        "B*",
        "b",
        "b*",
        "n",
    }
)
CLIP_OPS = frozenset({"W", "W*"})
MARKED_CONTENT_OPS = frozenset({"BMC", "BDC", "EMC", "MP", "DP"})
IMAGE_OPS = frozenset({"BI", "ID", "EI"})
GRAPHICS_STATE_OPS = frozenset({"q", "Q", "cm", "gs"})
COUNTED_CONTENT_OPS = frozenset(
    {
        "Do",
        *TEXT_SHOW_OPS,
        *TEXT_STATE_OPS,
        *PATH_OPS,
        *CLIP_OPS,
        *MARKED_CONTENT_OPS,
        *IMAGE_OPS,
        *GRAPHICS_STATE_OPS,
    }
)
COUNTED_CONTENT_OP_BYTES = {op.encode("latin-1"): op for op in COUNTED_CONTENT_OPS}
COUNTED_CONTENT_OP_STARTS = frozenset(op[0] for op in COUNTED_CONTENT_OP_BYTES)
MAX_COUNTED_CONTENT_OP_BYTES = max(len(op) for op in COUNTED_CONTENT_OP_BYTES)


class PageProfileHost(Protocol):
    document: Any
    extraction_cache: ExtractionCache | None

    @property
    def resources(self) -> PdfDict: ...

    @property
    def content_streams(self) -> tuple[PdfStream, ...]: ...


@dataclass(frozen=True)
class ContentStreamProfile:
    raw_bytes: int
    decoded_bytes: int | None
    filters: tuple[str, ...]
    decode_error: str | None
    may_show_text: bool
    text_show_ops: int
    text_state_ops: int
    xobject_ops: int
    inline_image_ops: int
    path_ops: int
    clip_ops: int
    marked_content_ops: int
    graphics_state_ops: int

    @property
    def decoded(self) -> bool:
        return self.decode_error is None

    @property
    def has_text_showing_ops(self) -> bool:
        return self.text_show_ops > 0

    @property
    def has_possible_text_content(self) -> bool:
        return self.may_show_text or self.xobject_ops > 0

    @property
    def has_drawn_content(self) -> bool:
        return self.inline_image_ops > 0 or self.path_ops > 0 or self.xobject_ops > 0


@dataclass(frozen=True)
class ResourceProfile:
    font_count: int
    xobject_count: int
    image_xobject_count: int
    form_xobject_count: int
    unknown_xobject_count: int

    @property
    def has_xobjects(self) -> bool:
        return self.xobject_count > 0

    @property
    def has_images(self) -> bool:
        return self.image_xobject_count > 0

    @property
    def has_forms(self) -> bool:
        return self.form_xobject_count > 0


@dataclass(frozen=True)
class PageProfile:
    content_streams: tuple[ContentStreamProfile, ...]
    resources: ResourceProfile
    recommended_strategy: str

    @property
    def content_stream_count(self) -> int:
        return len(self.content_streams)

    @property
    def raw_content_bytes(self) -> int:
        return sum(stream.raw_bytes for stream in self.content_streams)

    @property
    def decoded_content_bytes(self) -> int:
        return sum(stream.decoded_bytes or 0 for stream in self.content_streams)

    @property
    def decode_errors(self) -> tuple[str, ...]:
        return tuple(
            stream.decode_error
            for stream in self.content_streams
            if stream.decode_error is not None
        )

    @property
    def fully_scanned(self) -> bool:
        return not self.decode_errors

    @property
    def has_text_showing_ops(self) -> bool:
        return any(stream.has_text_showing_ops for stream in self.content_streams)

    @property
    def has_xobject_ops(self) -> bool:
        return any(stream.xobject_ops > 0 for stream in self.content_streams)

    @property
    def has_inline_images(self) -> bool:
        return any(stream.inline_image_ops > 0 for stream in self.content_streams)

    @property
    def has_path_ops(self) -> bool:
        return any(stream.path_ops > 0 for stream in self.content_streams)

    @property
    def has_drawn_content(self) -> bool:
        return any(stream.has_drawn_content for stream in self.content_streams)

    @property
    def likely_text_page(self) -> bool:
        return self.has_text_showing_ops or (self.has_xobject_ops and self.resources.has_forms)

    @property
    def likely_image_page(self) -> bool:
        return self.has_inline_images or (
            self.has_xobject_ops and self.resources.has_images and not self.likely_text_page
        )

    @property
    def likely_table_page(self) -> bool:
        return sum(stream.path_ops for stream in self.content_streams) >= 12

    @property
    def can_skip_native_text(self) -> bool:
        if not self.fully_scanned:
            return False
        return not self.has_text_showing_ops and not self.has_xobject_ops

    @property
    def can_skip_all_text(self) -> bool:
        if not self.can_skip_native_text:
            return False
        return not self.has_inline_images and not self.has_path_ops


def get_page_profile(page: PageProfileHost) -> PageProfile:
    cache = page.extraction_cache
    if cache is None:
        page.extraction_cache = cache = ExtractionCache()
    cached = cache.get("page_profile")
    if isinstance(cached, PageProfile):
        return cached

    profile = build_page_profile(page)
    cache["page_profile"] = profile
    return profile


def build_page_profile(page: PageProfileHost) -> PageProfile:
    content_profiles = tuple(content_stream_profile(stream) for stream in page.content_streams)
    resource_profile = page_resource_profile(page)
    return PageProfile(
        content_streams=content_profiles,
        resources=resource_profile,
        recommended_strategy=recommended_page_strategy(
            content_profiles,
            resource_profile,
        ),
    )


def content_stream_profile(stream: PdfStream) -> ContentStreamProfile:
    filters = stream_filter_names(stream)
    try:
        data = stream.data_view
    except PdfParseError as exc:
        return ContentStreamProfile(
            raw_bytes=len(stream.raw_data),
            decoded_bytes=None,
            filters=filters,
            decode_error=str(exc),
            may_show_text=False,
            text_show_ops=0,
            text_state_ops=0,
            xobject_ops=0,
            inline_image_ops=0,
            path_ops=0,
            clip_ops=0,
            marked_content_ops=0,
            graphics_state_ops=0,
        )

    may_show_text = content_stream_may_show_text(data)
    counts = content_operator_counts(
        data,
        profile_thresholds=True,
        may_show_text=may_show_text,
    )
    return ContentStreamProfile(
        raw_bytes=len(stream.raw_data),
        decoded_bytes=len(data),
        filters=filters,
        decode_error=None,
        may_show_text=may_show_text,
        text_show_ops=sum(counts.get(op, 0) for op in TEXT_SHOW_OPS),
        text_state_ops=sum(counts.get(op, 0) for op in TEXT_STATE_OPS),
        xobject_ops=counts.get("Do", 0),
        inline_image_ops=counts.get("BI", 0),
        path_ops=sum(counts.get(op, 0) for op in PATH_OPS),
        clip_ops=sum(counts.get(op, 0) for op in CLIP_OPS),
        marked_content_ops=sum(counts.get(op, 0) for op in MARKED_CONTENT_OPS),
        graphics_state_ops=sum(counts.get(op, 0) for op in GRAPHICS_STATE_OPS),
    )


def stream_filter_names(stream: PdfStream) -> tuple[str, ...]:
    spec = stream.spec
    if isinstance(spec, StreamDecodeSpec):
        return spec.filters
    dictionary = spec if isinstance(spec, dict) else stream.dictionary
    try:
        return normalize_stream_decode_spec(dictionary).filters
    except PdfParseError:
        return ()


def content_operator_counts(
    data: bytes | memoryview,
    *,
    profile_thresholds: bool = False,
    may_show_text: bool | None = None,
) -> dict[str, int]:
    data_len = len(data)
    raw_bytes: bytes | memoryview
    source_bytes = full_source_bytes(data)
    raw_bytes = source_bytes if source_bytes is not None else data

    inline_image_possible = True
    if profile_thresholds and type(raw_bytes) is bytes:
        inline_image_possible = raw_bytes.find(b"BI") >= 0
    text_or_xobject_possible = may_show_text
    if text_or_xobject_possible is None:
        text_or_xobject_possible = True

    counts: dict[str, int] = {}
    path_ops = 0
    inline_image_lexer: PdfLexer | None = None
    container_depth = 0
    pos = 0
    separator_table = SEPARATOR_TABLE
    while pos < data_len:
        byte = raw_bytes[pos]
        if WS_TABLE[byte] or byte == 37:
            pos = skip_pdf_ignored(raw_bytes, pos, data_len)
            continue
        if byte == 40:
            pos = skip_literal_string(raw_bytes, pos, data_len)
            continue
        if byte == 60:
            if pos + 1 < data_len and raw_bytes[pos + 1] == 60:
                container_depth += 1
                pos += 2
            else:
                pos = skip_hex_string(raw_bytes, pos, data_len)
            continue
        if byte == 62 and pos + 1 < data_len and raw_bytes[pos + 1] == 62:
            container_depth = max(0, container_depth - 1)
            pos += 2
            continue
        if byte == 91:
            container_depth += 1
            pos += 1
            continue
        if byte == 93:
            container_depth = max(0, container_depth - 1)
            pos += 1
            continue
        if byte == 47:
            pos = skip_name(raw_bytes, pos, data_len)
            continue
        if byte in (41, 62, 91, 93, 123, 125):
            pos += 1
            continue

        start = pos
        while pos < data_len and not separator_table[raw_bytes[pos]]:
            pos += 1
        if start == pos:
            pos += 1
            continue
        if not profile_thresholds:
            token = bytes(raw_bytes[start:pos]).decode("latin-1", "ignore")
            if token:
                counts[token] = counts.get(token, 0) + 1
            continue
        token_len = pos - start
        if (
            container_depth == 0
            and token_len <= MAX_COUNTED_CONTENT_OP_BYTES
            and raw_bytes[start] in COUNTED_CONTENT_OP_STARTS
        ):
            op = COUNTED_CONTENT_OP_BYTES.get(bytes(raw_bytes[start:pos]))
            if op is not None:
                if op in PATH_OPS:
                    path_ops += 1
                    if path_ops <= 12:
                        counts[op] = counts.get(op, 0) + 1
                else:
                    counts.setdefault(op, 1)
                if op == "BI":
                    if inline_image_lexer is None:
                        inline_image_lexer = PdfLexer(raw_bytes)
                    inline_image_lexer.pos = pos
                    try:
                        parse_inline_image(inline_image_lexer)
                    except PdfParseError:
                        pass
                    else:
                        pos = inline_image_lexer.pos
        if (
            profile_thresholds
            and path_ops >= 12
            and not text_or_xobject_possible
            and not inline_image_possible
        ):
            break
    return counts


def page_resource_profile(page: PageProfileHost) -> ResourceProfile:
    resources = page.resources
    fonts = lookup_dict_key(resources, "Font")
    xobjects = lookup_dict_key(resources, "XObject")
    font_count = len(fonts) if isinstance(fonts, dict) else 0
    if not isinstance(xobjects, dict):
        return ResourceProfile(
            font_count=font_count,
            xobject_count=0,
            image_xobject_count=0,
            form_xobject_count=0,
            unknown_xobject_count=0,
        )

    image_count = 0
    form_count = 0
    unknown_count = 0
    for value in xobjects.values():
        subtype = xobject_subtype(page, value)
        if subtype == "Image":
            image_count += 1
        elif subtype == "Form":
            form_count += 1
        else:
            unknown_count += 1

    return ResourceProfile(
        font_count=font_count,
        xobject_count=len(xobjects),
        image_xobject_count=image_count,
        form_xobject_count=form_count,
        unknown_xobject_count=unknown_count,
    )


def xobject_subtype(page: PageProfileHost, value: Any) -> str | None:
    try:
        resolved = page.document.resolver.resolve(value)
    except Exception:
        resolved = value
    dictionary = resolved.dictionary if isinstance(resolved, PdfStream) else resolved
    if not isinstance(dictionary, dict):
        return None
    return normalize_pdf_name(lookup_dict_key(dictionary, "Subtype"))


def recommended_page_strategy(
    content_profiles: tuple[ContentStreamProfile, ...],
    resources: ResourceProfile,
) -> str:
    if not content_profiles:
        return "empty"
    if any(stream.decode_error is not None for stream in content_profiles):
        return "generic"
    if any(stream.has_text_showing_ops for stream in content_profiles):
        if sum(stream.path_ops for stream in content_profiles) >= 12:
            return "text_table"
        return "native_text"
    has_xobject_ops = any(stream.xobject_ops > 0 for stream in content_profiles)
    if has_xobject_ops and resources.has_forms:
        return "form_xobject"
    if has_xobject_ops and resources.has_images:
        return "image"
    if any(stream.inline_image_ops > 0 for stream in content_profiles):
        return "image"
    if sum(stream.path_ops for stream in content_profiles) >= 12:
        return "vector_or_table"
    return "empty"


__all__ = (
    "ContentStreamProfile",
    "PageProfile",
    "ResourceProfile",
    "build_page_profile",
    "content_operator_counts",
    "content_stream_profile",
    "get_page_profile",
)
