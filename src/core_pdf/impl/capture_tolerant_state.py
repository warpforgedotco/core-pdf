# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import typing
from collections.abc import Mapping

from core_pdf.impl.capture_recovery import CaptureRecovery
from core_pdf.impl.fonts_helpers import strip_subset_tag
from core_pdf.impl.graphics_color_spec import parse_color_space
from core_pdf.impl.graphics_functions import compile_pdf_function
from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.scalars import clamp01
from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
from core_pdf_spec.s_07_content.operations import (
    ContentOperands,
    OperationHandler,
)
from core_pdf_spec.s_07_content.streams import ContentStreamFrame
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_08_graphics.color_spec import ColorSpace
from core_pdf_spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix
from core_pdf_spec.s_09_fonts.service import FontService as FontDecoder
from core_pdf_spec.s_11_transparency.soft_masks import SoftMask, parse_soft_mask
from core_pdf_spec.types import PdfReference, PdfString

FontCompanions = dict[str, tuple[tuple[int, int], ...]]
FontCompanionsCache = dict[int, tuple[dict[str, object], FontCompanions]]


def font_companions(
    fonts: dict[str, object],
    resolve: typing.Callable[[object], object],
    cache: FontCompanionsCache,
) -> FontCompanions:
    entry = cache.get(id(fonts))
    if entry is not None and entry[0] is fonts:
        return entry[1]
    grouped: dict[str, list[tuple[int, int]]] = {}
    for reference in fonts.values():
        # font_signature only calls this once every value is a reference.
        if type(reference) is not PdfReference:
            continue
        sibling = resolve(reference)
        if not isinstance(sibling, dict):
            continue
        name = strip_subset_tag(recover_pdf_name(sibling.get("BaseFont")) or "")
        if name:
            grouped.setdefault(name, []).append(
                (reference.object_number, reference.generation_number)
            )
    companions = {name: tuple(sorted(refs)) for name, refs in grouped.items()}
    cache[id(fonts)] = (fonts, companions)
    return companions


def font_signature(
    font_ref: PdfReference,
    font_obj: object,
    resources: object,
    resolve: typing.Callable[[object], object],
    companions_cache: FontCompanionsCache,
) -> object | None:
    if not isinstance(resources, dict) or not isinstance(font_obj, dict):
        return None
    fonts = resolve(resources.get("Font"))
    if not isinstance(fonts, dict):
        return None
    if any(type(value) is not PdfReference for value in fonts.values()):
        return None
    base_name = strip_subset_tag(recover_pdf_name(font_obj.get("BaseFont")) or "")
    companions: tuple[tuple[int, int], ...] = ()
    if base_name:
        companions = font_companions(fonts, resolve, companions_cache).get(base_name, ())
    return (font_ref.object_number, font_ref.generation_number, companions)


# Content streams re-state the same colour relentlessly: one corpus page
# issues RG 18,612 times with a single distinct operand tuple. The cache is
# cleared wholesale rather than evicted, since a page that exceeds this many
# distinct colours is not the case the cache exists for.
COLOR_CACHE_LIMIT = 4096

# Soft masks get the same treatment for a harsher reason. Every gs operator
# re-parses the ExtGState soft mask into a fresh SoftMask carrying a freshly
# compiled transfer closure, and each cache downstream is keyed by one of those
# identities, so none of them could ever hit: PyMuPDF/tests/resources/
# test_3450.pdf parsed 11,136 masks and rasterized the same 802 again and
# again, taking 137s and 7.9GB for half a megapixel.
#
# The bound matches the colour cache rather than undercutting it. An entry is
# three pointers to objects the capture already holds elsewhere, so a larger
# table costs almost nothing, while a bound below the working set would clear
# and refill on exactly the pages this exists for: that page peaks at 2,397
# entries, and reaches it without a single clear at this limit.
SOFT_MASK_CACHE_LIMIT = COLOR_CACHE_LIMIT


class RecoveringTextState(ContentInterpreter):
    recovery: CaptureRecovery
    normalized_colors: dict[tuple[ColorSpace, tuple[object, ...]], tuple[float, ...]]
    parsed_soft_masks: dict[tuple[int, Matrix, int], tuple[object, object, SoftMask | None]]

    def resolve_soft_mask(self, value: object) -> SoftMask | None:
        # The ctm is baked into the parsed mask, and the resource scope decides
        # what the mask's own content stream can name, so both belong in the
        # key: a mask reached through different resources stays a separate
        # object, exactly as it was before this cache existed.
        cache = self.parsed_soft_masks
        resources = self.resources
        ctm = self.graphics.ctm
        key = (id(value), ctm, id(resources))
        cached = cache.get(key)
        if cached is not None and cached[0] is value and cached[1] is resources:
            return cached[2]
        mask = parse_soft_mask(
            value,
            self.resolver,
            ctm=ctm,
            compile_function=compile_pdf_function,
        )
        if len(cache) >= SOFT_MASK_CACHE_LIMIT:
            cache.clear()
        # The source and the resources lead the entry so their ids cannot be
        # handed to another object while it is live, and so the identity check
        # above reads them without unpacking the result.
        cache[key] = (value, resources, mask)
        return mask

    def operation_table(self) -> Mapping[str, OperationHandler]:
        """Every operator this state handles, an override winning over its default.

        Both dispatch routes read this: the capture executor once per content
        stream, execute_operation once per call. Overrides are empty outside
        tests, so the common answer is the default table itself.
        """
        overrides = self.operator_overrides
        if not overrides:
            return self.default_handlers
        return {**self.default_handlers, **overrides}

    def execute_operation(
        self, name: str, operands: ContentOperands, depth: int
    ) -> ContentStreamFrame | None:
        handler = self.operation_table().get(name)
        return handler(operands, depth) if handler is not None else None

    capture_font_decoders: dict[object, list[tuple[object, object, FontDecoder]]]
    capture_font_companions: FontCompanionsCache

    def get_decoder(self) -> FontDecoder:
        if self.graphics.current_decoder is not None:
            return self.graphics.current_decoder

        try:
            font_obj_ref = (
                self.lookup_page_resource("Font", self.graphics.current_font)
                if self.graphics.current_font
                else None
            )
        except PdfParseError as error:
            font_obj_ref = self.reject(error, "font-resource", None)
        if font_obj_ref is None:
            return self.font_provider({}, self.resources)

        try:
            font_obj = self.resolver.resolve(font_obj_ref)
        except PdfParseError as error:
            font_obj = self.reject(error, "font-resolution", None)
        if isinstance(font_obj, PdfStream):
            font_obj = font_obj.dictionary
        resources = self.resources
        if isinstance(font_obj_ref, PdfReference):
            font_key: object = (font_obj_ref.object_number, font_obj_ref.generation_number)
        else:
            font_key = id(font_obj)
        owned = self.capture_font_decoders.setdefault(font_key, [])
        for owner_resources, owner_font, decoder in owned:
            if owner_resources is resources and owner_font is font_obj:
                self.graphics.current_decoder = decoder
                self.graphics.decoder_resources = resources
                return decoder

        document_decoders: dict[object, FontDecoder] | None = getattr(
            getattr(self, "document", None), "font_decoders", None
        )
        signature = None
        if document_decoders is not None and isinstance(font_obj_ref, PdfReference):
            signature = font_signature(
                font_obj_ref,
                font_obj,
                resources,
                self.resolver.resolve,
                self.capture_font_companions,
            )
        if signature is not None and document_decoders is not None:
            shared = document_decoders.get(signature)
            if shared is not None:
                owned.append((resources, font_obj, shared))
                self.graphics.current_decoder = shared
                self.graphics.decoder_resources = resources
                return shared

        if not isinstance(font_obj, dict):
            decoder = self.font_provider({}, resources)
        else:
            font_dict = font_obj
            resolved_font = self.resolver.resolve_font_dict(font_dict)
            decoder = self.font_provider(resolved_font, resources)
        owned.append((resources, font_obj, decoder))
        if signature is not None and document_decoders is not None:
            document_decoders[signature] = decoder
        self.graphics.current_decoder = decoder
        self.graphics.decoder_resources = resources
        return decoder

    def resolve_form_resources(self, value: object) -> PdfDict:
        return self.resolve_resources(value) or self.resources

    def tj_array_extra_bytes(self, item: object) -> bytes:
        return item.encode("latin-1") if type(item) is str else b""

    def op_Tj(self, operands: ContentOperands, depth: int) -> None:
        if not operands:
            return
        decoder = self.get_decoder()
        operand = operands[-1]
        if type(operand) is PdfString:
            self.append_text(operand.data, decoder=decoder)
        elif type(operand) is bytes:
            self.append_text(operand, decoder=decoder)
        else:
            text = operand if type(operand) is str else self.resolver.resolve_str(operand)
            if text is None:
                return
            data = text.encode("latin-1", "replace")
            self.append_decoded_text(text, data, decoder.decode_glyphs(data), decoder)

    def parse_named_color_space(self, value: object, name: str) -> ColorSpace:
        try:
            return parse_color_space(value)
        except (ValueError, TypeError) as error:
            base = value[0] if isinstance(value, (list, tuple)) and value else value
            return self.reject(error, "color-space", ColorSpace(recover_pdf_name(base) or name, ()))

    def color_component_values(self, components: typing.Sequence[object]) -> tuple[float, ...]:
        return tuple(self.as_float(value) for value in components)

    def recover_color_components(
        self, components: typing.Sequence[object]
    ) -> tuple[float, ...] | None:
        values: list[float] = []
        for component in components:
            try:
                values.append(clamp01(self.as_float(component)))
            except ValueError as error:
                return self.reject(error, "color-components", None)
        if not values:
            return None
        return tuple(values)

    def normalize_color_components(
        self, spec: ColorSpace, components: typing.Sequence[object]
    ) -> tuple[float, ...] | None:
        cache_key = (spec, tuple(components))
        try:
            cached = self.normalized_colors.get(cache_key)
            cacheable = True
        except TypeError:  # an unhashable operand, e.g. a malformed array
            cached = None
            cacheable = False
        if cached is not None:
            return cached
        normalized = super().normalize_color_components(spec, components)
        # A recovered colour is cached too: it is a function of the same key.
        if cacheable and normalized is not None:
            if len(self.normalized_colors) >= COLOR_CACHE_LIMIT:
                self.normalized_colors.clear()
            self.normalized_colors[cache_key] = normalized
        return normalized

    def reject[T](self, error: Exception, context: str, fallback: T) -> T:
        # Recovery proceeds with the reader's fallback wherever the spec
        # interpreter would refuse the content.
        return fallback

    def matrix_operand(self, value: object, context: str) -> Matrix:
        value = self.resolver.deep_resolve(value)
        if value is None:
            return IDENTITY_MATRIX
        try:
            return Matrix.from_operand(value)
        except ValueError:
            if context == "form" and isinstance(value, (list, tuple)) and len(value) > 6:
                return Matrix.from_operand(value[:6])
            if context == "pattern":
                return IDENTITY_MATRIX
            raise
