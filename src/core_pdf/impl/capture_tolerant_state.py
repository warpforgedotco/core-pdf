# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import typing
from collections.abc import Mapping

from core_pdf.impl.capture_recovery import CaptureRecovery
from core_pdf.impl.fonts_helpers import recover_strip_subset_tag
from core_pdf.impl.graphics_color_spec import parse_color_space
from core_pdf.impl.graphics_functions import compile_pdf_function
from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.scalars import clamp01
from core_pdf_spec.s_07_content.interpreter import ContentInterpreter
from core_pdf_spec.s_07_content.operations import (
    ContentOperands,
    OperationHandler,
)
from core_pdf_spec.s_07_content.streams import ContentStreamFrame
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
        if type(reference) is not PdfReference:
            continue
        sibling = resolve(reference)
        if not isinstance(sibling, dict):
            continue
        name = recover_strip_subset_tag(recover_pdf_name(sibling.get("BaseFont")) or "")
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
    base_name = recover_strip_subset_tag(recover_pdf_name(font_obj.get("BaseFont")) or "")
    companions: tuple[tuple[int, int], ...] = ()
    if base_name:
        companions = font_companions(fonts, resolve, companions_cache).get(base_name, ())
    return (font_ref.object_number, font_ref.generation_number, companions)


COLOR_CACHE_LIMIT = 4096

SOFT_MASK_CACHE_LIMIT = COLOR_CACHE_LIMIT


class RecoveringTextState(ContentInterpreter):
    recovery: CaptureRecovery
    normalized_colors: dict[tuple[ColorSpace, tuple[object, ...]], tuple[float, ...]]
    parsed_soft_masks: dict[tuple[int, Matrix, int], tuple[object, object, SoftMask | None]]

    def resolve_soft_mask(self, value: object) -> SoftMask | None:
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
        cache[key] = (value, resources, mask)
        return mask

    def operation_table(self) -> Mapping[str, OperationHandler]:
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

    def decoder_for(self, font_reference: object, font: object, resources: PdfDict) -> FontDecoder:
        if isinstance(font_reference, PdfReference):
            font_key: object = (font_reference.object_number, font_reference.generation_number)
        else:
            font_key = id(font)
        owned = self.capture_font_decoders.setdefault(font_key, [])
        for owner_resources, owner_font, decoder in owned:
            if owner_resources is resources and owner_font is font:
                return decoder

        document_decoders: dict[object, FontDecoder] | None = getattr(
            getattr(self, "document", None), "font_decoders", None
        )
        signature = None
        if document_decoders is not None and isinstance(font_reference, PdfReference):
            signature = font_signature(
                font_reference,
                font,
                resources,
                self.resolver.resolve,
                self.capture_font_companions,
            )
        if signature is not None and document_decoders is not None:
            shared = document_decoders.get(signature)
            if shared is not None:
                owned.append((resources, font, shared))
                return shared

        decoder = super().decoder_for(font_reference, font, resources)
        owned.append((resources, font, decoder))
        if signature is not None and document_decoders is not None:
            document_decoders[signature] = decoder
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
        except TypeError:
            cached = None
            cacheable = False
        if cached is not None:
            return cached
        normalized = super().normalize_color_components(spec, components)
        if cacheable and normalized is not None:
            if len(self.normalized_colors) >= COLOR_CACHE_LIMIT:
                self.normalized_colors.clear()
            self.normalized_colors[cache_key] = normalized
        return normalized

    def reject[T](self, error: Exception, context: str, fallback: T) -> T:
        return fallback

    def matrix_fallback(self, value: object, context: str) -> Matrix | None:
        if context == "form" and isinstance(value, (list, tuple)) and len(value) > 6:
            return Matrix.from_operand(value[:6])
        if context == "pattern":
            return IDENTITY_MATRIX
        return None
