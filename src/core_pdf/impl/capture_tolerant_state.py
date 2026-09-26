# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import typing
from collections.abc import Mapping
from math import isfinite

from core_pdf.impl.capture_recovery import CaptureRecovery
from core_pdf.impl.fonts_helpers import strip_subset_tag
from core_pdf.impl.graphics_color_spec import parse_color_space
from core_pdf.impl.graphics_functions import compile_pdf_function
from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.recovery_resolver import (
    resolve_resource_dict as recover_resources,
)
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


# Exactly these: bool subclasses int and must keep failing as it does.
NUMERIC_TYPES = frozenset((int, float))


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

    def as_floats(self, operands: ContentOperands, count: int) -> tuple[float, ...] | None:
        if len(operands) < count:
            return self.reject(PdfParseError("missing numeric operand"), "numeric-operands", None)
        # The content tokenizer already produced int and float operands, but
        # reaching them through as_float costs four call frames each
        # (as_float -> parse_float_strict -> parse_float) to re-derive a value
        # that is already a float. Path operators run this per segment, so the
        # already-numeric cases are handled inline and anything else still
        # falls back to the full coercion.
        # Operands that are exactly the finite floats asked for already form
        # the tuple the loop below would build, so it is returned as it is.
        if len(operands) == count and type(operands) is tuple:
            for value in operands:
                if type(value) is not float or not isfinite(value):
                    break
            else:
                return operands  # type: ignore[return-value]  # ty: ignore[invalid-return-type]
            # Ints and floats alike -- "0 0 612 792 re", a glyph procedure's
            # integer curves -- convert in C: float() of each, the value the
            # loop below appends, then its finiteness. An overflow or a
            # non-finite value falls through to the loop, which raises as it
            # always has.
            if all(map(NUMERIC_TYPES.__contains__, map(type, operands))):
                try:
                    converted = tuple(map(float, operands))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                except OverflowError:
                    pass
                else:
                    if all(map(isfinite, converted)):
                        return converted
        try:
            values: list[float] = []
            append = values.append
            for index in range(count):
                value = operands[index]
                kind = type(value)
                # `type(...) is int` rather than isinstance: bool subclasses
                # int and must keep failing the way the full coercion fails it.
                # No typing.cast around these: `kind is float` has already
                # established the type for a reader, and cast is a function
                # call that returns its argument, run here once per operand of
                # every path segment on the page.
                if kind is float:
                    if not isfinite(value):  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                        raise ValueError("invalid numeric operand")
                    append(value)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                elif kind is int:
                    try:
                        append(float(value))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                    except OverflowError:
                        raise ValueError("invalid numeric operand") from None
                else:
                    append(self.as_float(value))
            return tuple(values)
        except (TypeError, ValueError) as error:
            return self.reject(error, "numeric-operands", None)

    def as_int_operand(self, operands: ContentOperands) -> int | None:
        if not operands:
            return self.reject(PdfParseError("missing numeric operand"), "integer-operand", None)
        value = operands[0]
        # An int operand is its own answer; as_int's three frames only reach
        # the same `type(value) is int` test. J and j run this per path.
        if type(value) is int:
            return value
        try:
            return self.as_int(value)
        except (TypeError, ValueError) as error:
            return self.reject(error, "integer-operand", None)

    def op_BX(self, operands: ContentOperands, depth: int) -> None:
        self.compatibility_depth += 1

    def op_EX(self, operands: ContentOperands, depth: int) -> None:
        self.compatibility_depth = max(0, self.compatibility_depth - 1)

    def append_cubic_curve(
        self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float
    ) -> None:
        # Spec rejects a curve with no current point; recovery moves there.
        if self.current_point is None:
            self.current_point = (x3, y3)
            return
        super().append_cubic_curve(x1, y1, x2, y2, x3, y3)

    def op_w(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 1)) is not None:
            self.graphics.line_width = max(0.0, values[0])

    def op_M(self, operands: ContentOperands, depth: int) -> None:
        if (values := self.as_floats(operands, 1)) is not None:
            self.graphics.miter_limit = max(1.0, values[0])

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
        try:
            values = tuple(self.as_float(value) for value in components)
            normalized = super().normalize_color_components(spec, values)
            if cacheable and normalized is not None:
                if len(self.normalized_colors) >= COLOR_CACHE_LIMIT:
                    self.normalized_colors.clear()
                self.normalized_colors[cache_key] = normalized
            return normalized
        except (PdfParseError, TypeError, ValueError) as error:
            if spec.kind in {"Indexed", "Lab"}:
                return self.reject(error, "color-components", None)
            recovered = self.recover_color_components(components)
            return self.reject(error, "color-components", recovered)

    def initial_color_components(
        self, spec: ColorSpace, *, stroke: bool
    ) -> tuple[float, ...] | None:
        try:
            return super().initial_color_components(spec, stroke=stroke)
        except PdfParseError as error:
            return self.reject(
                error,
                "color-space",
                self.graphics.stroke_color if stroke else self.graphics.fill_color,
            )

    def resolve_resources(self, value: object) -> PdfDict | None:
        return recover_resources(value, self.resolver)

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
