"""Page-space projection of captured affine operands using reader precision."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import cast

from core_pdf.api.compat._shared import float32
from core_pdf.api.compat.pymupdf.geometry import Matrix, Point, Rect
from core_pdf.api.document import PdfPage
from core_pdf.impl._impl.capture.program import PageProgram
from core_pdf.impl._impl.model.glyphs import GlyphObservation, Matrix6
from core_pdf.impl.spec.s_07_document.annotation_appearance import normal_appearance_stream
from core_pdf.impl.spec.s_08_graphics.matrix import IDENTITY_MATRIX
from core_pdf.impl.spec.s_08_graphics.matrix import Matrix as NativeMatrix


def page_space_matrix(crop: Rect, unit: float, trace: tuple[Matrix6, ...]) -> Matrix:
    result = Matrix(unit, 0, 0, -unit, float32(-crop.x0 * unit), float32(crop.y1 * unit))
    for operands in trace:
        result = Matrix(operands) * result
    return result


def appearance_glyphs(page: PdfPage, program: PageProgram) -> Iterator[GlyphObservation]:
    """Retain reader-precision annotation placement beside the native transform."""
    yield from program.body.glyphs
    resolver = page.document.resolver
    for appearance in program.appearances:
        source = appearance.source
        placement: Matrix | None = None
        if isinstance(source, dict):
            try:
                stream = normal_appearance_stream(resolver, source.get("AP"), source.get("AS"))
                rect_value = resolver.resolve_box(source.get("Rect"))
                bbox_value = (
                    resolver.resolve_box(stream.dictionary.get("BBox"))
                    if stream is not None
                    else None
                )
                if stream is not None and rect_value is not None and bbox_value is not None:
                    rect = Rect(tuple(map(float32, rect_value))).normalize()
                    matrix = Matrix(stream.dictionary.get("Matrix", IDENTITY_MATRIX))
                    bbox = Rect(tuple(map(float32, bbox_value))) * matrix
                    width, height = float32(bbox.width), float32(bbox.height)
                    sx = float32(float32(rect.width) / width) if width else 0.0
                    sy = float32(float32(rect.height) / height) if height else 0.0
                    placement = matrix * Matrix(
                        sx,
                        0,
                        0,
                        sy,
                        float32(rect.x0 - float32(bbox.x0 * sx)),
                        float32(rect.y0 - float32(bbox.y0 * sy)),
                    )
            except (TypeError, ValueError):
                pass
        for glyph in appearance.program.glyphs:
            provenance = dict(glyph.provenance)
            trace = provenance.get("matrix_trace")
            if placement is not None and isinstance(trace, tuple) and trace:
                provenance["reader_matrix_trace"] = (tuple(placement), *trace[1:])
                glyph = replace(glyph, provenance=tuple(provenance.items()))
            yield glyph


def glyph_size_basis(
    glyph: GlyphObservation, unit: float
) -> tuple[float, float, float, float] | None:
    """Use the same rounded appearance placement for the reported font size."""
    provenance = dict(glyph.provenance)
    trace = provenance.get("reader_matrix_trace")
    source = provenance.get("source_text_matrix")
    horizontal_scale = provenance.get("horizontal_scale")
    if not (
        isinstance(trace, tuple)
        and isinstance(source, tuple)
        and len(source) == 6
        and isinstance(horizontal_scale, (int, float))
    ):
        return None
    scale = Matrix(
        float32(glyph.font_size * float32(horizontal_scale * 0.01)),
        0,
        0,
        float32(glyph.font_size),
        0,
        0,
    )
    basis = (
        scale
        * Matrix(source)
        * page_space_matrix(Rect(0, 0, 0, 0), unit, cast(tuple[Matrix6, ...], trace))
    )
    return basis.a, basis.b, basis.c, basis.d


def glyph_group_origin(
    glyph: GlyphObservation,
    crop: Rect,
    unit: float,
    *,
    line_origin: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    """Seed a text group from its source transform before native CTM composition."""
    provenance = dict(glyph.provenance)
    trace_value = provenance.get("matrix_trace")
    source_value = provenance.get("source_text_matrix")
    if (
        not isinstance(trace_value, tuple)
        or not trace_value
        or not isinstance(source_value, tuple)
        or len(source_value) != 6
        or glyph.baseline is None
    ):
        return None
    trace = cast(tuple[Matrix6, ...], trace_value)
    source = cast(Matrix6, source_value)
    if all(operands == IDENTITY_MATRIX for operands in trace):
        return None
    native = IDENTITY_MATRIX
    for operands in trace:
        native = NativeMatrix(*operands).multiply(native)
    native = NativeMatrix(*source).multiply(native)
    reader_source = Matrix(source)
    source_line = provenance.get("source_line_matrix")
    if line_origin is not None and isinstance(source_line, tuple) and len(source_line) == 6:
        line = cast(Matrix6, source_line)
        reader_source.e = float32(line_origin[0] + float32(source[4] - line[4]))
        reader_source.f = float32(line_origin[1] + float32(source[5] - line[5]))
    reader_trace = cast(tuple[Matrix6, ...], provenance.get("reader_matrix_trace", trace))
    projected = reader_source * page_space_matrix(crop, unit, reader_trace)
    rise_value = provenance.get("text_rise")
    rise = float(rise_value) if isinstance(rise_value, (int, float)) else 0.0
    origin_x, origin_y = native.e + native.c * rise, native.f + native.d * rise
    axis_squared = native.a * native.a + native.b * native.b
    if axis_squared == 0:
        return None
    # A captured glyph can start partway through a text-show operation, for
    # example after an empty Unicode mapping or inside an ActualText span.
    distance = (
        (glyph.baseline[0] - origin_x) * native.a + (glyph.baseline[1] - origin_y) * native.b
    ) / axis_squared
    point = Point(distance, rise) * projected
    return point.x, point.y
