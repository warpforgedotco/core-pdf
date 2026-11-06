from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from core_pdf import PdfPage
from core_pdf.impl.capture_program import CapturedProgram
from core_pdf.impl.types import GeneratedRecord
from core_pdf_spec.s_07_content.model import PATH_CLOSE, PATH_CURVE, PATH_OPERAND_COUNTS, PATH_RECT

from ._capture import (
    pdfminer_media_box,
    pdfminer_page_program,
    pdfminer_validate_page_resources,
)
from ._fonts import (
    EMPTY_GLYPH_TEXT,
    _font_value,
    _pdfminer_builtin_width,
    decoder_encoding_differences,
    pdfminer_descent,
    pdfminer_embedded_cmap_is_unusable,
    pdfminer_font_name,
    pdfminer_glyph_text,
    pdfminer_ligature_overrides,
    pdfminer_normalized_width,
)
from ._layout import (
    LAParams,
    LTChar,
    LTFigure,
    LTImage,
    LTPage,
    LTTextLine,
    _group_lines,
    _group_objects,
    _reading_order,
)
from .layout import LTCurve, LTLine, LTRect


class ProjectionPolicy(GeneratedRecord):
    strict_resources: bool
    tolerant_pages: bool


PDFMINER_POLICY = ProjectionPolicy(strict_resources=True, tolerant_pages=False)
UNSTRUCTURED_POLICY = ProjectionPolicy(strict_resources=False, tolerant_pages=True)


def _pdfminer_form_glyph_is_clipped(provenance: dict[str, Any]) -> bool:
    if int(provenance.get("xobject_depth", 0) or 0) <= 0:
        return False
    clip = provenance.get("clip_bbox")
    if not isinstance(clip, (tuple, list)) or len(clip) != 4:
        return False
    left, bottom, right, top = (float(value) for value in clip)
    return right <= left and top <= bottom


def _pdfminer_layout_origin(
    baseline: tuple[float, float, float, float],
    *,
    normalize_noise: bool,
) -> tuple[float, float]:
    origin_x, origin_y = baseline[0], baseline[1]
    if normalize_noise:
        return round(origin_x, 12), round(origin_y, 12)
    return origin_x, origin_y


def _pdfminer_rotated_text_matrix(
    origin_x: float,
    origin_y: float,
    matrix: tuple[float, float, float, float],
    rotation: int,
    page_width: float,
    page_height: float,
) -> tuple[float, float, float, float, float, float]:
    matrix_a, matrix_b, matrix_c, matrix_d = matrix
    if rotation == 90:
        return origin_y, page_width - origin_x, matrix_b, -matrix_a, matrix_d, -matrix_c
    if rotation == 180:
        return (
            page_width - origin_x,
            page_height - origin_y,
            -matrix_a,
            -matrix_b,
            -matrix_c,
            -matrix_d,
        )
    if rotation == 270:
        return page_height - origin_y, origin_x, -matrix_b, matrix_a, -matrix_d, matrix_c
    return origin_x, origin_y, matrix_a, matrix_b, matrix_c, matrix_d


def _pdfminer_layout_figure_box(
    figure_box: tuple[float, float, float, float],
    media_box: tuple[float, float, float, float],
    rotation: int,
    page_width: float,
    page_height: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = figure_box
    media_left, media_bottom, _media_right, _media_top = media_box
    x0 -= media_left
    x1 -= media_left
    y0 -= media_bottom
    y1 -= media_bottom
    if rotation == 90:
        return (y0, page_width - x1, y1, page_width - x0)
    if rotation == 180:
        return (
            page_width - x1,
            page_height - y1,
            page_width - x0,
            page_height - y0,
        )
    if rotation == 270:
        return (page_height - y1, x0, page_height - y0, x1)
    return (x0, y0, x1, y1)


def pdfminer_projected_widths(
    glyph: Any, width_code: int | None, glyph_text: str | None
) -> tuple[float | None, float]:
    """The glyph's font width and its pdfminer advance width, both in text space units."""
    width_lookup = getattr(glyph.font_decoder, "glyph_width", None)
    font_width = (
        float(width_lookup(width_code)) * 0.001
        if width_code is not None and callable(width_lookup)
        else None
    )
    normalized_width = 0.0
    if font_width is not None and getattr(glyph.font_decoder, "is_type3", False):
        # pdfminer scales a Type3 font's /Widths by its /FontMatrix, not by 0.001.
        normalized_width = pdfminer_normalized_width(glyph, glyph_text)
    elif font_width is not None:
        normalized_width = font_width
        builtin_width = _pdfminer_builtin_width(glyph, glyph_text)
        if builtin_width is not None:
            normalized_width = builtin_width * 0.001
        base_font = str(_font_value(glyph.font_decoder.font, "BaseFont"))
        glyph_name = decoder_encoding_differences(glyph.font_decoder).get(glyph.char_code)
        if (
            glyph_name
            and _font_value(glyph.font_decoder.font, "Widths") is None
            and base_font.split("+")[-1] in {"Symbol", "ZapfDingbats"}
        ):
            normalized_width = 0.0
    return font_width, normalized_width


def project_page(
    page: PdfPage,
    params: LAParams | None,
    *,
    policy: ProjectionPolicy = PDFMINER_POLICY,
    products: CapturedProgram | None = None,
) -> LTPage:
    page_media_box = pdfminer_media_box(page)
    page_width = abs(page_media_box[2] - page_media_box[0])
    page_height = abs(page_media_box[3] - page_media_box[1])
    if policy.strict_resources:
        pdfminer_validate_page_resources(page)
    if products is None:
        products = pdfminer_page_program(page, strict_cmaps=policy.strict_resources)
    projected_glyphs: tuple[Any, ...] = products.glyphs
    ligatures, skipped_ligature_parts = pdfminer_ligature_overrides(projected_glyphs)
    chars_by_glyph: dict[int, LTChar] = {}
    glyph_matrices: dict[int, Any] = getattr(products, "glyph_matrices", {})
    vertical_positions: dict[tuple[str | None, int], tuple[float, int]] = {}
    provenance_dicts: dict[int, tuple[object, dict[str, Any]]] = {}
    # A page repeats a few hundred distinct glyphs; their pdfminer text, widths and
    # descent depend only on the font decoder and the glyph fields in these keys.
    glyph_texts: dict[tuple[Any, ...], str] = {}
    glyph_widths: dict[tuple[Any, ...], tuple[float | None, float]] = {}
    descents: dict[int, tuple[Any, float]] = {}
    for glyph_index, glyph in enumerate(projected_glyphs):
        if policy.strict_resources and pdfminer_embedded_cmap_is_unusable(glyph):
            continue
        source_provenance = glyph.provenance
        if source_provenance:
            known = provenance_dicts.get(id(source_provenance))
            if known is None or known[0] is not source_provenance:
                known = provenance_dicts[id(source_provenance)] = (
                    source_provenance,
                    dict(source_provenance),
                )
            glyph_provenance = known[1]
        else:
            glyph_provenance = {}
        if _pdfminer_form_glyph_is_clipped(glyph_provenance):
            continue
        if id(glyph) in skipped_ligature_parts:
            continue
        if not glyph.text:
            continue
        ligature = ligatures.get(id(glyph))
        x0, y0, x1, y1 = ligature[1] if ligature is not None else glyph.advance_bbox
        baseline = ligature[2] if ligature is not None else glyph.baseline
        glyph_key = (
            glyph.font_decoder,
            glyph.char_code,
            glyph.cid,
            glyph.code_bytes,
            glyph.unicode_source,
            glyph.text,
            glyph.alternates,
        )
        glyph_text: str | None = None
        if ligature is None:
            glyph_text = glyph_texts.get(glyph_key)
            if glyph_text is None:
                glyph_text = glyph_texts[glyph_key] = pdfminer_glyph_text(glyph)
        text = (ligature[0] if ligature is not None else glyph_text) or ""
        if not text and glyph.text != EMPTY_GLYPH_TEXT:
            continue
        effective_font_size = glyph.effective_font_size or glyph.font_size
        effective_font_height = glyph.effective_font_height or effective_font_size
        if (
            glyph.baseline is not None
            and glyph.font_size > 0
            and not glyph.effective_font_size
            and x1 - x0 >= glyph.font_size * 0.8
            and y1 - y0 <= glyph.font_size * 0.1
        ):
            baseline_x = glyph.baseline[0]
            key = (glyph.font_name, round(baseline_x))
            anchor, position = vertical_positions.get(key, (glyph.baseline[1], 0))
            baseline_y = anchor - position * glyph.font_size
            vertical_positions[key] = (anchor, position + 1)
            x0 = baseline_x - glyph.font_size * 0.5
            x1 = baseline_x + glyph.font_size * 0.5
            y0 = baseline_y - glyph.font_size * 0.88
            y1 = y0 + glyph.font_size
        width_code = (
            glyph.cid
            if getattr(glyph.font_decoder, "is_cid_font", False)
            else glyph.char_code
            if glyph.char_code is not None
            else glyph.cid
        )
        widths_key = (*glyph_key, glyph_text is None)
        known_widths = glyph_widths.get(widths_key)
        if known_widths is None:
            known_widths = glyph_widths[widths_key] = pdfminer_projected_widths(
                glyph, width_code, glyph_text
            )
        font_width, normalized_width = known_widths
        if (
            glyph.rotation_angle % 180 == 0
            and baseline is not None
            and not glyph.effective_font_size
            and font_width is not None
            and font_width > 0
        ):
            baseline_x0, baseline_y0, baseline_x1, baseline_y1 = baseline
            baseline_length = (
                (baseline_x1 - baseline_x0) ** 2 + (baseline_y1 - baseline_y0) ** 2
            ) ** 0.5
            effective_font_size = baseline_length / font_width
        orientation = glyph.rotation_angle % 360
        text_matrix = glyph_provenance.get("text_matrix")
        if isinstance(text_matrix, (tuple, list)) and len(text_matrix) == 6:
            resolved_text_matrix = text_matrix
        elif isinstance(text_matrix, (tuple, list)) and len(text_matrix) == 4:
            resolved_text_matrix = (*text_matrix, 0.0, 0.0)
        else:
            resolved_text_matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        horizontal_scale = float(glyph_provenance.get("horizontal_scale", 100.0)) * 0.01
        coordinates_in_layout_space = False
        if (
            getattr(glyph.font_decoder, "is_vertical", False)
            and baseline is not None
            and width_code is not None
        ):
            metric = glyph.font_decoder.vertical_glyph_metric(width_code)
            origin_x, origin_y = _pdfminer_layout_origin(
                baseline,
                normalize_noise=effective_font_size == glyph.font_size,
            )
            matrix_a, matrix_b, matrix_c, matrix_d, _matrix_e, _matrix_f = (
                float(value) for value in resolved_text_matrix
            )
            local_font_size = glyph.font_size
            local_left = -float(metric[1]) * local_font_size * 0.001
            # The engine's baseline already sits at the vertical origin, so the glyph's
            # top is one font size above it.
            local_top = local_font_size
            local_advance = float(metric[0]) * local_font_size * 0.001 * horizontal_scale
            corners = tuple(
                (
                    local_horizontal * matrix_a + local_vertical * matrix_c + origin_x,
                    local_horizontal * matrix_b + local_vertical * matrix_d + origin_y,
                )
                for local_horizontal in (local_left, local_left + local_font_size)
                for local_vertical in (local_top + local_advance, local_top)
            )
            exact_matrix = glyph_matrices.get(glyph_index)
            if exact_matrix is not None:
                # pdfminer's own arithmetic, from LTChar: its rounding decides which boxes
                # touch, and so which lines group.
                char_a, char_b, char_c, char_d, char_e, char_f = exact_matrix
                rise = float(glyph_provenance.get("text_rise", 0.0) or 0.0)
                font_size = glyph.font_size
                vertical_metrics = getattr(glyph.font_decoder, "vertical_metrics", {})
                # pdfminer centres a glyph W2 gives no displacement for at half the size.
                vx = (
                    float(metric[1]) * font_size * 0.001
                    if width_code in vertical_metrics
                    else font_size * 0.5
                )
                vy = (1000 - float(metric[2])) * font_size * 0.001
                advance = float(metric[0]) * 0.001 * font_size * horizontal_scale
                corners = tuple(
                    (
                        char_a * local_x + char_c * local_y + char_e,
                        char_b * local_x + char_d * local_y + char_f,
                    )
                    for local_x, local_y in (
                        (-vx, vy + rise + advance),
                        (-vx + font_size, vy + rise + advance),
                        (-vx + font_size, vy + rise),
                        (-vx, vy + rise),
                    )
                )
                x0 = min(point[0] for point in corners)
                y0 = min(point[1] for point in corners)
                x1 = max(point[0] for point in corners)
                y1 = max(point[1] for point in corners)
                coordinates_in_layout_space = True
            else:
                # Rounded so that glyphs one half-size apart do not count as overlapping by
                # a fraction of the last float bit, which merges them into a line.
                x0 = round(min(point[0] for point in corners), 9)
                y0 = round(min(point[1] for point in corners), 9)
                x1 = round(max(point[0] for point in corners), 9)
                y1 = round(max(point[1] for point in corners), 9)
            effective_font_height = x1 - x0
        elif (
            baseline is not None
            and isinstance(text_matrix, (tuple, list))
            and len(text_matrix) == 4
        ):
            matrix_a, matrix_b, matrix_c, matrix_d = map(float, text_matrix)
            origin_x, origin_y = _pdfminer_layout_origin(
                baseline,
                normalize_noise=(
                    bool(glyph_index) and projected_glyphs[glyph_index - 1].font_decoder.is_vertical
                ),
            )
            decoder = glyph.font_decoder
            known_descent = descents.get(id(decoder))
            if known_descent is None or known_descent[0] is not decoder:
                known_descent = descents[id(decoder)] = (decoder, pdfminer_descent(glyph))
            # The engine's baseline already includes text rise.
            descent = known_descent[1] * glyph.font_size
            top = descent + glyph.font_size
            advance = normalized_width * horizontal_scale * glyph.font_size
            exact_matrix = glyph_matrices.get(glyph_index)
            if exact_matrix is not None:
                # pdfminer's own arithmetic, from LTChar: its rounding decides which boxes
                # touch, and so which lines group.
                char_a, char_b, char_c, char_d, char_e, char_f = exact_matrix
                rise = float(glyph_provenance.get("text_rise", 0.0) or 0.0)
                font_size = glyph.font_size
                exact_advance = normalized_width * font_size * horizontal_scale
                low = descent + rise
                high = descent + rise + font_size
                xs = (
                    char_a * 0 + char_c * low + char_e,
                    char_a * exact_advance + char_c * low + char_e,
                    char_a * exact_advance + char_c * high + char_e,
                    char_a * 0 + char_c * high + char_e,
                )
                ys = (
                    char_b * 0 + char_d * low + char_f,
                    char_b * exact_advance + char_d * low + char_f,
                    char_b * exact_advance + char_d * high + char_f,
                    char_b * 0 + char_d * high + char_f,
                )
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            else:
                media_left, media_bottom, _media_right, _media_top = page_media_box
                page_rotation = int(page.rotation) % 360
                layout_origin_x = origin_x - media_left
                layout_origin_y = origin_y - media_bottom
                (
                    layout_origin_x,
                    layout_origin_y,
                    matrix_a,
                    matrix_b,
                    matrix_c,
                    matrix_d,
                ) = _pdfminer_rotated_text_matrix(
                    layout_origin_x,
                    layout_origin_y,
                    (matrix_a, matrix_b, matrix_c, matrix_d),
                    page_rotation,
                    page_width,
                    page_height,
                )
                bottom_left_x = 0.0 * matrix_a + descent * matrix_c + layout_origin_x
                bottom_left_y = 0.0 * matrix_b + descent * matrix_d + layout_origin_y
                top_left_x = 0.0 * matrix_a + top * matrix_c + layout_origin_x
                top_left_y = 0.0 * matrix_b + top * matrix_d + layout_origin_y
                bottom_right_x = advance * matrix_a + descent * matrix_c + layout_origin_x
                bottom_right_y = advance * matrix_b + descent * matrix_d + layout_origin_y
                top_right_x = advance * matrix_a + top * matrix_c + layout_origin_x
                top_right_y = advance * matrix_b + top * matrix_d + layout_origin_y
                x0 = min(bottom_left_x, top_left_x, bottom_right_x, top_right_x)
                y0 = min(bottom_left_y, top_left_y, bottom_right_y, top_right_y)
                x1 = max(bottom_left_x, top_left_x, bottom_right_x, top_right_x)
                y1 = max(bottom_left_y, top_left_y, bottom_right_y, top_right_y)
            effective_font_height = x1 - x0 if orientation % 180 else y1 - y0
            coordinates_in_layout_space = True
        elif orientation == 0:
            if normalized_width > 0:
                x1 = x0 + normalized_width * effective_font_size
            y1 = y0 + effective_font_height
        elif orientation == 90:
            x0 = x1 - effective_font_height
            if normalized_width > 0:
                y1 = y0 + normalized_width * effective_font_size
        elif orientation == 180:
            if normalized_width > 0:
                x0 = x1 - normalized_width * effective_font_size
            y0 = y1 - effective_font_height
        elif orientation == 270:
            x1 = x0 + effective_font_height
            if normalized_width > 0:
                y0 = y1 - normalized_width * effective_font_size
        if not coordinates_in_layout_space:
            x0, y0, x1, y1 = _pdfminer_layout_figure_box(
                (x0, y0, x1, y1),
                page_media_box,
                int(page.rotation) % 360,
                page_width,
                page_height,
            )
        character = LTChar(
            (x0, y0, x1, y1),
            text,
            pdfminer_font_name(glyph),
            # pdfminer's size is the box's height, or its width for vertical writing.
            x1 - x0 if getattr(glyph.font_decoder, "is_vertical", False) else y1 - y0,
        )
        chars_by_glyph[glyph_index] = character
    events = getattr(products, "layout_events", ())
    painted_paths = getattr(products, "painted_paths", ())
    rotation = int(page.rotation) % 360
    layout_width, layout_height = (
        (page_height, page_width) if rotation % 180 else (page_width, page_height)
    )
    media_left, media_bottom = page_media_box[0], page_media_box[1]

    def layout_point(x: float, y: float) -> tuple[float, float]:
        x -= media_left
        y -= media_bottom
        if rotation == 90:
            return y, page_width - x
        if rotation == 180:
            return page_width - x, page_height - y
        if rotation == 270:
            return page_height - y, x
        return x, y

    def layout_box(
        points: list[tuple[float, float]],
    ) -> tuple[float, float, float, float]:
        placed = [layout_point(x, y) for x, y in points]
        return (
            min(point[0] for point in placed),
            min(point[1] for point in placed),
            max(point[0] for point in placed),
            max(point[1] for point in placed),
        )

    page_items: list[Any] = []
    current = page_items
    stack: list[tuple[list[Any], LTFigure]] = []
    figure_count = 0
    if not events:
        # Without the layout walk, glyphs are all the page has.
        page_items.extend(chars_by_glyph.values())
    for event in events:
        kind = event[0]
        if kind == "glyphs":
            for glyph_index in range(event[1], event[2]):
                placed_char = chars_by_glyph.get(glyph_index)
                if placed_char is not None:
                    current.append(placed_char)
        elif kind == "path":
            current.extend(pdfminer_path_items(painted_paths[event[1]], layout_point))
        elif kind == "begin":
            x0, y0, x1, y1 = event[2]
            figure = LTFigure(
                layout_box([(x0, y0), (x1, y0), (x0, y1), (x1, y1)]),
                event[1] or f"Form{figure_count}",
            )
            figure_count += 1
            stack.append((current, figure))
            current = figure._objs
        elif kind == "end" and stack:
            parent, figure = stack.pop()
            parent.append(figure)
            current = parent
        elif kind == "image":
            matrix = event[2]
            box = layout_box(
                [
                    (matrix.e, matrix.f),
                    (matrix.a + matrix.e, matrix.b + matrix.f),
                    (matrix.c + matrix.e, matrix.d + matrix.f),
                    (matrix.a + matrix.c + matrix.e, matrix.b + matrix.d + matrix.f),
                ]
            )
            current.append(LTFigure(box, event[1], [LTImage(box, event[1], event[3])]))
    while stack:
        parent, figure = stack.pop()
        parent.append(figure)
    page_box = (0.0, 0.0, layout_width, layout_height)
    return LTPage(
        page_box,
        page.page_number,
        page.rotation,
        page_items if params is None else analyze_layout(page_items, params, page_box),
    )


def analyze_layout(
    items: list[Any], params: LAParams, bbox: tuple[float, float, float, float]
) -> list[Any]:
    """A container's children after pdfminer's layout analysis.

    Follows pdfminer.six's LTLayoutContainer.analyze, Copyright (c) 2004-2016 Yusuke
    Shinyama, under the MIT License.
    """
    chars = [item for item in items if isinstance(item, LTChar)]
    others = [item for item in items if not isinstance(item, LTChar)]
    if params.all_texts:
        for item in others:
            if isinstance(item, LTFigure):
                item._objs = analyze_layout(item._objs, params, item.bbox)
    if not chars:
        return items
    lines = _group_objects(chars, params)
    empties = [line for line in lines if _line_is_empty(line)]
    boxes = _reading_order(
        _group_lines(
            [line for line in lines if not _line_is_empty(line)], params.line_margin, bbox
        ),
        params.boxes_flow,
        bbox,
    )
    return [*boxes, *others, *empties]


def _line_is_empty(line: LTTextLine) -> bool:
    # pdfminer asks before it appends the line's closing newline, so a line of characters
    # that map to no text is not empty.
    text = "".join(item.get_text() for item in line._objs[:-1])
    return line.width <= 0 or line.height <= 0 or text.isspace()


def pdfminer_path_items(
    path: Any, layout_point: Callable[[float, float], tuple[float, float]]
) -> list[Any]:
    """The LTLine, LTRect and LTCurve objects pdfminer makes of one painted path.

    Ported from pdfminer.six's PDFLayoutAnalyzer.paint_path, Copyright (c) 2004-2016
    Yusuke Shinyama, under the MIT License.
    """
    matrix = path.ctm
    coords = path.coords

    def point(x: float, y: float) -> tuple[float, float]:
        return layout_point(
            matrix.a * x + matrix.c * y + matrix.e, matrix.b * x + matrix.d * y + matrix.f
        )

    segments: list[tuple[Any, ...]] = []
    position = 0
    for op in path.ops:
        if op == PATH_CLOSE:
            segments.append(("h",))
        elif op == PATH_RECT:
            x, y, width, height = coords[position : position + 4]
            segments += [
                ("m", point(x, y)),
                ("l", point(x + width, y)),
                ("l", point(x + width, y + height)),
                ("l", point(x, y + height)),
                ("h",),
            ]
        elif op == PATH_CURVE:
            _, _, x1, y1, x2, y2, x3, y3 = coords[position : position + 8]
            segments.append(("c", point(x1, y1), point(x2, y2), point(x3, y3)))
        else:
            segments.append((chr(op), point(coords[position], coords[position + 1])))
        position += PATH_OPERAND_COUNTS[op]
    shape = "".join(segment[0] for segment in segments)
    if not shape.startswith("m"):
        return []
    if shape.count("m") > 1:
        subpaths = [
            segments[match.start() : match.end()] for match in re.finditer(r"m[^m]+", shape)
        ]
    else:
        subpaths = [segments]
    stroke = path.kind != "fill"
    fill = path.kind != "stroke"
    evenodd = path.fill_rule == "evenodd"
    style = (stroke, fill, evenodd, path.stroke_color, path.fill_color)
    items: list[Any] = []
    for subpath in subpaths:
        sub_shape = "".join(segment[0] for segment in subpath)
        points = [subpath[0][-1] if segment[0] == "h" else segment[-1] for segment in subpath]
        original = [(segment[0], *segment[1:]) for segment in subpath]
        if len(sub_shape) > 3 and sub_shape.endswith("lh") and points[-2] == points[0]:
            sub_shape = f"{sub_shape[:-2]}h"
            points.pop()
        if sub_shape in {"mlh", "ml"}:
            items.append(LTLine(path.line_width, points[0], points[1], *style, original, path.dash))
            continue
        if sub_shape in {"mlllh", "mllll"}:
            (x0, y0), (x1, y1), (x2, y2), (x3, y3), _last = points
            if points[0] == points[4] and (
                (x0 == x1 and y1 == y2 and x2 == x3 and y3 == y0)
                or (y0 == y1 and x1 == x2 and y2 == y3 and x3 == x0)
            ):
                items.append(LTRect(path.line_width, (x0, y0, x2, y2), *style, original, path.dash))
                continue
        items.append(LTCurve(path.line_width, points, *style, original, path.dash))
    return items
