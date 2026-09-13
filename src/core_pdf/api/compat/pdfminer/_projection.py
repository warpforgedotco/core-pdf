"""Project captured page evidence into PDFMiner layout objects."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Any, cast

from core_pdf import PdfPage
from core_pdf.impl._impl.model.geometry import bbox_union, overlap_ratio_of
from core_pdf.impl.exceptions import PdfError

from ._capture import (
    internal_pdfminer_page_program,
    internal_pdfminer_validate_page_resources,
)
from ._fonts import (
    _font_value,
    _pdfminer_builtin_width,
    internal_pdfminer_descent,
    internal_pdfminer_embedded_cmap_is_unusable,
    internal_pdfminer_font_name,
    internal_pdfminer_glyph_text,
    internal_pdfminer_ligature_overrides,
)
from ._layout import (
    LAParams,
    LTChar,
    LTFigure,
    LTItem,
    LTPage,
    _group_lines,
    _group_objects,
    _reading_order,
)


def _pdfminer_form_glyph_is_clipped(provenance: dict[str, Any]) -> bool:
    if int(provenance.get("xobject_depth", 0) or 0) <= 0:
        return False
    clip = provenance.get("clip_bbox")
    if not isinstance(clip, (tuple, list)) or len(clip) != 4:
        return False
    left, bottom, right, top = (float(value) for value in clip)
    # PDFMiner ignores page-content clipping during layout, but form
    # traversal still rejects a form whose transformed bounds are empty.
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


def internal_project_page(
    page: PdfPage,
    params: LAParams,
    *,
    unstructured_mode: bool = False,
) -> LTPage:
    """Project one already-open page using PDFMiner's geometry and layout policies."""
    page_width = abs(page.width)
    page_height = abs(page.height)
    page_media_box = page.media_box or (0.0, 0.0, page_width, page_height)
    chars: list[LTChar] = []
    if not unstructured_mode:
        internal_pdfminer_validate_page_resources(page)
    products = internal_pdfminer_page_program(page)
    projected_glyphs: tuple[Any, ...] = products.glyphs
    ligatures, skipped_ligature_parts = internal_pdfminer_ligature_overrides(projected_glyphs)
    runs = sorted(products.runs, key=lambda run: run.seqno)
    run_sequences = [run.seqno for run in runs]
    figure_chars: dict[tuple[object, ...], list[tuple[LTChar, int]]] = {}
    figure_boxes: dict[tuple[object, ...], tuple[float, float, float, float]] = {}
    figure_depths: dict[tuple[object, ...], int] = {}
    figure_identifiers: dict[tuple[object, ...], object] = {}
    form_ancestor_boxes: dict[tuple[object, ...], tuple[float, float, float, float]] = {}
    try:
        page_annotations = page.get_annotations()
    except (PdfError, ValueError):
        page_annotations = []
    annotation_boxes = tuple(
        tuple(annotation.rect) for annotation in page_annotations if annotation.rect is not None
    )
    vertical_positions: dict[tuple[str | None, int], tuple[float, int]] = {}
    for glyph_index, glyph in enumerate(projected_glyphs):
        if not unstructured_mode and internal_pdfminer_embedded_cmap_is_unusable(glyph):
            continue
        glyph_provenance = dict(glyph.provenance) if glyph.provenance else {}
        if _pdfminer_form_glyph_is_clipped(glyph_provenance):
            continue
        run_index = bisect_right(run_sequences, glyph.seqno) - 1
        if id(glyph) in skipped_ligature_parts:
            continue
        if not glyph.text:
            continue
        ligature = ligatures.get(id(glyph))
        x0, y0, x1, y1 = ligature[1] if ligature is not None else glyph.advance_bbox
        baseline = ligature[2] if ligature is not None else glyph.baseline
        text = ligature[0] if ligature is not None else internal_pdfminer_glyph_text(glyph)
        if not text:
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
        # PDF text size precedes the text matrix, so it can be much larger than the
        # effective glyph size after horizontal scaling. Recover the transformed size
        # from core's baseline advance; the advance box already has pdfminer's x bounds.
        width_code = (
            glyph.cid
            if getattr(glyph.font_decoder, "is_cid_font", False)
            else glyph.char_code
            if glyph.char_code is not None
            else glyph.cid
        )
        width_lookup = getattr(glyph.font_decoder, "glyph_width", None)
        if (
            glyph.rotation_angle % 180 == 0
            and baseline is not None
            and not glyph.effective_font_size
            and width_code is not None
            and callable(width_lookup)
        ):
            normalized_width = float(width_lookup(width_code)) * 0.001
            if normalized_width > 0:
                baseline_x0, baseline_y0, baseline_x1, baseline_y1 = baseline
                baseline_length = (
                    (baseline_x1 - baseline_x0) ** 2 + (baseline_y1 - baseline_y0) ** 2
                ) ** 0.5
                effective_font_size = baseline_length / normalized_width
        normalized_width = 0.0
        if width_code is not None and callable(width_lookup):
            normalized_width = float(width_lookup(width_code)) * 0.001
            builtin_width = _pdfminer_builtin_width(glyph)
            if builtin_width is not None:
                normalized_width = builtin_width * 0.001
            base_font = str(_font_value(glyph.font_decoder.font, "BaseFont"))
            glyph_name = getattr(glyph.font_decoder, "encoding_differences", {}).get(
                glyph.char_code
            )
            if (
                glyph_name
                and _font_value(glyph.font_decoder.font, "Widths") is None
                and base_font.split("+")[-1] in {"Symbol", "ZapfDingbats"}
            ):
                # pdfminer indexes built-in Symbol/Zapf metrics by its
                # legacy encoded character keys. A Differences entry
                # resolves to Unicode and therefore has no built-in
                # width unless /Widths explicitly supplies one.
                normalized_width = 0.0
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
            # The engine advances an entire text-show array in bulk,
            # while PDFMiner advances one token at a time. Normalize
            # the resulting sub-picopoint accumulation noise before
            # applying vertical displacement metrics; otherwise two
            # mathematically touching boxes can miss by one ULP.
            origin_x, origin_y = _pdfminer_layout_origin(
                baseline,
                normalize_noise=effective_font_size == glyph.font_size,
            )
            # PDFMiner applies the vertical displacement and advance
            # as a local LTChar rectangle before transforming it.  In
            # particular, the character origin is not the lower edge:
            # a normal vertical advance extends down from ``v1y``.
            matrix_a, matrix_b, matrix_c, matrix_d, _matrix_e, _matrix_f = (
                float(value) for value in resolved_text_matrix
            )
            local_font_size = glyph.font_size
            local_left = -float(metric[1]) * local_font_size * 0.001
            local_top = (1000.0 - float(metric[2])) * local_font_size * 0.001 + float(
                glyph_provenance.get("text_rise", 0.0)
            )
            local_advance = float(metric[0]) * local_font_size * 0.001 * horizontal_scale
            corners = tuple(
                (
                    local_horizontal * matrix_a + local_vertical * matrix_c + origin_x,
                    local_horizontal * matrix_b + local_vertical * matrix_d + origin_y,
                )
                for local_horizontal in (local_left, local_left + local_font_size)
                for local_vertical in (local_top + local_advance, local_top)
            )
            x0 = min(point[0] for point in corners)
            y0 = min(point[1] for point in corners)
            x1 = max(point[0] for point in corners)
            y1 = max(point[1] for point in corners)
            effective_font_height = x1 - x0
            line_origin = glyph_provenance.get("line_matrix_origin")
            if isinstance(line_origin, (tuple, list)) and len(line_origin) == 2:
                matrix_a, matrix_b, matrix_c, matrix_d, _matrix_e, _matrix_f = (
                    float(value) for value in resolved_text_matrix
                )
                determinant = matrix_a * matrix_d - matrix_b * matrix_c
                if determinant:
                    translate_x, translate_y = (float(value) for value in line_origin)
                    baseline_x0, baseline_y0, baseline_x1, baseline_y1 = baseline
                    advance_x = baseline_x1 - baseline_x0
                    advance_y = baseline_y1 - baseline_y0
                    local_advance = (-matrix_b * advance_x + matrix_a * advance_y) / determinant
                    half_width = float(metric[1]) * glyph.font_size * 0.001
                    local_top = (1000.0 - float(metric[2])) * glyph.font_size * 0.001 + float(
                        glyph_provenance.get("text_rise", 0.0)
                    )
                    corners = tuple(
                        (
                            matrix_a * local_horizontal + matrix_c * local_vertical + translate_x,
                            matrix_b * local_horizontal + matrix_d * local_vertical + translate_y,
                        )
                        for local_horizontal in (
                            -half_width,
                            -half_width + glyph.font_size,
                        )
                        for local_vertical in (local_top + local_advance, local_top)
                    )
                    x0 = min(point[0] for point in corners)
                    y0 = min(point[1] for point in corners)
                    x1 = max(point[0] for point in corners)
                    y1 = max(point[1] for point in corners)
        elif (
            baseline is not None
            and isinstance(text_matrix, (tuple, list))
            and len(text_matrix) == 4
        ):
            matrix_a, matrix_b, matrix_c, matrix_d = (float(value) for value in text_matrix)
            # A horizontal show immediately following vertical writing
            # resumes at the vertical cursor. Normalize only that
            # hand-off; ordinary horizontal origins must retain
            # PDFMiner's native floating-point arithmetic.
            origin_x, origin_y = _pdfminer_layout_origin(
                baseline,
                normalize_noise=(
                    bool(glyph_index) and projected_glyphs[glyph_index - 1].font_decoder.is_vertical
                ),
            )
            text_rise = float(glyph_provenance.get("text_rise", 0.0))
            descent = internal_pdfminer_descent(glyph) * glyph.font_size + text_rise
            # ``LTChar`` uses the font descent only to anchor horizontal
            # glyphs; its box is always exactly one text-space unit tall.
            # FontBBox/ascent describes ink, not pdfminer's layout box.
            top = descent + glyph.font_size
            advance = normalized_width * horizontal_scale * glyph.font_size
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
            # Compute the four corners directly, preserving evaluation order
            # and zero products while avoiding temporary tuples and generators.
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
        # PDFMiner places the media-box lower-left at layout-space
        # (0, 0). Core's canonical geometry remains in PDF user space,
        # so normalize non-zero and negative media-box origins here.
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
            internal_pdfminer_font_name(glyph),
            effective_font_height,
        )
        provenance = (
            glyph_provenance
            if glyph.provenance
            else (dict(runs[run_index].provenance) if run_index >= 0 else {})
        )
        raw_xobject_depth = provenance.get("xobject_depth")
        xobject_depth = raw_xobject_depth if type(raw_xobject_depth) is int else 0
        if xobject_depth <= 0:
            chars.append(character)
            continue
        clip_bbox = provenance.get("clip_bbox")
        if isinstance(clip_bbox, (tuple, list)) and len(clip_bbox) == 4:
            clip = tuple(float(cast(Any, value)) for value in clip_bbox)
            if any(overlap_ratio_of(clip, annotation) > 0.5 for annotation in annotation_boxes):
                continue
        raw_stream_order = provenance.get("stream_order")
        stream_order = raw_stream_order if type(raw_stream_order) is int else 0
        layout_form_id = provenance.get("layout_form_id")
        layout_bbox = provenance.get("layout_form_bbox")
        if isinstance(layout_form_id, tuple):
            for ancestor_index, ancestor_entry in enumerate(layout_form_id, start=1):
                if (
                    isinstance(ancestor_entry, tuple)
                    and len(ancestor_entry) == 2
                    and isinstance(ancestor_entry[1], (tuple, list))
                    and len(ancestor_entry[1]) == 4
                ):
                    form_ancestor_boxes[layout_form_id[:ancestor_index]] = cast(
                        tuple[float, float, float, float],
                        tuple(float(cast(Any, value)) for value in ancestor_entry[1]),
                    )
        if isinstance(layout_bbox, (tuple, list)) and len(layout_bbox) == 4:
            resolved_layout_bbox = cast(
                tuple[float, float, float, float],
                tuple(float(cast(Any, value)) for value in layout_bbox),
            )
            figure_key: tuple[object, ...] = (
                "form",
                layout_form_id if layout_form_id is not None else stream_order,
                *resolved_layout_bbox,
            )
            figure_identifiers[figure_key] = layout_form_id
            figure_boxes[figure_key] = resolved_layout_bbox
        else:
            figure_key = ("stream", stream_order)
            layout_bbox = provenance.get("clip_bbox")
        figure_chars.setdefault(figure_key, []).append((character, glyph.seqno))
        figure_depths[figure_key] = xobject_depth
        if isinstance(layout_bbox, (tuple, list)) and len(layout_bbox) == 4:
            figure_boxes[figure_key] = cast(
                tuple[float, float, float, float],
                tuple(float(cast(Any, value)) for value in layout_bbox),
            )
    lines = _group_objects(chars, params)
    empty_lines = [line for line in lines if line.get_text().isspace()]
    layout_width, layout_height = (
        (page_height, page_width) if int(page.rotation) % 180 else (page_width, page_height)
    )
    boxes: list[LTItem] = list(
        _reading_order(
            _group_lines(
                [line for line in lines if not line.get_text().isspace()],
                params.line_margin,
                (0.0, 0.0, layout_width, layout_height),
            ),
            params.boxes_flow,
            (0.0, 0.0, layout_width, layout_height),
        )
    )
    boxes.extend(empty_lines)
    # PDFMiner inserts layout children only for marks that reach its
    # device. Graphics-state and clipping records are engine
    # provenance, not LTItems, and therefore cannot delimit figure
    # text during recursive extraction.
    drawing_sequences_by_depth: dict[int, list[int]] = {}
    for drawing in products.drawings:
        if drawing.kind in {
            "clip",
            "group-begin",
            "group-end",
            "scope-begin",
            "scope-end",
            "state-push",
            "state-pop",
        }:
            continue
        drawing_sequences_by_depth.setdefault(drawing.xobject_depth, []).append(drawing.seqno)
    for sequences in drawing_sequences_by_depth.values():
        sequences.sort()
    figures: list[tuple[LTFigure, int, object]] = []

    for figure_index, (figure_key, entries) in enumerate(figure_chars.items()):
        drawing_sequences = drawing_sequences_by_depth.get(figure_depths[figure_key], [])
        snippets: list[str] = []
        current: list[str] = []
        previous_sequence: int | None = None
        for character, sequence in entries:
            if previous_sequence is not None and bisect_left(
                drawing_sequences, sequence
            ) > bisect_right(drawing_sequences, previous_sequence):
                snippets.append("".join(current))
                current = []
            current.append(character.get_text())
            previous_sequence = sequence
        if current:
            snippets.append("".join(current))
        merged_snippets: list[str] = []
        for snippet in snippets:
            if (
                merged_snippets
                and snippet
                and not any(character.isalnum() for character in snippet)
                and not any(character.isalnum() for character in merged_snippets[-1])
                and merged_snippets[-1][-1] == snippet[0]
            ):
                merged_snippets[-1] += snippet
            else:
                merged_snippets.append(snippet)
        snippets = merged_snippets
        figure_box = figure_boxes.get(figure_key) or bbox_union(
            character.bbox for character, _ in entries
        )
        if figure_box is not None:
            x0, y0, x1, y1 = _pdfminer_layout_figure_box(
                figure_box,
                page_media_box,
                int(page.rotation) % 360,
                page_width,
                page_height,
            )
            figures.append(
                (
                    LTFigure(
                        (x0, y0, x1, y1),
                        f"Form{figure_index}",
                        [character for character, _ in entries],
                        tuple(snippets),
                    ),
                    figure_depths[figure_key],
                    figure_identifiers.get(figure_key),
                )
            )
    represented_identifiers = {identifier for _figure, _depth, identifier in figures}
    for identifier, ancestor_box in form_ancestor_boxes.items():
        if identifier in represented_identifiers:
            continue
        figures.append(
            (
                LTFigure(
                    _pdfminer_layout_figure_box(
                        ancestor_box,
                        page_media_box,
                        int(page.rotation) % 360,
                        page_width,
                        page_height,
                    ),
                    f"Form{len(figures)}",
                    [],
                    (),
                ),
                len(identifier),
                identifier,
            )
        )
    for figure, depth, figure_identifier in figures:
        parent_identifier = (
            figure_identifier[:-1]
            if isinstance(figure_identifier, tuple) and len(figure_identifier) > 1
            else None
        )
        parent = min(
            (
                candidate
                for candidate, candidate_depth, candidate_identifier in figures
                if candidate_depth == depth - 1
                and (
                    candidate_identifier == parent_identifier
                    if parent_identifier is not None
                    else candidate.x0 <= figure.x0
                    and candidate.y0 <= figure.y0
                    and candidate.x1 >= figure.x1
                    and candidate.y1 >= figure.y1
                )
            ),
            key=lambda candidate: candidate.width * candidate.height,
            default=None,
        )
        if parent is None:
            boxes.append(figure)
        else:
            parent._objs.append(figure)
    return LTPage(
        (0.0, 0.0, layout_width, layout_height),
        page.page_number,
        page.rotation,
        boxes,
    )
