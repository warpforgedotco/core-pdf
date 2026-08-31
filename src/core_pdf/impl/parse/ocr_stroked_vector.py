# SPDX-License-Identifier: AGPL-3.0-only
"""Recovering text from stroked vector outlines.

Some documents draw their text as vector strokes rather than glyphs, so there is
nothing for a font decoder to read. This module rasterizes those strokes, packs the
isolated runs into a compact sheet Tesseract can recognize in one pass, and maps the
recognized characters back onto the original runs.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy

from core_pdf.impl.layout.spatial import SpatialIndex
from core_pdf.impl.model.geometry import bbox_intersection_area, rect_tuple
from core_pdf.impl.parse.model import (
    MAX_OCR_PIXELS,
    CapturedPage,
    ObservationBatch,
    ObservationSource,
    RecognitionReport,
    internal_bbox_tuple,
    internal_Candidate,
    internal_candidate,
)
from core_pdf.impl.parse.ocr_model import (
    internal_PackedStrokedTextRaster,
    internal_Raster,
    internal_RasterRegion,
    internal_StrokedTextCell,
)
from core_pdf.impl.parse.stroked_text import (
    StrokedTextDecode,
    StrokedTextObservation,
    StrokedTextProfile,
    StrokedTextRun,
    StrokedTextSeed,
    decode_stroked_text_profile,
    decode_stroked_text_profile_with_supplemental_seeds,
    profile_stroked_text,
    stroked_text_isolated_runs,
)
from core_pdf.impl.render.display import (
    DisplayList,
    PathPaintItem,
    PathPaintKind,
)
from core_pdf.impl.render.kernels import rasterize_packed_stroked_paths
from core_pdf.impl.render.page import RenderedPage
from core_pdf.impl.runtime.array_views import finite_median

STROKED_VECTOR_PACK_WIDTH = 240.0


STROKED_VECTOR_PACK_HORIZONTAL_PADDING = 4.0


STROKED_VECTOR_PACK_VERTICAL_PADDING = 4.0


STROKED_VECTOR_PACK_DENSE_VERTICAL_PADDING = 2.0


STROKED_VECTOR_PACK_DENSE_MIN_CELLS = 96


STROKED_VECTOR_PACK_REMAP_TOLERANCE = 4.0


STROKED_VECTOR_PACK_MIN_ALIGNED_SEEDS = 12


STROKED_VECTOR_PACK_MIN_LEARNED_SIGNATURES = 16


STROKED_VECTOR_PACK_MIN_DECODED_RUNS = 16


@dataclass(frozen=True, slots=True)
class internal_CachedStrokedTextProfile:
    drawings: tuple[Any, ...]
    drawing_indexes: tuple[int, ...]
    profile: StrokedTextProfile


def internal_stroked_text_profile(capture: CapturedPage) -> StrokedTextProfile:
    """Return the single structural glyph profile shared by OCR and document reuse."""
    evidence = capture.evidence.stroked_vector_text
    cache = capture.page.extraction_cache
    cached = cache.get("_stroked_text_profile")
    if (
        isinstance(cached, internal_CachedStrokedTextProfile)
        and cached.drawings is capture.drawings
        and cached.drawing_indexes == evidence.drawing_indexes
    ):
        return cached.profile
    profile = profile_stroked_text(capture.drawings, evidence.drawing_indexes)
    cache["_stroked_text_profile"] = internal_CachedStrokedTextProfile(
        capture.drawings,
        evidence.drawing_indexes,
        profile,
    )
    return profile


def internal_pack_stroked_text_runs(
    runs: tuple[StrokedTextRun, ...],
    *,
    width: float = STROKED_VECTOR_PACK_WIDTH,
    horizontal_padding: float = STROKED_VECTOR_PACK_HORIZONTAL_PADDING,
    vertical_padding: float = STROKED_VECTOR_PACK_VERTICAL_PADDING,
) -> tuple[tuple[internal_StrokedTextCell, ...], float]:
    """Shelf-pack vector words without scaling their glyph geometry."""
    if not runs:
        return (), 0.0
    ordered = sorted(
        runs,
        key=lambda run: (
            -(run.bbox[3] - run.bbox[1]),
            -(run.bbox[2] - run.bbox[0]),
            run.drawing_indexes[0],
        ),
    )
    x = horizontal_padding
    y = vertical_padding
    row_height = 0.0
    cells: list[internal_StrokedTextCell] = []
    for run in ordered:
        source = run.bbox
        run_width = source[2] - source[0]
        run_height = source[3] - source[1]
        cell_width = run_width + horizontal_padding * 2.0
        cell_height = run_height + vertical_padding * 2.0
        if x > horizontal_padding and x + cell_width > width:
            y += row_height
            x = horizontal_padding
            row_height = 0.0
        packed = (
            x + horizontal_padding,
            y + vertical_padding,
            x + horizontal_padding + run_width,
            y + vertical_padding + run_height,
        )
        cells.append(
            internal_StrokedTextCell(
                source_box=source,
                packed_box=packed,
                drawing_indexes=run.drawing_indexes,
            )
        )
        x += cell_width
        row_height = max(row_height, cell_height)
    return tuple(cells), y + row_height + vertical_padding


def internal_stroked_vector_text_raster(
    capture: CapturedPage,
    requested_scale: float,
    *,
    max_pixels: int = MAX_OCR_PIXELS,
    variant: str = "seed",
    report: RecognitionReport | None = None,
) -> internal_PackedStrokedTextRaster | None:
    """Pack vector words into a compact seed raster with piecewise page mapping."""
    evidence = capture.evidence.stroked_vector_text
    if not evidence.trusted or not evidence.drawing_indexes:
        return None
    profile = internal_stroked_text_profile(capture)
    runs = stroked_text_isolated_runs(profile) if variant == "isolated" else profile.seed_runs
    if variant == "isolated":
        # Glyph-sized cells sit closer together than the remap tolerance, so
        # observation centres would match several cells and be dropped. Keep
        # the cell separation comfortably above the tolerance.
        horizontal_padding = STROKED_VECTOR_PACK_REMAP_TOLERANCE * 1.5
        vertical_padding = STROKED_VECTOR_PACK_REMAP_TOLERANCE * 1.5
    else:
        horizontal_padding = STROKED_VECTOR_PACK_HORIZONTAL_PADDING
        vertical_padding = (
            STROKED_VECTOR_PACK_DENSE_VERTICAL_PADDING
            if len(runs) >= STROKED_VECTOR_PACK_DENSE_MIN_CELLS
            else STROKED_VECTOR_PACK_VERTICAL_PADDING
        )
    cells, packed_height = internal_pack_stroked_text_runs(
        runs,
        horizontal_padding=horizontal_padding,
        vertical_padding=vertical_padding,
    )
    if not cells or packed_height <= 0.0:
        return None
    packed_width = STROKED_VECTOR_PACK_WIDTH
    area = max(1.0, packed_width * packed_height)
    safe_scale = math.sqrt(max_pixels / area) * 0.999
    scale = min(requested_scale, safe_scale)
    if variant == "isolated":
        # Isolated glyphs are the smallest text on the sheet (pin numbers run
        # 1-2pt tall); at the seed scale they raster below OCR's working size.
        # The montage area is tiny, so trade unused pixel budget for scale
        # until the median glyph is comfortably readable.
        median_height = finite_median(
            numpy.asarray([run.bbox[3] - run.bbox[1] for run in runs], dtype=numpy.float64)
        )
        target_scale = 24.0 / max(0.5, median_height)
        scale = min(max(requested_scale, target_scale), safe_scale, 48.0)
    page = capture.page

    compose_started = time.perf_counter()
    display_list = DisplayList(width=packed_width, height=packed_height)
    for cell in cells:
        tx = cell.packed_box[0] - cell.source_box[0]
        ty = cell.packed_box[1] - cell.source_box[1]
        # Hairline strokes raster as one-pixel skeletons, which OCR reads
        # confidently inside a word but misclassifies on a lone character.
        # Give isolated glyphs a stroke proportional to their own size so
        # they render solid at the boosted montage scale, and stand rotated
        # glyphs upright: pin numbers beside vertical wires are drawn
        # sideways, and OCR cannot classify a lone rotated character.
        rotation_matrix = None
        line_width_floor = 0.0
        if variant == "isolated":
            cell_width = cell.source_box[2] - cell.source_box[0]
            cell_height = cell.source_box[3] - cell.source_box[1]
            line_width_floor = min(cell_width, cell_height) * 0.14
            if cell_width > cell_height * 1.2:
                center_x = (cell.source_box[0] + cell.source_box[2]) * 0.5
                center_y = (cell.source_box[1] + cell.source_box[3]) * 0.5
                rotation_matrix = (
                    0.0,
                    -1.0,
                    1.0,
                    0.0,
                    center_x - center_y,
                    center_y + center_x,
                )
        for index in cell.drawing_indexes:
            drawing = capture.drawings[index]
            drawing_box = rect_tuple(getattr(drawing, "rect", None))
            path = getattr(drawing, "path", None)
            if drawing_box is None or path is None:
                continue
            if rotation_matrix is not None:
                path = path.transformed(rotation_matrix)
            # Packed cells preserve capture order and paint style. Reuse the
            # display-list stroke coalescer so thousands of tiny glyph paths do
            # not become thousands of renderer dispatches.
            display_list.append_captured_drawing(
                drawing.replace(
                    bbox=None,
                    path=path.translated(tx, ty),
                    fill_pattern=None,
                    stroke_pattern=None,
                    fill=(0.0, 0.0, 0.0),
                    stroke_color=(0.0, 0.0, 0.0),
                    line_width=max(float(drawing.line_width or 0.0), line_width_floor),
                )
            )
    rendered = RenderedPage(
        page_number=int(getattr(page, "page_number", 0)),
        width=packed_width,
        height=packed_height,
        rotate=0,
        display_list=display_list,
    )
    compose_seconds = time.perf_counter() - compose_started
    raster_started = time.perf_counter()
    # The Wu kernel draws one-pixel skeletons regardless of stroke width, so
    # the isolated variant must use the general renderer to honour the widened
    # strokes it just requested.
    fast_path = (
        variant != "isolated"
        and bool(display_list.items)
        and all(
            type(item) is PathPaintItem
            and item.paint_kind is PathPaintKind.STROKE
            and not (item.dash_pattern and item.dash_pattern[0])
            and item.blend_mode is None
            and item.soft_mask_alpha is None
            and (item.stroke_opacity is None or float(item.stroke_opacity) >= 0.999)
            and int(item.line_cap or 0) == 1
            and int(item.line_join or 0) == 1
            for item in display_list.items
        )
    )
    if fast_path:
        data = rasterize_packed_stroked_paths(
            tuple(display_list.items),
            packed_width,
            packed_height,
            scale,
        )
    else:
        data = rendered.rasterize(
            background=(255, 255, 255, 255),
            scale=scale,
            max_pixels=max_pixels,
            cache=False,
        )
    raster_seconds = time.perf_counter() - raster_started
    render_report: dict[str, object] = {
        "compose_seconds": compose_seconds,
        "rasterize_seconds": raster_seconds,
        "raster_mode": "packed-stroked-vector-text",
        "raster_kernel": "wu" if fast_path else "general",
        "crop": (0.0, 0.0, packed_width, packed_height),
        "raster_pixels": data.width * data.height,
        "pixel_budget": max_pixels,
        "include_native_text": False,
        "image_timings": {},
        "display_items": len(display_list.items),
        "display_item_kinds": {"compact-stroke": len(display_list.items)},
        "image_filters": (),
        "packed_cells": len(cells),
        "horizontal_padding": horizontal_padding,
        "vertical_padding": vertical_padding,
        "source_bbox": evidence.bbox,
    }
    raster = internal_Raster(
        data,
        max(70, int(round(72.0 * scale))),
        render_report,
    )
    packed = internal_PackedStrokedTextRaster(
        raster=raster,
        packed_box=(0.0, 0.0, packed_width, packed_height),
        cells=cells,
    )
    # The isolated-glyph supplement uses the general renderer by design. Keep
    # the pass-level timing tied to the seed atlas, matching the actual primary
    # recognition path and preserving the Wu-kernel performance invariant.
    if report is not None and variant == "seed":
        report.render_timings = render_report
    return packed


def internal_full_stroked_vector_text_raster(
    capture: CapturedPage,
    requested_scale: float,
    *,
    max_pixels: int = MAX_OCR_PIXELS,
    report: RecognitionReport | None = None,
) -> internal_RasterRegion | None:
    """Render the full compact-stroke layer when packed seed OCR is insufficient."""
    evidence = capture.evidence.stroked_vector_text
    if not evidence.trusted or evidence.bbox is None or not evidence.drawing_indexes:
        return None
    page = capture.page
    page_width = float(page.width)
    page_height = float(page.height)
    padding = 4.0
    bbox = evidence.bbox
    crop = (
        max(0.0, bbox[0] - padding),
        max(0.0, bbox[1] - padding),
        min(page_width, bbox[2] + padding),
        min(page_height, bbox[3] + padding),
    )
    area = max(1.0, (crop[2] - crop[0]) * (crop[3] - crop[1]))
    safe_scale = math.sqrt(max_pixels / area) * 0.999
    scale = min(requested_scale, safe_scale)
    compose_started = time.perf_counter()
    display_list = DisplayList(width=page_width, height=page_height)
    for index in evidence.drawing_indexes:
        drawing = capture.drawings[index]
        display_list.append(
            drawing.kind,
            drawing.seqno,
            bbox=drawing.rect,
            path=drawing.path,
            fill=(0.0, 0.0, 0.0),
            fill_opacity=drawing.fill_opacity,
            stroke_color=(0.0, 0.0, 0.0),
            stroke_opacity=drawing.stroke_opacity,
            line_width=drawing.line_width,
            line_cap=drawing.line_cap,
            line_join=drawing.line_join,
            dash_pattern=drawing.dash_pattern,
            fill_rule=drawing.fill_rule,
            blend_mode=drawing.blend_mode,
            soft_mask_alpha=drawing.soft_mask_alpha,
        )
    rendered = RenderedPage(
        page_number=int(getattr(page, "page_number", 0)),
        width=page_width,
        height=page_height,
        rotate=0,
        display_list=display_list,
    )
    compose_seconds = time.perf_counter() - compose_started
    raster_started = time.perf_counter()
    data = rendered.rasterize(
        background=(255, 255, 255, 255),
        scale=scale,
        max_pixels=max_pixels,
        crop=crop,
        cache=False,
    )
    render_report: dict[str, object] = {
        "compose_seconds": compose_seconds,
        "rasterize_seconds": time.perf_counter() - raster_started,
        "raster_mode": "stroked-vector-text-fallback",
        "crop": crop,
        "raster_pixels": data.width * data.height,
        "pixel_budget": max_pixels,
        "include_native_text": False,
        "image_timings": {},
        "display_items": len(display_list.items),
        "display_item_kinds": {"compact-stroke": len(display_list.items)},
        "image_filters": (),
    }
    raster = internal_Raster(
        data,
        max(70, int(round(72.0 * scale))),
        render_report,
    )
    region = internal_RasterRegion(raster, crop)
    if report is not None:
        report.render_timings = render_report
    return region


def internal_remap_stroked_vector_observations(
    observations: ObservationBatch,
    packed: internal_PackedStrokedTextRaster,
) -> tuple[ObservationBatch, int]:
    """Translate montage observations back through their containing cells."""
    texts: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []
    polygons: list[tuple[float, ...]] = []
    confidences: list[float] = []
    sequences: list[int] = []
    references: list[Any | None] = []
    tolerance = STROKED_VECTOR_PACK_REMAP_TOLERANCE
    cell_index = SpatialIndex.from_items(
        packed.cells,
        bbox=lambda cell: (
            cell.packed_box[0] - tolerance,
            cell.packed_box[1] - tolerance,
            cell.packed_box[2] + tolerance,
            cell.packed_box[3] + tolerance,
        ),
    )
    for index, packed_box in enumerate(observations.bbox):
        box = internal_bbox_tuple(packed_box)
        center_x = (box[0] + box[2]) * 0.5
        center_y = (box[1] + box[3]) * 0.5
        # Broad-phase grid lookup narrows to spatially nearby cells; the exact
        # tolerance check below is unchanged so results are identical to a
        # full scan, just without touching every cell on the page per glyph.
        # The query box needs a positive area (SpatialIndex rejects a
        # degenerate point), so pad it by a fixed epsilon far smaller than
        # any real geometry difference -- it only widens the broad-phase
        # candidate set, never the final exact-tolerance result.
        query_box = (
            center_x - 1e-6,
            center_y - 1e-6,
            center_x + 1e-6,
            center_y + 1e-6,
        )
        cells = tuple(
            cell
            for cell in cell_index.candidates(query_box)
            if cell.packed_box[0] - tolerance <= center_x <= cell.packed_box[2] + tolerance
            and cell.packed_box[1] - tolerance <= center_y <= cell.packed_box[3] + tolerance
        )
        if len(cells) != 1:
            continue
        cell = cells[0]
        dx = cell.source_box[0] - cell.packed_box[0]
        dy = cell.source_box[1] - cell.packed_box[1]
        mapped = (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)
        texts.append(observations.text[index])
        boxes.append(mapped)
        polygons.append(
            (
                mapped[0],
                mapped[1],
                mapped[2],
                mapped[1],
                mapped[2],
                mapped[3],
                mapped[0],
                mapped[3],
            )
        )
        confidences.append(float(observations.confidence[index]))
        sequences.append(cell.drawing_indexes[0])
        references.append(observations.references[index])
    return (
        ObservationBatch.from_columns(
            texts,
            boxes,
            polygon=polygons,
            source=ObservationSource.OCR,
            confidence=confidences,
            sequence=sequences,
            rotation=(0 for _ in texts),
            font_size=(box[3] - box[1] for box in boxes),
            line_break_before=(True for _ in texts),
            references=references,
        ),
        len(observations) - len(texts),
    )


def internal_isolated_pin_label(text: str) -> bool:
    """Report whether an isolated-glyph OCR read looks like a pin label.

    The isolated montage inevitably contains junction dots and wire stubs
    alongside the real lone glyphs, and OCR renders those as stray letters
    and slashes. Genuine isolated labels on a schematic are short and carry
    a digit -- pin numbers, reference suffixes -- so keep only those.
    """
    stripped = text.strip()
    return 1 <= len(stripped) <= 4 and any(character.isdigit() for character in stripped)


def internal_remap_stroked_vector_candidate(
    candidate: internal_Candidate,
    packed: internal_PackedStrokedTextRaster,
    *,
    digit_bearing_only: bool = False,
) -> tuple[internal_Candidate, int]:
    """Translate montage OCR words and symbols into their source page cells."""
    remapped, unmapped = internal_remap_stroked_vector_observations(
        candidate.observations,
        packed,
    )
    if digit_bearing_only:
        remapped = remapped.take(
            tuple(
                index
                for index, text in enumerate(remapped.text)
                if internal_isolated_pin_label(text)
            )
        )
    remapped_symbols, _unmapped_symbols = internal_remap_stroked_vector_observations(
        candidate.symbols,
        packed,
    )
    return (
        internal_candidate(
            candidate.mode,
            remapped,
            symbols=remapped_symbols,
            api_seconds=candidate.api_seconds,
            setup_seconds=candidate.setup_seconds,
            recognition_seconds=candidate.recognition_seconds,
            iterator_seconds=candidate.iterator_seconds,
            cleanup_seconds=candidate.cleanup_seconds,
            candidate_seconds=candidate.candidate_seconds,
            recognition_status=candidate.recognition_status,
            median_text_height=candidate.metrics.median_text_height,
        ),
        unmapped,
    )


STROKED_VECTOR_DECODE_MIN_OVERLAP = 0.55


STROKED_VECTOR_MULTI_EDIT_MIN_OVERLAP = 0.90


STROKED_VECTOR_MULTI_EDIT_MAX_CONFIDENCE = 85.0


def internal_stroked_vector_decoded_batch(
    observations: tuple[StrokedTextObservation, ...],
) -> ObservationBatch:
    boxes = tuple(observation.bbox for observation in observations)
    return ObservationBatch.from_columns(
        (observation.text for observation in observations),
        boxes,
        polygon=((box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3]) for box in boxes),
        source=ObservationSource.STRUCTURE,
        confidence=(observation.confidence for observation in observations),
        sequence=(observation.first_drawing for observation in observations),
        rotation=(0 for _ in observations),
        font_size=(max(0.0, box[3] - box[1]) for box in boxes),
        line_break_before=(True for _ in observations),
    )


def internal_single_character_substitution(left: str, right: str) -> bool:
    return len(left) == len(right) and sum(a != b for a, b in zip(left, right, strict=True)) == 1


def internal_bounded_edit_distance(left: str, right: str, maximum: int) -> int:
    """Return a small Levenshtein distance, stopping once the bound is exceeded."""
    if abs(len(left) - len(right)) > maximum:
        return maximum + 1
    previous = list(range(len(right) + 1))
    for row, left_character in enumerate(left, 1):
        current = [row]
        for column, right_character in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_character != right_character),
                )
            )
        if min(current) > maximum:
            return maximum + 1
        previous = current
    return previous[-1]


def internal_stroked_vector_substitution(
    recognized: str,
    decoded: str,
    *,
    confidence: float,
    overlap: float,
) -> bool:
    if internal_single_character_substitution(recognized, decoded):
        return True
    left = recognized.casefold()
    right = decoded.casefold()
    return bool(
        confidence < STROKED_VECTOR_MULTI_EDIT_MAX_CONFIDENCE
        and overlap >= STROKED_VECTOR_MULTI_EDIT_MIN_OVERLAP
        and 2 <= len(left) <= 12
        and 2 <= len(right) <= 12
        and abs(len(left) - len(right)) <= 1
        and left[0] == right[0]
        and any(character.isalnum() for character in left)
        and any(character.isalnum() for character in right)
        and internal_bounded_edit_distance(left, right, 2) <= 2
    )


def internal_stroked_vector_symbol_seeds(
    capture: CapturedPage,
    symbols: ObservationBatch,
) -> tuple[StrokedTextSeed, ...]:
    """Join character boxes only when they exactly fill one known vector run."""
    if not len(symbols):
        return ()
    runs_by_sequence = {
        run.drawing_indexes[0]: run for run in internal_stroked_text_profile(capture).seed_runs
    }
    grouped: dict[
        int,
        list[tuple[float, str, float]],
    ] = defaultdict(list)
    for text, raw_box, confidence, raw_sequence in zip(
        symbols.text,
        symbols.bbox,
        symbols.confidence,
        symbols.sequence,
        strict=True,
    ):
        character = text.strip()
        sequence = int(raw_sequence)
        if len(character) != 1 or sequence not in runs_by_sequence:
            continue
        grouped[sequence].append((internal_bbox_tuple(raw_box)[0], character, float(confidence)))

    seeds: list[StrokedTextSeed] = []
    for sequence, items in grouped.items():
        run = runs_by_sequence[sequence]
        if len(items) != run.glyph_count:
            continue
        ordered = sorted(items)
        seeds.append(
            StrokedTextSeed(
                text="".join(character for ignored_x, character, ignored_confidence in ordered),
                bbox=run.bbox,
                confidence=min(confidence for ignored_x, ignored_character, confidence in ordered),
                sequence=sequence,
            )
        )
    return tuple(seeds)


def internal_decode_stroked_vector_text(
    capture: CapturedPage,
    ocr: ObservationBatch,
    symbols: ObservationBatch | None = None,
) -> StrokedTextDecode:
    evidence = capture.evidence.stroked_vector_text
    if not evidence.trusted or not evidence.drawing_indexes or not len(ocr):
        return StrokedTextDecode()
    profile = internal_stroked_text_profile(capture)
    word_seeds = tuple(
        StrokedTextSeed(
            text=text,
            bbox=internal_bbox_tuple(box),
            confidence=float(confidence),
            sequence=int(sequence),
        )
        for text, box, confidence, sequence in zip(
            ocr.text,
            ocr.bbox,
            ocr.confidence,
            ocr.sequence,
            strict=True,
        )
    )
    symbol_seeds = internal_stroked_vector_symbol_seeds(
        capture,
        symbols if symbols is not None else ObservationBatch.empty(),
    )
    if not symbol_seeds:
        return decode_stroked_text_profile(profile, word_seeds)
    return decode_stroked_text_profile_with_supplemental_seeds(
        profile,
        word_seeds,
        symbol_seeds,
    )


def internal_packed_stroked_vector_decode_gate(
    decoded: StrokedTextDecode,
    cell_count: int,
) -> tuple[bool, dict[str, int | bool]]:
    """Require enough learned geometry before skipping the full-layer OCR fallback."""
    aligned_required = min(
        STROKED_VECTOR_PACK_MIN_ALIGNED_SEEDS,
        max(4, cell_count // 4),
    )
    learned_required = min(
        STROKED_VECTOR_PACK_MIN_LEARNED_SIGNATURES,
        max(8, cell_count // 3),
    )
    decoded_required = min(
        STROKED_VECTOR_PACK_MIN_DECODED_RUNS,
        max(8, cell_count // 3),
    )
    accepted = bool(
        decoded.aligned_seeds >= aligned_required
        and decoded.learned_signatures >= learned_required
        and len(decoded.observations) >= decoded_required
    )
    return accepted, {
        "accepted": accepted,
        "cells": cell_count,
        "aligned_seeds": decoded.aligned_seeds,
        "aligned_required": aligned_required,
        "learned_signatures": decoded.learned_signatures,
        "learned_required": learned_required,
        "decoded_runs": len(decoded.observations),
        "decoded_required": decoded_required,
    }


def internal_recover_stroked_vector_text(
    capture: CapturedPage,
    ocr: ObservationBatch,
    report: RecognitionReport | None = None,
    *,
    cached_decode: tuple[int, StrokedTextDecode, float] | None = None,
) -> ObservationBatch:
    """Augment one OCR pass with text decoded from repeated vector glyphs."""
    evidence = capture.evidence.stroked_vector_text
    if not evidence.trusted or not evidence.drawing_indexes or not len(ocr):
        return ocr
    report = report or RecognitionReport()
    started = time.perf_counter()
    if cached_decode is not None and cached_decode[0] == id(ocr):
        decoded = cached_decode[1]
        prior_decode_seconds = float(cached_decode[2])
    else:
        decoded = internal_decode_stroked_vector_text(capture, ocr)
        prior_decode_seconds = 0.0
    ocr_index = SpatialIndex.from_boxes(ocr.bbox)
    replacements: set[int] = set()
    accepted: list[StrokedTextObservation] = []
    additions = 0
    corrections = 0
    for observation in decoded.observations:
        candidate_area = max(
            0.01,
            (observation.bbox[2] - observation.bbox[0])
            * (observation.bbox[3] - observation.bbox[1]),
        )
        overlaps: list[tuple[float, int]] = []
        for hit in ocr_index.intersecting_hits(observation.bbox):
            hit_area = max(0.01, (hit.bbox[2] - hit.bbox[0]) * (hit.bbox[3] - hit.bbox[1]))
            overlap = bbox_intersection_area(observation.bbox, hit.bbox) / min(
                candidate_area, hit_area
            )
            if overlap >= STROKED_VECTOR_DECODE_MIN_OVERLAP:
                overlaps.append((overlap, int(hit.item)))
        if not overlaps:
            accepted.append(observation)
            additions += 1
            continue
        best_overlap, best_index = max(overlaps)
        recognized_text = ocr.text[best_index].strip()
        if recognized_text == observation.text:
            continue
        if best_index not in replacements and internal_stroked_vector_substitution(
            recognized_text,
            observation.text,
            confidence=float(ocr.confidence[best_index]),
            overlap=best_overlap,
        ):
            replacements.add(best_index)
            accepted.append(observation)
            corrections += 1

    report.stroked_vector_decode = {
        "seconds": prior_decode_seconds + time.perf_counter() - started,
        "eligible_seeds": decoded.eligible_seeds,
        "aligned_seeds": decoded.aligned_seeds,
        "accepted_seeds": decoded.accepted_seeds,
        "initial_signatures": decoded.initial_signatures,
        "learned_signatures": decoded.learned_signatures,
        "approximate_signatures": decoded.approximate_signatures,
        "candidate_runs": decoded.candidate_runs,
        "decoded_candidate_runs": decoded.decoded_candidate_runs,
        "candidate_run_coverage": decoded.candidate_run_coverage,
        "candidate_glyph_coverage": decoded.candidate_glyph_coverage,
        "decoded_runs": len(decoded.observations),
        "additions": additions,
        "corrections": corrections,
    }
    report.stroked_vector_alphabet = tuple(decoded.alphabet)
    if not accepted:
        return ocr
    retained = ocr.take(tuple(index for index in range(len(ocr)) if index not in replacements))
    return ObservationBatch.concatenate(
        retained,
        internal_stroked_vector_decoded_batch(tuple(accepted)),
    )
