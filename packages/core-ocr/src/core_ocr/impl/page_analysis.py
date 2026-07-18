# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from core_layout.impl.layout.models import TextRun

from core_ocr.impl import text_analysis as ocr_text_analysis
from core_ocr.impl.geometry import TextGeometryLine, text_geometry_line_from_bbox
from core_ocr.impl.services import get_candidate_services
from core_ocr.impl.text_analysis import uninterpretable_char_count

OCR_FALLBACK_DPI = 300
OCR_FALLBACK_SPARSE_TEXT_TOKENS = 120
OCR_FALLBACK_IMAGE_AREA_RATIO = 0.50
OCR_FIGURE_IMAGE_MIN_PIXELS = 300_000
OCR_FIGURE_IMAGE_MIN_AREA_RATIO = 0.035
OCR_FIGURE_IMAGE_LEGACY_MIN_AREA_RATIO = 0.09
OCR_FIGURE_HIGH_DENSITY_MIN_AREA_RATIO = 0.055
OCR_FIGURE_IMAGE_MAX_AREA_RATIO = 0.55
OCR_FIGURE_CAPTIONED_IMAGE_MAX_AREA_RATIO = 1.01
OCR_FIGURE_MAX_REGIONS = 4
OCR_FIGURE_CAPTION_MAX_DISTANCE = 90.0
OCR_FIGURE_MAX_NATIVE_OVERLAP_RATIO = 0.28
OCR_FIGURE_MIN_PIXEL_DENSITY = 24.0
OCR_FIGURE_IMAGE_ONLY_MIN_PIXEL_DENSITY = 12.0
OCR_LARGE_EMBEDDED_IMAGE_MIN_PIXELS = 350_000
OCR_LARGE_EMBEDDED_IMAGE_MIN_AREA_RATIO = 0.12
VECTOR_SPATIAL_TEXT_RE = re.compile(r"[A-Za-z0-9_+\-]")
VECTOR_SPATIAL_ALLOWED_PUNCTUATION = frozenset("+-._/")


@dataclass(frozen=True)
class FigureOcrRegion:
    bbox: tuple[float, float, float, float]
    item_index: int
    source_kind: str
    caption_text: str = ""
    caption_bbox: tuple[float, float, float, float] | None = None
    signals: dict[str, float | int | str | bool] | None = None


def rendered_page_for_ocr_analysis(page: Any) -> Any:
    cache = page.extraction_cache
    cache_key = "ocr_analysis_rendered_page"
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
    rendered = get_candidate_services().render_page_for_ocr_analysis(page)
    if cache is not None:
        cache[cache_key] = rendered
    return rendered


def native_text_run_has_corrupt_control_text(run: TextRun) -> bool:
    text = run.text
    if len(text) < 8:
        return False
    controls = sum(1 for ch in text if ord(ch) < 0x20 and ch not in "\t\n\r")
    if controls < 2:
        return False
    nonspace = sum(1 for ch in text if not ch.isspace())
    if controls / max(1, nonspace) >= 0.12:
        return True
    return controls >= 4 and any(
        ch in text
        for ch in (
            "\ufb01",
            "\u02d9",
            "\u2212",
            "\u0142",
            "\u017d",
            "\u017e",
        )
    )


def native_text_runs_without_corrupt_control_text(
    runs: list[TextRun],
) -> list[TextRun]:
    if not any(native_text_run_has_corrupt_control_text(run) for run in runs):
        return runs
    return [run for run in runs if not native_text_run_has_corrupt_control_text(run)]


def native_text_run_looks_symbol_encoded_artifact(run: TextRun) -> bool:
    text = run.text.strip()
    if len(text) < 12:
        return False
    if any(ord(ch) < 0x20 and ch not in "\t\n\r" for ch in text):
        return False
    nonspace = sum(1 for ch in text if not ch.isspace())
    if nonspace < 12:
        return False
    alnum = sum(1 for ch in text if ch.isalnum() or ch == "_")
    punctuation = nonspace - alnum
    ascii_punctuation = sum(
        1 for ch in text if 0x21 <= ord(ch) <= 0x7E and not ch.isalnum() and ch != "_"
    )
    return punctuation / max(1, nonspace) >= 0.32 and ascii_punctuation >= 6 and alnum >= 4


def page_has_symbol_encoded_native_text_artifacts(page: Any) -> bool:
    artifact_runs = 0
    for run in native_text_runs_without_corrupt_control_text(page.chars):
        if native_text_run_looks_symbol_encoded_artifact(run):
            artifact_runs += 1
            if artifact_runs >= 8:
                return True
    return False


def native_text_geometry_lines(
    page: Any,
    *,
    include_hidden: bool = False,
) -> list[TextGeometryLine]:
    cache = getattr(page, "extraction_cache", None)
    cache_key = ("native_text_geometry_lines", include_hidden)
    if isinstance(cache, dict) and cache_key in cache:
        return cache[cache_key]
    runs = [
        run
        for run in getattr(page, "chars", ())
        if run.has_text and run.stripped_text and (include_hidden or run.visible)
    ]
    if not runs:
        if isinstance(cache, dict):
            cache[cache_key] = []
        return []
    lines: list[TextGeometryLine] = []
    for line in get_candidate_services().layout_analyzer.cluster_into_lines(runs):
        text = line.text().strip()
        if not text:
            continue
        lines.append(
            text_geometry_line_from_bbox(
                text,
                (line.x0, line.y0, line.x1, line.y1),
                source="native",
                kind="native_text_line",
            )
        )
    if isinstance(cache, dict):
        cache[cache_key] = lines
    return lines


def has_uninterpretable_type3_fonts(page: Any) -> bool:
    fonts = get_candidate_services().pdf_analysis.lookup_dict_key(page.resources, "Font")
    if not isinstance(fonts, dict):
        return False
    resolver = page.document.resolver
    for font_ref in fonts.values():
        font = resolver.resolve(font_ref)
        if isinstance(font, get_candidate_services().pdf_analysis.stream_type):
            font = font.dictionary
        if not isinstance(font, dict):
            continue
        pdf_analysis = get_candidate_services().pdf_analysis
        if (
            pdf_analysis.normalize_pdf_name(pdf_analysis.lookup_dict_key(font, "Subtype"))
            != "Type3"
        ):
            continue
        if get_candidate_services().pdf_analysis.lookup_dict_key(font, "ToUnicode") is not None:
            continue
        if not has_meaningful_type3_encoding(font):
            return True
    return False


def dense_numeric_native_text_layer_is_preferable(
    text: str, *, text_tokens: int | None = None
) -> bool:
    if text_tokens is None:
        text_tokens = ocr_text_analysis.extracted_text_token_count(text)
    if text_tokens < 300:
        return False
    if ocr_text_analysis.numeric_token_ratio(text) < 0.45:
        return False
    quality = ocr_text_analysis.text_ocr_quality_score(text)
    artifact = ocr_text_analysis.scanned_ocr_artifact_score(text)
    if artifact >= 0.08 and ocr_text_analysis.sparse_text_looks_noisy(text):
        return False
    if quality >= 0.24 and artifact >= 0.05:
        return False
    return quality <= 0.60


def scanned_table_native_text_layer_looks_weak(
    page: Any,
    text: str,
    *,
    text_tokens: int | None = None,
) -> bool:
    if text_tokens is None:
        text_tokens = ocr_text_analysis.extracted_text_token_count(text)
    if text_tokens < 220:
        return False
    numeric_ratio = ocr_text_analysis.numeric_token_ratio(text)
    if numeric_ratio < 0.18 and not ocr_text_analysis.text_has_many_digit_lines(text):
        return False
    quality = ocr_text_analysis.text_ocr_quality_score(text)
    artifact = ocr_text_analysis.scanned_ocr_artifact_score(text)
    if quality < 0.14 and artifact < 0.04 and not ocr_text_analysis.sparse_text_looks_noisy(text):
        return False
    try:
        if has_dominant_page_image(page) or page_has_large_embedded_image(page):
            return True
    except Exception:
        return False
    return False


def native_text_layer_has_substantial_page_coverage(page: Any, text_tokens: int) -> bool:
    if text_tokens < 430:
        return False
    if getattr(page, "rotation", 0) % 360 != 0:
        return False

    media_box = get_candidate_services().page_geometry.rect_box_tuple(
        getattr(page, "media_box", None)
    )
    if media_box is None:
        return False
    page_width = media_box[2] - media_box[0]
    page_height = media_box[3] - media_box[1]
    if page_width <= 0 or page_height <= 0:
        return False

    x_slop = max(6.0, page_width * 0.03)
    y_slop = max(6.0, page_height * 0.03)
    min_x = media_box[0] - x_slop
    max_x = media_box[2] + x_slop
    min_y = media_box[1] - y_slop
    max_y = media_box[3] + y_slop
    runs = [
        run
        for run in getattr(page, "chars", ())
        if run.has_text
        and run.stripped_text
        and run.visible
        and not run.is_vertical
        and run.x1 >= min_x
        and run.x0 <= max_x
        and run.y1 >= min_y
        and run.y0 <= max_y
    ]
    if len(runs) < 20:
        return False

    lines = [
        line
        for line in get_candidate_services().layout_analyzer.cluster_into_lines(runs)
        if any(run.stripped_text for run in line.runs)
    ]
    if len(lines) < 40:
        return False

    x0 = min(run.x0 for run in runs)
    x1 = max(run.x1 for run in runs)
    y0 = min(run.y0 for run in runs)
    y1 = max(run.y1 for run in runs)
    horizontal_coverage = (x1 - x0) / page_width
    vertical_coverage = (y1 - y0) / page_height
    if horizontal_coverage < 0.45 or vertical_coverage < 0.55:
        return False

    occupied_bands: set[int] = set()
    for line in lines:
        relative_mid_y = (line.mid_y - media_box[1]) / page_height
        if 0.0 <= relative_mid_y <= 1.0:
            occupied_bands.add(min(4, max(0, int(relative_mid_y * 5))))
    if len(occupied_bands) < 3:
        return False

    cache = getattr(page, "extraction_cache", None)
    if isinstance(cache, dict):
        cache["native_substantial_coverage_ocr_supplement_skipped"] = True
    return True


def native_text_layer_has_sparse_page_coverage(page: Any) -> bool:
    if getattr(page, "rotation", 0) % 360 != 0:
        return False

    media_box = get_candidate_services().page_geometry.rect_box_tuple(
        getattr(page, "media_box", None)
    )
    if media_box is None:
        return False
    page_width = media_box[2] - media_box[0]
    page_height = media_box[3] - media_box[1]
    if page_width <= 0 or page_height <= 0:
        return False

    runs = [
        run
        for run in getattr(page, "chars", ())
        if run.has_text and run.stripped_text and run.visible and not run.is_vertical
    ]
    if not runs:
        return False

    lines = [
        line
        for line in get_candidate_services().layout_analyzer.cluster_into_lines(runs)
        if any(run.stripped_text for run in line.runs)
    ]
    if len(lines) < 8:
        return True

    x0 = min(run.x0 for run in runs)
    x1 = max(run.x1 for run in runs)
    y0 = min(run.y0 for run in runs)
    y1 = max(run.y1 for run in runs)
    horizontal_coverage = (x1 - x0) / page_width
    vertical_coverage = (y1 - y0) / page_height
    if horizontal_coverage < 0.30 or vertical_coverage < 0.45:
        return True

    occupied_bands: set[int] = set()
    for line in lines:
        relative_mid_y = (line.mid_y - media_box[1]) / page_height
        if 0.0 <= relative_mid_y <= 1.0:
            occupied_bands.add(min(4, max(0, int(relative_mid_y * 5))))
    return len(occupied_bands) < 3


def dominant_image_requires_ocr_verification(page: Any) -> bool:
    """Return whether a raster page should be checked against its native text.

    A scanned page can carry a large invisible text layer that looks coherent
    enough to suppress OCR while still being badly decoded.  Sparse painted
    text is the useful discriminator here: a page whose visible text covers
    only a small part of a dominant image should get an independent OCR pass.
    """
    try:
        return has_dominant_page_image(page) and native_text_layer_has_sparse_page_coverage(page)
    except Exception:
        return False


def native_text_should_be_omitted_from_ocr_render(
    page: Any,
    text: str,
) -> bool:
    """Return whether native text would contaminate the OCR raster.

    Some scanned PDFs contain a visible text layer that is not merely stale;
    its decoded glyphs are wrong and are painted over the scan.  Tesseract can
    then recognize the bad overlay instead of the underlying page image.
    """
    try:
        if not dominant_image_requires_ocr_verification(page):
            return False
    except Exception:
        return False
    # A bad Unicode mapping does not imply bad painted glyphs: NASA's page is
    # the counterexample.  Suppress text only when the sparse overlay also
    # contains characters that cannot be rendered/reconciled reliably.
    return uninterpretable_char_count(text) > 0


def page_has_many_non_image_drawings(page: Any) -> bool:
    try:
        rendered = rendered_page_for_ocr_analysis(page)
    except Exception:
        return False
    count = 0
    for item in rendered.display_list.items:
        if item.kind not in {"fill", "fillstroke", "stroke", "shading"}:
            continue
        count += 1
        if count >= 20:
            return True
    return False


def has_dominant_page_image(page: Any) -> bool:
    rendered = rendered_page_for_ocr_analysis(page)
    page_area = max(1.0, float(rendered.width) * float(rendered.height))
    for item in rendered.display_list.items:
        if item.kind not in {"image", "inline-image"}:
            continue
        metadata = item.data.get("image_metadata")
        if not isinstance(metadata, dict):
            continue
        if int(metadata.get("pixels") or 0) < 100_000:
            continue
        box = get_candidate_services().page_geometry.rect_box_tuple(item.data.get("bbox"))
        if box is None:
            continue
        x0, y0, x1, y1 = box
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if area >= page_area * OCR_FALLBACK_IMAGE_AREA_RATIO:
            return True
    return False


def page_has_large_embedded_image(page: Any) -> bool:
    rendered = rendered_page_for_ocr_analysis(page)
    page_area = max(1.0, float(rendered.width) * float(rendered.height))
    for item in rendered.display_list.items:
        if item.kind not in {"image", "inline-image"}:
            continue
        metadata = item.data.get("image_metadata")
        if not isinstance(metadata, dict):
            continue
        if int(metadata.get("pixels") or 0) < OCR_LARGE_EMBEDDED_IMAGE_MIN_PIXELS:
            continue
        box = get_candidate_services().page_geometry.rect_box_tuple(item.data.get("bbox"))
        if box is None:
            continue
        x0, y0, x1, y1 = box
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if area >= page_area * OCR_LARGE_EMBEDDED_IMAGE_MIN_AREA_RATIO:
            return True
    return False


def figure_ocr_regions(page: Any) -> tuple[FigureOcrRegion, ...]:
    if getattr(page, "rotation", 0) % 360 != 0:
        return ()
    cache = getattr(page, "extraction_cache", None)
    cache_key = "figure_ocr_regions"
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    rendered = rendered_page_for_ocr_analysis(page)
    page_area = max(1.0, float(rendered.width) * float(rendered.height))
    page_height = max(1.0, float(rendered.height))
    native_lines = native_text_geometry_lines(page)
    regions: list[FigureOcrRegion] = []
    for item_index, item in enumerate(rendered.display_list.items):
        if item.kind not in {"image", "inline-image"}:
            continue
        metadata = item.data.get("image_metadata")
        if not isinstance(metadata, dict):
            continue
        if int(metadata.get("pixels") or 0) < OCR_FIGURE_IMAGE_MIN_PIXELS:
            continue
        box = get_candidate_services().page_geometry.rect_box_tuple(item.data.get("bbox"))
        if box is None:
            continue
        x0, y0, x1, y1 = box
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if area <= 0:
            continue
        area_ratio = area / page_area
        normalized_box = get_candidate_services().page_geometry.normalize_rect((x0, y0, x1, y1))
        if normalized_box is None:
            continue
        caption = figure_caption_for_box(native_lines, normalized_box)
        source_pixels = int(metadata.get("pixels") or 0)
        image_only_full_page_candidate = (
            area_ratio >= 0.85
            and not native_lines
            and source_pixels / max(area, 1.0) >= OCR_FIGURE_IMAGE_ONLY_MIN_PIXEL_DENSITY
        )
        max_area_ratio = (
            OCR_FIGURE_CAPTIONED_IMAGE_MAX_AREA_RATIO
            if caption is not None or image_only_full_page_candidate
            else OCR_FIGURE_IMAGE_MAX_AREA_RATIO
        )
        if not (OCR_FIGURE_IMAGE_MIN_AREA_RATIO <= area_ratio < max_area_ratio):
            continue
        mid_y_ratio = ((y0 + y1) * 0.5) / page_height
        if not (0.05 <= mid_y_ratio <= 0.95):
            continue
        native_overlap = text_geometry_overlap_ratio(native_lines, normalized_box)
        pixel_density = source_pixels / area
        has_caption = caption is not None
        high_density = pixel_density >= OCR_FIGURE_MIN_PIXEL_DENSITY
        image_only_full_page_region = image_only_full_page_candidate
        legacy_area = area_ratio >= OCR_FIGURE_IMAGE_LEGACY_MIN_AREA_RATIO
        high_density_region = area_ratio >= OCR_FIGURE_HIGH_DENSITY_MIN_AREA_RATIO and high_density
        should_keep = native_overlap <= OCR_FIGURE_MAX_NATIVE_OVERLAP_RATIO and (
            legacy_area
            or (has_caption and high_density)
            or high_density_region
            or image_only_full_page_region
        )
        if not should_keep:
            continue
        caption_text, caption_bbox = caption if caption is not None else ("", None)
        regions.append(
            FigureOcrRegion(
                normalized_box,
                item_index,
                str(item.kind),
                caption_text,
                caption_bbox,
                {
                    "area_ratio": round(area_ratio, 5),
                    "pixels": source_pixels,
                    "pixel_density": round(pixel_density, 2),
                    "native_overlap": round(native_overlap, 5),
                    "caption": has_caption,
                    "legacy_area": legacy_area,
                    "high_density_region": high_density_region,
                    "image_only_full_page_region": image_only_full_page_region,
                },
            )
        )
    if len(regions) > OCR_FIGURE_MAX_REGIONS:
        regions = sorted(
            regions,
            key=lambda region: get_candidate_services().page_geometry.rect_area(region.bbox),
            reverse=True,
        )[:OCR_FIGURE_MAX_REGIONS]
    regions.sort(key=lambda region: (-region.bbox[3], region.bbox[0]))
    result = tuple(regions)
    if cache is not None:
        cache[cache_key] = result
    return result


def figure_caption_for_box(
    lines: list[TextGeometryLine],
    box: tuple[float, float, float, float],
) -> tuple[str, tuple[float, float, float, float]] | None:
    best: tuple[float, str, tuple[float, float, float, float]] | None = None
    for line in lines:
        if not text_line_starts_with_figure_caption(line.text):
            continue
        line_bbox = line.observation.bbox
        if line_bbox is None:
            continue
        horizontal_overlap = max(
            0.0,
            min(box[2], line_bbox[2]) - max(box[0], line_bbox[0]),
        )
        if horizontal_overlap <= 0.0:
            continue
        distance = figure_caption_distance(box, line_bbox)
        if distance is None or distance > OCR_FIGURE_CAPTION_MAX_DISTANCE:
            continue
        overlap_ratio = horizontal_overlap / max(
            1.0,
            min(box[2] - box[0], line_bbox[2] - line_bbox[0]),
        )
        if overlap_ratio < 0.12:
            continue
        score = distance - overlap_ratio * 10.0
        if best is None or score < best[0]:
            best = (score, line.text.strip(), line_bbox)
    if best is None:
        return None
    return best[1], best[2]


def text_line_starts_with_figure_caption(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    lower = stripped.casefold()
    offset = 0
    if lower.startswith("figure"):
        offset = len("figure")
    elif lower.startswith("fig"):
        offset = len("fig")
        if offset < len(stripped) and stripped[offset] == ".":
            offset += 1
    else:
        return False
    while offset < len(stripped) and stripped[offset].isspace():
        offset += 1
    return offset < len(stripped) and stripped[offset].isdigit()


def figure_caption_distance(
    box: tuple[float, float, float, float],
    caption_box: tuple[float, float, float, float],
) -> float | None:
    if caption_box[3] <= box[1]:
        return box[1] - caption_box[3]
    if caption_box[1] >= box[3]:
        return caption_box[1] - box[3]
    box_height = max(1.0, box[3] - box[1])
    caption_mid_y = (caption_box[1] + caption_box[3]) * 0.5
    if box[1] <= caption_mid_y <= box[1] + box_height * 0.14:
        return 0.0
    if box[3] - box_height * 0.14 <= caption_mid_y <= box[3]:
        return 0.0
    return None


def text_geometry_overlap_ratio(
    lines: list[TextGeometryLine],
    box: tuple[float, float, float, float],
) -> float:
    area = get_candidate_services().page_geometry.rect_area(box)
    if area <= 0.0:
        return 1.0
    overlap = 0.0
    for line in lines:
        if line.observation.bbox is None:
            continue
        overlap += get_candidate_services().page_geometry.rect_intersection_area(
            line.observation.bbox, box
        )
    return overlap / area


def page_has_figure_ocr_region(page: Any) -> bool:
    return bool(figure_ocr_regions(page))


def dominant_image_text_layer_looks_weak(text: str) -> bool:
    tokens = ocr_text_analysis.extracted_text_token_count(text)
    if tokens < 80:
        return False
    quality = ocr_text_analysis.text_ocr_quality_score(text)
    return ocr_text_analysis.sparse_text_looks_noisy(text) or quality >= 0.10


def symbol_encoded_native_text_layer_looks_weak(
    page: Any,
    text: str,
) -> bool:
    tokens = ocr_text_analysis.extracted_text_token_count(text)
    if tokens < 240:
        return False
    if ocr_text_analysis.text_ocr_quality_score(text) < 0.40:
        return False
    return page_has_symbol_encoded_native_text_artifacts(page)


def has_meaningful_type3_encoding(font: Any) -> bool:
    encoding = get_candidate_services().pdf_analysis.lookup_dict_key(font, "Encoding")
    if not isinstance(encoding, dict):
        return False
    differences_obj = get_candidate_services().pdf_analysis.lookup_dict_key(encoding, "Differences")
    if not isinstance(differences_obj, (list, tuple)):
        return False
    try:
        pdf_analysis = get_candidate_services().pdf_analysis
        differences = pdf_analysis.parse_differences(
            list(differences_obj), pdf_analysis.normalize_pdf_name
        )
    except ValueError:
        return False
    for glyph_name in differences.values():
        mapped = get_candidate_services().pdf_analysis.glyph_name_to_unicode(glyph_name)
        if mapped and mapped != glyph_name:
            return True
    return False
