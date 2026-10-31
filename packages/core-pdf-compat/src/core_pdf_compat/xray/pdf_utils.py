# Ported from x-ray, Copyright (c) Free Law Project, under the BSD 2-Clause License.
from typing import Any

Rect = Any


def intersects(
    text_rect: Rect,
    rectangles: list[Rect],
    occlusion_threshold: float = 0.0,
) -> bool:
    """Determine if a rectangle intersects is occluded by a list of others

    This uses Rect objects, but note that they must have extra attributes of
    "fill" and "seqno".

    :param text_rect: The rectangle around the text to check for intersections.
    :param rectangles: A list of rectangles to check for intersections.
    :param occlusion_threshold: How much the rectangle must be occluded by at
    least one of the rectangles for it to be considered an intersection, as a
    percentage. E.g., 1.0 means that the bbox must be fully occluded, 0.10
    means it must be 10% occluded. The default, 0.0, means they must intersect
    at least a little.
    :return True if any part of the bbox intersects with any of the rectangles,
    else False.
    """
    for rect in rectangles + [text_rect]:
        assert all([hasattr(rect, "seqno"), hasattr(rect, "fill")]), (
            "Rectangle lacks required 'seqno' or 'fill' attribute."
        )

    overlapping_areas = []
    for rect in rectangles:
        intersecting_area = abs(text_rect & rect)
        if intersecting_area > 0 and rect.seqno > text_rect.seqno:
            # Intersecting text was drawn first, meaning it's behind the rect.
            overlapping_areas.append(intersecting_area)
            continue
        if intersecting_area > 0 and rect.fill == text_rect.fill:
            # Intersecting and same color. This makes text invisible even if
            # it's drawn on top of the rect.
            overlapping_areas.append(intersecting_area)
            continue

    if not overlapping_areas:
        return False

    greatest_occluded = max(overlapping_areas)
    area_of_bbox = abs(text_rect.get_area())

    percent_occluded = greatest_occluded / area_of_bbox
    return percent_occluded > occlusion_threshold


def _needs_mupdf(name: str) -> NotImplementedError:
    return NotImplementedError(f"{name} reads PyMuPDF pages; use xray.inspect instead")


def get_good_rectangles(page: Any) -> list[Rect]:
    raise _needs_mupdf("get_good_rectangles")


def get_intersecting_chars(page: Any, rectangles: list[Rect]) -> list[Any]:
    raise _needs_mupdf("get_intersecting_chars")


def get_bad_redactions(page: Any) -> list[Any]:
    raise _needs_mupdf("get_bad_redactions")


__all__ = ("get_bad_redactions", "get_good_rectangles", "get_intersecting_chars", "intersects")
