from __future__ import annotations

from core_pdf.impl.engine.extraction.ocr.types import OcrComponentBox


def vertical_regions_from_component_boxes(
    boxes: list[OcrComponentBox],
    *,
    image_width: int,
    image_height: int,
    max_regions: int = 8,
) -> tuple[tuple[int, int, int, int], ...]:
    """Group OCR component boxes into tall, narrow column regions."""
    if image_width <= 0 or image_height <= 0:
        return ()
    groups: list[list[OcrComponentBox]] = []
    for box in sorted(boxes, key=lambda item: item.left):
        center = box.left + box.width / 2
        target = next(
            (
                group
                for group in groups
                if abs(
                    center
                    - sum(item.left + item.width / 2 for item in group) / len(group)
                )
                <= max(16, box.width * 2)
            ),
            None,
        )
        if target is None:
            groups.append([box])
        else:
            target.append(box)
    regions: list[tuple[int, int, int, int]] = []
    for group in groups:
        left = max(0, min(item.left for item in group) - 8)
        top = max(0, min(item.top for item in group) - 8)
        right = min(image_width, max(item.left + item.width for item in group) + 8)
        bottom = min(image_height, max(item.top + item.height for item in group) + 8)
        width = right - left
        height = bottom - top
        if len(group) < 4 or height < 5 * width or height < image_height // 12:
            continue
        regions.append((left, top, right, bottom))
    regions.sort(key=lambda region: (region[0], region[1]))
    return tuple(regions[:max_regions])


def map_clockwise_rotated_regions_to_source(
    regions: tuple[tuple[int, int, int, int], ...],
    *,
    source_width: int,
    source_height: int,
) -> tuple[tuple[int, int, int, int], ...]:
    """Map boxes from a clockwise-rotated image back to source coordinates."""
    mapped: list[tuple[int, int, int, int]] = []
    for left, top, right, bottom in regions:
        mapped.append((top, source_height - right, bottom, source_height - left))
    return tuple(mapped)
