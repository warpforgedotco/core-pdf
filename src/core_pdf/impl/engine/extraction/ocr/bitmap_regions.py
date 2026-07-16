from __future__ import annotations

from collections import deque

from core_pdf.impl.engine.extraction.ocr.types import OcrImage


def bitmap_vertical_regions(
    image: OcrImage,
    *,
    max_regions: int = 8,
    grid_width: int = 600,
) -> tuple[tuple[int, int, int, int], ...]:
    """Find tall text-shaped bitmap components in source image coordinates."""
    if image.bytes_per_pixel not in {1, 3, 4} or not image.data:
        return ()
    scale = max(1, (image.width + grid_width - 1) // grid_width)
    width = (image.width + scale - 1) // scale
    height = (image.height + scale - 1) // scale
    foreground = bytearray(width * height)
    for gy in range(height):
        for gx in range(width):
            source = min(image.height - 1, gy * scale) * image.bytes_per_line
            source += min(image.width - 1, gx * scale) * image.bytes_per_pixel
            if image.bytes_per_pixel == 1:
                gray = image.data[source]
            else:
                gray = (
                    image.data[source] * 30
                    + image.data[source + 1] * 59
                    + image.data[source + 2] * 11
                ) // 100
            foreground[gy * width + gx] = gray < 210
    seen = bytearray(width * height)
    regions: list[tuple[int, int, int, int]] = []
    for start in range(width * height):
        if not foreground[start] or seen[start]:
            continue
        queue = deque([start])
        seen[start] = 1
        min_x = max_x = start % width
        min_y = max_y = start // width
        count = 0
        while queue:
            point = queue.popleft()
            x, y = point % width, point // width
            count += 1
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < width and 0 <= ny < height:
                    neighbor = ny * width + nx
                    if foreground[neighbor] and not seen[neighbor]:
                        seen[neighbor] = 1
                        queue.append(neighbor)
        component_width = max_x - min_x + 1
        component_height = max_y - min_y + 1
        if count < 12 or component_height < component_width * 4:
            continue
        regions.append(
            (
                min_x * scale,
                min_y * scale,
                min(image.width, (max_x + 1) * scale),
                min(image.height, (max_y + 1) * scale),
            )
        )
    regions.sort(key=lambda region: (region[0], region[1]))
    return tuple(regions[:max_regions])
