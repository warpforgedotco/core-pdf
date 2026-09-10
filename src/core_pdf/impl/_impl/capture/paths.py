# SPDX-License-Identifier: AGPL-3.0-only
"""Project PDF paths using the capture pipeline's established flatness policy."""

from math import ceil, hypot

from core_pdf.impl._impl.capture.records import CapturedPath
from core_pdf.impl._impl.model.geometry import points_bbox
from core_pdf.impl.spec.s_07_content.paths import PdfPath
from core_pdf.impl.spec.s_08_graphics.matrix import Matrix
from core_pdf.impl.types import Rectangle


def control_point_bounds(source: PdfPath, matrix: Matrix) -> Rectangle | None:
    """Bound drawn segments and their Bézier controls without flattening them.

    These conservative bounds are useful to consumers that approximate clipping
    by the curve's control hull. An isolated or trailing move has no geometry.
    """
    points: list[tuple[float, float]] = []
    current: tuple[float, float] | None = None
    start: tuple[float, float] | None = None
    for command in source.commands:
        values = command.operands
        match command.operator:
            case "m":
                current = start = (values[0], values[1])
            case "l":
                if current is not None:
                    points.append(current)
                current = (values[0], values[1])
                points.append(current)
            case "c":
                points.extend(zip(values[::2], values[1::2], strict=True))
                current = (values[6], values[7])
            case "re":
                x, y, width, height = values
                points.extend(((x, y), (x + width, y), (x, y + height), (x + width, y + height)))
                current = start = (x, y)
            case "h":
                current = start
    a, b, c, d, e, f = matrix
    return points_bbox((x * a + y * c + e, x * b + y * d + f) for x, y in points)


def flatten_path(source: PdfPath) -> CapturedPath:
    path = CapturedPath()
    for command in source.commands:
        values = command.operands
        match command.operator:
            case "m":
                path.move_to(*values)
            case "l":
                path.line_to(*values)
            case "h":
                path.close()
            case "re":
                path.rect(*values)
            case "c":
                x0, y0, x1, y1, x2, y2, x3, y3 = values
                matrix = command.ctm
                scale = max(hypot(matrix.a, matrix.b), hypot(matrix.c, matrix.d), 1.0)
                control_len = (
                    hypot(x1 - x0, y1 - y0) + hypot(x2 - x1, y2 - y1) + hypot(x3 - x2, y3 - y2)
                )
                flatness = max(0.1, command.flatness or 0.25)
                segments = max(4, min(128, ceil(control_len * scale / (flatness * 8.0))))
                previous_x, previous_y = x0, y0
                segment_step = 1.0 / segments
                for i in range(1, segments + 1):
                    t = i * segment_step
                    mt = 1.0 - t
                    mt2 = mt * mt
                    t2 = t * t
                    b0, b1, b2, b3 = mt2 * mt, 3.0 * mt2 * t, 3.0 * mt * t2, t2 * t
                    x = b0 * x0 + b1 * x1 + b2 * x2 + b3 * x3
                    y = b0 * y0 + b1 * y1 + b2 * y2 + b3 * y3
                    if not path.subpaths:
                        path.move_to(previous_x, previous_y)
                    path.line_to(x, y)
                    previous_x, previous_y = x, y
    return path
