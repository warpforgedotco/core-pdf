# SPDX-License-Identifier: AGPL-3.0-only
"""Native page-coordinate transformation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core_pdf.impl.engine.model.runs import TextRun


def rotate_page_point(
    x: float,
    y: float,
    *,
    rotate: int,
    page_width: float,
    page_height: float,
) -> tuple[float, float]:
    match rotate:
        case 90:
            return (y, page_width - x)
        case 180:
            return (page_width - x, page_height - y)
        case 270:
            return (page_height - y, x)
        case _:
            return (x, y)


def rotate_page_runs(
    runs: list[TextRun],
    *,
    rotate: int,
    page_width: float,
    page_height: float,
) -> list[TextRun]:
    rotate %= 360
    if rotate == 0:
        return runs

    transformed: list[TextRun] = []
    for run in runs:
        points = [
            rotate_page_point(
                run.x0,
                run.y0,
                rotate=rotate,
                page_width=page_width,
                page_height=page_height,
            ),
            rotate_page_point(
                run.x0,
                run.y1,
                rotate=rotate,
                page_width=page_width,
                page_height=page_height,
            ),
            rotate_page_point(
                run.x1,
                run.y0,
                rotate=rotate,
                page_width=page_width,
                page_height=page_height,
            ),
            rotate_page_point(
                run.x1,
                run.y1,
                rotate=rotate,
                page_width=page_width,
                page_height=page_height,
            ),
        ]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        tx, ty = rotate_page_point(
            run.tx,
            run.ty,
            rotate=rotate,
            page_width=page_width,
            page_height=page_height,
        )
        transformed.append(
            run.replace(
                x0=min(xs),
                y0=min(ys),
                x1=max(xs),
                y1=max(ys),
                tx=tx,
                ty=ty,
                rotation_angle=(run.rotation_angle - rotate) % 360,
            )
        )
    return transformed


__all__ = (
    "rotate_page_point",
    "rotate_page_runs",
)
