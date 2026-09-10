# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2004-2026 Artifex Software, Inc.
# Python adaptation of MuPDF 1.28.2 shade.c and draw-mesh.c.
# See README.md for the exact upstream files and adaptation boundary.
"""Rasterize axial/radial shading parameters into byte lookup-table indices.

This module has no PDF parser, color converter, or core-pdf dependency. The
caller supplies geometry and consumes the index/coverage planes. Single
precision and fixed-point edge stepping preserve the upstream sampling policy.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence

import numpy

Vertex = tuple[float, float, float]
Point = tuple[float, float]


def f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def transform(point: Point, matrix: Sequence[float], *, vector: bool = False) -> Point:
    x, y = point
    a, b, c, d, e, f = matrix
    return (
        f32(f32(f32(x * a) + f32(y * c)) + (0 if vector else e)),
        f32(f32(f32(x * b) + f32(y * d)) + (0 if vector else f)),
    )


def circle(point: Point, radius: float, theta: float) -> Point:
    # Preserve the fused multiply-add used for circle vertices in the reference
    # build. An extra product rounding can move an entire scan edge by a pixel.
    return (
        f32(math.fma(f32(math.cos(theta)), radius, point[0])),
        f32(math.fma(f32(math.sin(theta)), radius, point[1])),
    )


class Edge:
    def __init__(self, top: Vertex, bottom: Vertex, y: int) -> None:
        inverse = f32(1.0 / f32(bottom[1] - top[1]))
        offset = f32(f32(y - top[1]) * inverse)
        dx = f32(bottom[0] - top[0])
        self.x = f32(top[0] + f32(dx * offset))
        self.dx = f32(dx * inverse)
        dc = f32(bottom[2] - top[2])
        self.value = int(f32(65536.0 * f32(top[2] + f32(dc * offset))))
        self.delta = int(f32(f32(65536.0 * dc) * inverse))

    def step(self) -> None:
        self.x = f32(self.x + self.dx)
        self.value += self.delta


class Raster:
    def __init__(self, bbox: tuple[int, int, int, int]) -> None:
        self.bbox = bbox
        x0, y0, x1, y1 = bbox
        self.indices = numpy.zeros((y1 - y0, x1 - x0), dtype=numpy.uint8)
        self.coverage = numpy.zeros_like(self.indices, dtype=numpy.bool_)

    def scan(self, y: int, edge0: Edge, edge1: Edge) -> None:
        fx0, fx1 = int(edge0.x), int(edge1.x)
        value0, value1 = edge0.value, edge1.value
        if fx0 > fx1:
            fx0, fx1, value0, value1 = fx1, fx0, value1, value0
        x0, x1 = max(fx0, self.bbox[0]), min(fx1, self.bbox[2])
        if x0 >= x1:
            return
        delta = int(f32(f32(value1 - value0) * f32(1.0 / (fx1 - fx0))))
        value = value0 + int(f32(f32(delta) * f32(x0 - fx0)))
        columns = numpy.arange(x1 - x0, dtype=numpy.int64)
        samples = (value + delta * columns) >> 16
        self.indices[y - self.bbox[1], x0 - self.bbox[0] : x1 - self.bbox[0]] = samples
        self.coverage[y - self.bbox[1], x0 - self.bbox[0] : x1 - self.bbox[0]] = True

    def triangle(self, *vertices: Vertex) -> None:
        top = bottom = 0
        if vertices[1][1] < vertices[0][1]:
            top = 1
        else:
            bottom = 1
        if vertices[2][1] < vertices[top][1]:
            top = 2
        elif vertices[2][1] > vertices[bottom][1]:
            bottom = 2
        if vertices[top][1] == vertices[bottom][1]:
            return
        if vertices[bottom][1] < self.bbox[1] or vertices[top][1] > self.bbox[3]:
            return
        middle = 3 ^ top ^ bottom
        y = math.ceil(max(self.bbox[1], vertices[top][1]))
        end = math.ceil(min(self.bbox[3], vertices[middle][1]))
        edge0 = Edge(vertices[top], vertices[bottom], y)
        if y < end:
            edge1 = Edge(vertices[top], vertices[middle], y)
            while y < end:
                self.scan(y, edge0, edge1)
                edge0.step()
                edge1.step()
                y += 1
        end = math.ceil(min(self.bbox[3], vertices[bottom][1]))
        if y < end:
            edge1 = Edge(vertices[middle], vertices[bottom], y)
            while y < end:
                self.scan(y, edge0, edge1)
                y += 1
                if y < end:
                    edge0.step()
                    edge1.step()

    def quad(self, a: Vertex, b: Vertex, c: Vertex, d: Vertex) -> None:
        self.triangle(a, b, d)
        self.triangle(d, c, b)

    def axial(
        self, coords: Sequence[float], matrix: Sequence[float], extend: tuple[bool, bool]
    ) -> None:
        p0 = transform((coords[0], coords[1]), matrix)
        p1 = transform((coords[2], coords[3]), matrix)
        direction = transform(
            (f32(coords[1] - coords[3]), f32(coords[2] - coords[0])), matrix, vector=True
        )
        theta = f32(math.atan2(direction[1], direction[0]))
        x0, y0, x1, y1 = self.bbox
        rx = max(f32(p0[0] - x0), f32(x1 - p0[0]), f32(p0[0] - x1), f32(x1 - p1[0]))
        ry = max(f32(p0[1] - y0), f32(y1 - p0[1]), f32(p0[1] - y1), f32(y1 - p1[1]))
        radius = f32(rx + ry)
        q0, q1 = circle(p0, radius, theta), circle(p1, radius, theta)
        v0, v1 = (*q0, 0.0), (*q1, 255.0)
        v2 = (f32(f32(2 * p0[0]) - q0[0]), f32(f32(2 * p0[1]) - q0[1]), 0.0)
        v3 = (f32(f32(2 * p1[0]) - q1[0]), f32(f32(2 * p1[1]) - q1[1]), 255.0)
        self.quad(v0, v2, v3, v1)
        dx, dy = f32(p1[0] - p0[0]), f32(p1[1] - p0[1])
        distance = max(abs(dx), abs(dy))
        if distance:
            radius = f32(radius / distance)
        if extend[0]:
            e0 = (f32(v0[0] - f32(dx * radius)), f32(v0[1] - f32(dy * radius)), 0.0)
            e1 = (f32(v2[0] - f32(dx * radius)), f32(v2[1] - f32(dy * radius)), 0.0)
            self.quad(e0, v0, v2, e1)
        if extend[1]:
            e0 = (f32(v1[0] + f32(dx * radius)), f32(v1[1] + f32(dy * radius)), 255.0)
            e1 = (f32(v3[0] + f32(dx * radius)), f32(v3[1] + f32(dy * radius)), 255.0)
            self.quad(e0, v1, v3, e1)

    def annulus(
        self,
        matrix: Sequence[float],
        p0: Point,
        r0: float,
        c0: float,
        p1: Point,
        r1: float,
        c1: float,
        count: int,
    ) -> None:
        theta = f32(math.atan2(f32(p1[1] - p0[1]), f32(p1[0] - p0[0])))
        step = f32(f32(math.pi) / count)
        angle = 0.0
        for index in range(1, count + 1):
            next_angle = f32(index * step)
            for sign in (1, -1):
                a, b = f32(theta + sign * angle), f32(theta + sign * next_angle)
                v0 = (*transform(circle(p0, r0, a), matrix), c0)
                v1 = (*transform(circle(p0, r0, b), matrix), c0)
                v2 = (*transform(circle(p1, r1, a), matrix), c1)
                v3 = (*transform(circle(p1, r1, b), matrix), c1)
                self.quad(v0, v2, v3, v1)
            angle = next_angle

    def radial(
        self, coords: Sequence[float], matrix: Sequence[float], extend: tuple[bool, bool]
    ) -> None:
        p0, p1 = (coords[0], coords[1]), (coords[3], coords[4])
        r0, r1 = coords[2], coords[5]
        a, b, c, d = matrix[:4]
        expansion = f32(math.sqrt(abs(f32(f32(a * d) - f32(b * c)))))
        count = min(1024, max(3, int(f32(4 * f32(math.sqrt(f32(expansion * max(r0, r1))))))))
        if extend[0]:
            ratio = f32(r0 / f32(r0 - r1)) if r0 < r1 else -32000.0
            end = (
                f32(p0[0] + f32(f32(p1[0] - p0[0]) * ratio)),
                f32(p0[1] + f32(f32(p1[1] - p0[1]) * ratio)),
            )
            radius = f32(r0 + f32(f32(r1 - r0) * ratio))
            self.annulus(matrix, end, radius, 0.0, p0, r0, 0.0, count)
        self.annulus(matrix, p0, r0, 0.0, p1, r1, 255.0, count)
        if extend[1]:
            ratio = f32(r1 / f32(r1 - r0)) if r0 > r1 else -32000.0
            end = (
                f32(p1[0] + f32(f32(p0[0] - p1[0]) * ratio)),
                f32(p1[1] + f32(f32(p0[1] - p1[1]) * ratio)),
            )
            radius = f32(r1 + f32(f32(r0 - r1) * ratio))
            self.annulus(matrix, p1, r1, 255.0, end, radius, 255.0, count)
