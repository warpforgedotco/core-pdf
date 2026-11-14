# SPDX-License-Identifier: AGPL-3.0-only

"""Check every path that blends against the specification's blend functions, byte for byte.

core_pdf.impl.spec.s_11_transparency.blend is the reference implementation of the sixteen
standard blend modes (ISO 32000-2 11.3.5). The renderer blends in six places: the compiled
blend_coverage_counts, shading_blend, blend_visible_rgba and composite_nonisolated_blend,
and the Python blend_px and blend_channels_f64. For every mode, and for ColorDodge and
ColorBurn with both the revised and the classic endpoints, this composites thousands of
random backdrop and source colours and alphas, and edge cases (0, 1, 254 and 255, equal
colours, grays, saturated colours), through each path, and compares the bytes with a
compositing written here from the reference: cs' = (1 - ab) cs + ab B(cb, cs) and
C = (cs' as + cb ab (1 - as)) / ar, as each path computes it. It also checks the scalar
reference against the numpy one bit for bit, that each path's mode table agrees with
BLEND_MODES, and that the kernels refuse a code they do not know.

The same paths blend in the DeviceGray and DeviceCMYK blending spaces of transparency
groups (core_pdf.impl.render.raster_spaces): a gray group holds its gray level in all three
components, and a CMYK group paints twice, once holding the complements of C, M and Y and
once the complement of K, the K pass in BLACK_BLEND_OPS's op. For every mode, gray and CMYK
backdrops and sources go through each path that way, and the bytes they give back are
compared with a compositing of each component written from the specification's
blending_space.additive_blend.

uv run --all-packages python scripts/check_blend_modes.py [random cases per mode]
"""

from __future__ import annotations

import itertools
import struct
import sys
from collections.abc import Callable, Iterable
from types import SimpleNamespace
from typing import Any

import numpy

from core_pdf.impl.render.blend import (
    BLACK_BLEND_OPS,
    BLEND_OPS,
    KERNEL_BLEND_CODES,
    BlendOp,
    blend_channels_f64,
    blend_op,
    kernel_blend_code,
)
from core_pdf.impl.render.raster_blend import blend_px
from core_pdf.impl.spec.s_11_transparency.blend import (
    BLEND_MODES,
    NON_SEPARABLE_BLEND_MODES,
    BlendMode,
    blend_color,
    blend_colors,
    first_recognized_blend_mode,
)
from core_pdf.impl.spec.s_11_transparency.blending_space import (
    BlendingSpace,
    additive_blend,
    additive_components,
    blend_in_space,
)
from core_pdf.impl.spec.standards import PdfVersion, SemanticContext
from core_pdf_cythonized import (
    blend_coverage_counts,
    blend_visible_rgba,
    composite_nonisolated_blend,
    shading_blend,
)

CONTEXTS = {True: SemanticContext(PdfVersion(2, 0)), False: SemanticContext(PdfVersion(1, 7))}
EDGE_COMPONENTS = (0, 1, 2, 63, 64, 127, 128, 191, 192, 253, 254, 255)
EDGE_ALPHAS = (1, 2, 127, 128, 254, 255)
EDGE_BACKDROP_ALPHAS = (0, 1, 128, 254, 255)

type Pixel = tuple[int, int, int, int]
type Rgb = tuple[float, float, float]
type Case = tuple[Pixel, Pixel]


def edge_colors() -> list[tuple[int, int, int]]:
    colors: set[tuple[int, int, int]] = set()
    near = (0, 1, 128, 254, 255)
    colors.update((r, g, b) for r in near for g in near for b in near)  # saturated and near
    colors.update((v, v, v) for v in EDGE_COMPONENTS)  # grays
    colors.update((v, 0, 0) for v in EDGE_COMPONENTS)
    colors.update((0, v, 255) for v in EDGE_COMPONENTS)
    colors.update({(255, 255, 0), (255, 0, 255), (0, 255, 255), (64, 128, 192), (200, 100, 50)})
    return sorted(colors)


def cases(count: int, seed: int) -> list[Case]:
    """(backdrop RGBA, source RGBA) pairs, the source alpha never 0."""
    random = numpy.random.default_rng(seed)
    colors = edge_colors()
    sources = colors[:: max(1, len(colors) // 24)]
    found: list[Case] = [
        ((*backdrop, 255), (*source, 255)) for backdrop in colors for source in sources
    ]
    found.extend(((*color, 255), (*color, 255)) for color in colors)  # equal colours
    found.extend(
        ((*color, backdrop_alpha), (*opposite, source_alpha))
        for color, opposite in zip(colors, reversed(colors), strict=True)
        for backdrop_alpha, source_alpha in itertools.product(EDGE_BACKDROP_ALPHAS, EDGE_ALPHAS)
    )
    values = random.integers(0, 256, size=(count, 8))
    values[:, 7] = numpy.maximum(values[:, 7], 1)
    # A quarter of the random cases with an opaque backdrop, as on a page, and a few grays.
    quarter, sixteenth = count // 4, count // 16
    values[:quarter, 3] = 255
    values[quarter : quarter + sixteenth, 1:3] = values[quarter : quarter + sixteenth, :1]
    found.extend(
        ((int(v[0]), int(v[1]), int(v[2]), int(v[3])), (int(v[4]), int(v[5]), int(v[6]), int(v[7])))
        for v in values
    )
    return found


def channel(value: float) -> int:
    # rint, then the clamp to a byte every path applies.
    return max(0, min(255, round(value)))


def composited(backdrop: Pixel, color: Rgb, src_a: float, mode: BlendMode, revised: bool) -> Pixel:
    """color, of alpha src_a, over backdrop: (1 - ab) cs + ab B(cb, cs) as the source colour
    (cs itself for Normal), then the compositing formula."""
    dst_a = backdrop[3] / 255.0
    out_a = src_a + dst_a * (1.0 - src_a)
    if out_a <= 0.0:
        return 0, 0, 0, 0
    source = color
    if mode != "Normal":
        blended = blend_color(
            (backdrop[0] / 255.0, backdrop[1] / 255.0, backdrop[2] / 255.0),
            color,
            mode,
            context=CONTEXTS[revised],
        )
        source = (
            color[0] * (1.0 - dst_a) + dst_a * blended[0],
            color[1] * (1.0 - dst_a) + dst_a * blended[1],
            color[2] * (1.0 - dst_a) + dst_a * blended[2],
        )
    red, green, blue = (
        channel(((source[k] * 255.0) * src_a + backdrop[k] * dst_a * (1.0 - src_a)) / out_a)
        for k in range(3)
    )
    return red, green, blue, channel(out_a * 255.0)


def expected(case: Case, mode: BlendMode, revised: bool) -> Pixel:
    backdrop, source = case
    color = (source[0] / 255.0, source[1] / 255.0, source[2] / 255.0)
    return composited(backdrop, color, source[3] / 255.0, mode, revised)


def backdrop_removed(
    color: float, complete: float, backdrop: float, initial: float, accumulated: float
) -> float:
    premultiplied = color * complete - backdrop * ((1.0 - accumulated) * initial)
    removed = premultiplied / accumulated if accumulated != 0.0 else 0.0
    return min(1.0, max(0.0, removed))


def expected_nonisolated(case: Case, accumulated: float, mode: BlendMode, revised: bool) -> Pixel:
    """A non-isolated group's pixel: the group backdrop removed (PDF 2.0, 11.4.4), clipped,
    then composited at the quantized effective alpha."""
    backdrop, source = case
    level = int(numpy.rint(accumulated * 1.0 * 255.0))
    if level == 0:
        return backdrop
    dst_a = backdrop[3] / 255.0
    complete = source[3] / 255.0
    red, green, blue = (
        backdrop_removed(source[k] / 255.0, complete, backdrop[k] / 255.0, dst_a, accumulated)
        for k in range(3)
    )
    return composited(backdrop, (red, green, blue), level / 255.0, mode, revised)


def pixels_of(rows: Iterable[Any]) -> list[Pixel]:
    return [(int(p[0]), int(p[1]), int(p[2]), int(p[3])) for p in rows]


def through_coverage(batch: list[Case], code: int, revised: bool) -> list[Pixel]:
    out: list[Pixel] = []
    counts = numpy.full((1, 1), 16, dtype=numpy.uint8)
    for backdrop, source in batch:
        pixels = numpy.array([[backdrop]], dtype=numpy.uint8)
        blend_coverage_counts(
            pixels, counts, 0, 0, None, *source, None, None, False, 1.0, code, revised, False
        )
        out.extend(pixels_of(pixels[0]))
    return out


def through_shading(batch: list[Case], code: int, revised: bool) -> list[Pixel]:
    pixels = numpy.array([[backdrop for backdrop, _ in batch]], dtype=numpy.uint8)
    colors = numpy.array([source for _, source in batch], dtype=numpy.intc)
    painted = numpy.ones((1, len(batch)), dtype=numpy.uint8)
    index = numpy.arange(len(batch), dtype=numpy.longlong)
    shading_blend(pixels, 0, 0, painted, index, colors, code, revised, None, None, 1.0, False)
    return pixels_of(pixels[0])


def through_visible(batch: list[Case], code: int, revised: bool) -> list[Pixel]:
    destination = numpy.array([[backdrop for backdrop, _ in batch]], dtype=numpy.uint8)
    sources = numpy.array([source for _, source in batch], dtype=numpy.float64) / 255.0
    visible = numpy.ones((1, len(batch)), dtype=numpy.uint8)
    red, green, blue, alpha = (numpy.ascontiguousarray(sources[:, k]) for k in range(4))
    blend_visible_rgba(destination, visible, red, green, blue, alpha, code, revised)
    return pixels_of(destination[0])


def through_nonisolated(
    batch: list[Case], accumulated: numpy.ndarray, code: int, revised: bool
) -> list[Pixel]:
    destination = numpy.array([[backdrop for backdrop, _ in batch]], dtype=numpy.uint8)
    rendered = numpy.array([[source for _, source in batch]], dtype=numpy.uint8)
    composite_nonisolated_blend(
        destination, rendered, accumulated.reshape(1, -1), 1.0, None, code, False, revised
    )
    return pixels_of(destination[0])


def through_blend_px(batch: list[Case], code: int, revised: bool) -> list[Pixel]:
    # blend_px reads only these attributes of its RasterTarget when no plane is recorded.
    target: Any = SimpleNamespace(
        pixels=bytearray(numpy.array([b for b, _ in batch], dtype=numpy.uint8).tobytes()),
        group_source_alpha=None,
        group_source_shape=None,
        paint_window=None,
        width=len(batch),
        semantic_context=CONTEXTS[revised],
    )
    for index, (_, source) in enumerate(batch):
        blend_px(target, index * 4, source, BlendOp(code))
    data = target.pixels
    return pixels_of(data[i : i + 4] for i in range(0, len(data), 4))


def through_f64(batch: list[Case], code: int, revised: bool) -> list[Pixel]:
    backdrops = numpy.array([b for b, _ in batch], dtype=numpy.float64)
    sources = numpy.array([s for _, s in batch], dtype=numpy.float64) / 255.0
    channels = blend_channels_f64(
        sources[:, 0],
        sources[:, 1],
        sources[:, 2],
        sources[:, 3],
        backdrops[:, 0],
        backdrops[:, 1],
        backdrops[:, 2],
        backdrops[:, 3],
        BlendOp(code),
        semantic_context=CONTEXTS[revised],
    )
    return pixels_of(numpy.clip(numpy.column_stack(channels), 0, 255).astype(numpy.uint8))


PATHS: dict[str, Callable[[list[Case], int, bool], list[Pixel]]] = {
    "blend_coverage_counts": through_coverage,
    "shading_blend": through_shading,
    "blend_visible_rgba": through_visible,
    "blend_px": through_blend_px,
    "blend_channels_f64": through_f64,
}


def bits(values: Iterable[float]) -> bytes:
    return b"".join(struct.pack(">d", float(v)) for v in values)


def modes_and_revisions() -> list[tuple[BlendMode, bool]]:
    return [(mode, True) for mode in BLEND_MODES] + [("ColorDodge", False), ("ColorBurn", False)]


def check_references(batch: list[Case], failures: list[str]) -> None:
    """The scalar and numpy references agree bit for bit, on byte colours and on random
    floats."""
    random = numpy.random.default_rng(7)
    floats = random.random((2000, 6))
    floats[:200] = numpy.round(floats[:200] * 4) / 4  # 0, 0.25, 0.5, 0.75, 1: the branch points
    floats[200:300, 1:3] = floats[200:300, :1]
    floats[300:400, 4:6] = floats[300:400, 3:4]
    byte_colors = numpy.array([(*b[:3], *s[:3]) for b, s in batch], dtype=numpy.float64) / 255.0
    samples = numpy.vstack((floats, byte_colors))
    rows: list[list[float]] = samples.tolist()
    for mode, revised in modes_and_revisions():
        context = CONTEXTS[revised]
        backdrop = (samples[:, 0], samples[:, 1], samples[:, 2])
        source = (samples[:, 3], samples[:, 4], samples[:, 5])
        vector = blend_colors(backdrop, source, mode, context=context)
        scalar = [
            blend_color((r[0], r[1], r[2]), (r[3], r[4], r[5]), mode, context=context) for r in rows
        ]
        if any(bits(color[k] for color in scalar) != bits(vector[k].tolist()) for k in range(3)):
            failures.append(f"{mode} (revised={revised}): blend_colors differs from blend_color")


def check_tables(failures: list[str]) -> None:
    for code, mode in enumerate(BLEND_MODES):
        op = BlendOp(code)
        if BLEND_OPS[mode.lower()] is not op or blend_op(mode) is not op:
            failures.append(f"{mode}: BlendOp {BLEND_OPS.get(mode.lower())} is not code {code}")
        if kernel_blend_code(op) != code or KERNEL_BLEND_CODES[op] != code:
            failures.append(f"{mode}: kernel code {kernel_blend_code(op)} is not {code}")
    failures.extend(
        f"{name} does not paint as Normal"
        for name in ("Compatible", "Unknown", "Normal")
        if kernel_blend_code(blend_op(name)) != 0
    )
    if first_recognized_blend_mode(["Foo", None, "Darken", "Multiply"]) != "Darken":
        failures.append("an array of blend modes does not select the first recognized one")
    if first_recognized_blend_mode(["Foo"]) != "Normal":
        failures.append("an array of unrecognized blend modes does not select Normal")
    case: list[Case] = [((10, 20, 30, 255), (40, 50, 60, 255))]
    kernels: dict[str, Callable[[int], object]] = {
        "blend_coverage_counts": lambda code: through_coverage(case, code, True),
        "shading_blend": lambda code: through_shading(case, code, True),
        "blend_visible_rgba": lambda code: through_visible(case, code, True),
        "composite_nonisolated_blend": lambda code: through_nonisolated(
            case, numpy.ones(1, numpy.float32), code, True
        ),
    }
    for name, op in (("BACKDROP", BlendOp.BACKDROP), ("SOURCE", BlendOp.SOURCE)):
        if KERNEL_BLEND_CODES[op] != op.value or kernel_blend_code(op) != op.value:
            failures.append(f"{name}: kernel code {kernel_blend_code(op)} is not {op.value}")
    for op, black in BLACK_BLEND_OPS.items():
        mode = BLEND_MODES[op]
        if mode not in NON_SEPARABLE_BLEND_MODES or black not in {BlendOp.BACKDROP, BlendOp.SOURCE}:
            failures.append(f"{mode}: K blends as {black.name}")
    for code in (-1, BlendOp.SOURCE.value + 1, 255):
        for name, call in kernels.items():
            try:
                call(code)
            except ValueError:
                continue
            failures.append(f"{name} accepted blend code {code}")


def compare(
    name: str,
    mode: BlendMode,
    revised: bool,
    batch: list[Case],
    got: list[Pixel],
    want: list[Pixel],
    failures: list[str],
) -> None:
    for case, value, reference in zip(batch, got, want, strict=True):
        if value != reference:
            failures.append(
                f"{name} {mode} (revised={revised}): backdrop {case[0]} source {case[1]} "
                f"gave {value}, the reference {reference}"
            )
            return


type SpacePixel = tuple[tuple[int, ...], int]
type SpaceCase = tuple[SpacePixel, SpacePixel]


def space_cases(space: BlendingSpace, count: int, seed: int) -> list[SpaceCase]:
    """(backdrop, source) pairs of a blending space's colours, each its components as bytes
    (colorant amounts for CMYK) and an alpha, the source alpha never 0."""
    random = numpy.random.default_rng(seed)
    n = 1 if space == "DeviceGray" else 4
    levels = (0, 1, 127, 128, 254, 255)
    edges = [tuple(int(v) for v in row) for row in itertools.product(levels, repeat=n)]
    if n == 4:
        edges = edges[:: max(1, len(edges) // 160)]
    found: list[SpaceCase] = [
        ((color, 255), (other, alpha))
        for color in edges
        for other, alpha in zip(edges[::7], itertools.cycle(EDGE_ALPHAS), strict=False)
    ]
    found.extend(((color, 255), (color, 255)) for color in edges)
    found.extend(
        ((color, backdrop_alpha), (edges[-1 - index], source_alpha))
        for index, color in enumerate(edges)
        for backdrop_alpha, source_alpha in itertools.product(EDGE_BACKDROP_ALPHAS, (1, 128, 255))
    )
    values = random.integers(0, 256, size=(count, 2 * n + 2))
    values[:, -1] = numpy.maximum(values[:, -1], 1)
    values[: count // 4, n] = 255
    found.extend(
        (
            (tuple(int(v) for v in row[:n]), int(row[n])),
            (tuple(int(v) for v in row[n + 1 : 2 * n + 1]), int(row[-1])),
        )
        for row in values
    )
    return found


def space_planes(space: BlendingSpace, case: SpaceCase) -> list[Case]:
    """A case as the (backdrop, source) RGBA pairs each pass of its group composites: a gray
    level in all three components, or a CMYK colour's complemented C, M, Y, and then its
    complemented K three times."""
    (backdrop, backdrop_alpha), (source, source_alpha) = case
    if space == "DeviceGray":
        gray, level = backdrop[0], source[0]
        return [((gray, gray, gray, backdrop_alpha), (level, level, level, source_alpha))]
    cb = [255 - v for v in backdrop]
    cs = [255 - v for v in source]
    return [
        ((cb[0], cb[1], cb[2], backdrop_alpha), (cs[0], cs[1], cs[2], source_alpha)),
        ((cb[3], cb[3], cb[3], backdrop_alpha), (cs[3], cs[3], cs[3], source_alpha)),
    ]


def from_planes(space: BlendingSpace, planes: list[Pixel]) -> SpacePixel:
    """The colour the passes give back, as components and an alpha; -1 for an alpha the
    passes disagree on."""
    if space == "DeviceGray":
        return (planes[0][0],), planes[0][3]
    cmy, black = planes
    alpha = cmy[3] if cmy[3] == black[3] else -1
    return (255 - cmy[0], 255 - cmy[1], 255 - cmy[2], 255 - black[0]), alpha


def expected_in_space(
    space: BlendingSpace, case: SpaceCase, mode: BlendMode, revised: bool, removed: float | None
) -> SpacePixel:
    """The reference: each component of the additive forms composited as composited does,
    with B from additive_blend; for removed, a non-isolated group's backdrop removed first, as
    expected_nonisolated does."""
    (backdrop, backdrop_alpha), (source, source_alpha) = case
    count = len(backdrop)
    cb_bytes = [255 - v for v in backdrop] if space == "DeviceCMYK" else list(backdrop)
    cs = tuple((255 - v if space == "DeviceCMYK" else v) / 255.0 for v in source)
    dst_a = backdrop_alpha / 255.0
    src_a = source_alpha / 255.0
    if removed is not None:
        level = int(numpy.rint(removed * 1.0 * 255.0))
        if level == 0:
            return backdrop, backdrop_alpha
        cs = tuple(
            backdrop_removed(cs[k], src_a, cb_bytes[k] / 255.0, dst_a, removed)
            for k in range(count)
        )
        src_a = level / 255.0
    out_a = src_a + dst_a * (1.0 - src_a)
    if out_a <= 0.0:
        return (0,) * count, 0
    color = cs
    if mode != "Normal":
        cb = tuple(v / 255.0 for v in cb_bytes)
        blended = additive_blend(cb, cs, mode, space, revised)
        color = tuple(cs[k] * (1.0 - dst_a) + dst_a * blended[k] for k in range(count))
    out = [
        channel(((color[k] * 255.0) * src_a + cb_bytes[k] * dst_a * (1.0 - src_a)) / out_a)
        for k in range(count)
    ]
    if space == "DeviceCMYK":
        out = [255 - v for v in out]
    return tuple(out), channel(out_a * 255.0)


def check_space_references(failures: list[str]) -> None:
    """blend_in_space is additive_blend between complements, and in CMYK takes K as 11.3.5.3
    has it."""
    random = numpy.random.default_rng(11)
    for _ in range(400):
        backdrop = tuple(float(v) for v in random.random(4))
        source = tuple(float(v) for v in random.random(4))
        for mode in BLEND_MODES:
            colorants = blend_in_space(backdrop, source, mode, "DeviceCMYK")
            additive = additive_blend(
                additive_components("DeviceCMYK", backdrop),
                additive_components("DeviceCMYK", source),
                mode,
                "DeviceCMYK",
            )
            if colorants != additive_components("DeviceCMYK", additive):
                failures.append(f"{mode}: blend_in_space is not additive_blend complemented")
                return
            if mode in NON_SEPARABLE_BLEND_MODES:
                black = source[3] if mode == "Luminosity" else backdrop[3]
                if colorants[3] != black:
                    failures.append(f"{mode}: CMYK K is {colorants[3]}, not {black}")
                    return


def through_space_path(
    name: str, cases: list[Case], code: int, revised: bool, removed: numpy.ndarray
) -> list[Pixel]:
    if name == "composite_nonisolated_blend":
        return through_nonisolated(cases, removed, code, revised)
    return PATHS[name](cases, code, revised)


def check_spaces(count: int, accumulated: numpy.ndarray, failures: list[str]) -> int:
    checked = 0
    for space in ("DeviceGray", "DeviceCMYK"):
        batch = space_cases(space, count, 20261115)
        removed = numpy.resize(accumulated, len(batch))
        planes = [space_planes(space, case) for case in batch]
        checked += len(batch)
        for mode, revised in modes_and_revisions():
            op = BlendOp(BLEND_MODES.index(mode))
            codes = [op.value, BLACK_BLEND_OPS.get(op, op).value]
            want = [expected_in_space(space, case, mode, revised, None) for case in batch]
            want_removed = [
                expected_in_space(space, case, mode, revised, float(alpha))
                for case, alpha in zip(batch, removed, strict=True)
            ]
            for name in (*PATHS, "composite_nonisolated_blend"):
                passes = [
                    through_space_path(
                        name, [rows[index] for rows in planes], codes[index], revised, removed
                    )
                    for index in range(len(planes[0]))
                ]
                got = [
                    from_planes(space, [plane[row] for plane in passes])
                    for row in range(len(batch))
                ]
                reference = want_removed if name == "composite_nonisolated_blend" else want
                for case, value, expected_value in zip(batch, got, reference, strict=True):
                    if value != expected_value:
                        failures.append(
                            f"{space} {name} {mode} (revised={revised}): backdrop {case[0]} "
                            f"source {case[1]} gave {value}, the reference {expected_value}"
                        )
                        break
    return checked


def main(argv: list[str]) -> int:
    count = int(argv[0]) if argv else 4000
    batch = cases(count, 20261114)
    random = numpy.random.default_rng(3)
    accumulated = random.random(len(batch)).astype(numpy.float32)
    accumulated[:64] = numpy.float32(1.0)
    accumulated[64:96] = numpy.float32(0.001)
    failures: list[str] = []
    check_tables(failures)
    check_references(batch, failures)
    for mode, revised in modes_and_revisions():
        code = BLEND_MODES.index(mode)
        want = [expected(case, mode, revised) for case in batch]
        for name, path in PATHS.items():
            compare(name, mode, revised, batch, path(batch, code, revised), want, failures)
        want_nonisolated = [
            expected_nonisolated(case, float(alpha), mode, revised)
            for case, alpha in zip(batch, accumulated, strict=True)
        ]
        got = through_nonisolated(batch, accumulated, code, revised)
        compare(
            "composite_nonisolated_blend", mode, revised, batch, got, want_nonisolated, failures
        )
    check_space_references(failures)
    in_spaces = check_spaces(count // 2, accumulated, failures)
    separable = len(BLEND_MODES) - len(NON_SEPARABLE_BLEND_MODES)
    print(
        f"{len(BLEND_MODES)} blend modes ({separable} separable), "
        f"{len(batch)} RGB cases each through {len(PATHS) + 1} paths, and {in_spaces} gray "
        f"and CMYK cases through the passes of their groups: {len(failures)} failures"
    )
    for failure in failures[:20]:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
