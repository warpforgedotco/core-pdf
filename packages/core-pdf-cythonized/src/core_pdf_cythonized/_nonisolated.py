# SPDX-License-Identifier: AGPL-3.0-only

import typing

import cython
import numpy
from cython.cimports.core_pdf_cythonized._nonisolated import OUT_OF_RANGE, VISIBLE
from cython.cimports.core_pdf_cythonized._pixel_blend import (
    MODE_COLOR_BURN,
    MODE_COLOR_DODGE,
    MODE_MULTIPLY,
    MODE_NORMAL,
    MODE_SCREEN,
    color_burn,
    color_dodge,
)
from cython.cimports.libc.math import rint
from cython.cimports.libc.stdint import uint32_t, uint64_t
from cython.cimports.libc.string import memcpy, memset

__all__ = ("composite_nonisolated_blend", "remove_group_backdrop_samples")


# Multiplicative hashing of the result cache's key; the top ten bits pick one
# of its 1024 slots.
HASH_SOURCE = cython.declare(uint32_t, 0x9E3779B1)
HASH_BACKDROP = cython.declare(uint32_t, 0x85EBCA77)
HASH_ALPHA = cython.declare(uint32_t, 0xC2B2AE3D)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def backdrop_removed(
    color: cython.double,
    complete: cython.double,
    backdrop: cython.double,
    initial: cython.double,
    accumulated: cython.double,
) -> cython.double:
    """One component of core_pdf_spec's remove_group_backdrop, which stays the
    definition: the same float64 operations in the same order."""
    backdrop_weight: cython.double = (1.0 - accumulated) * initial
    premultiplied: cython.double = color * complete - backdrop * backdrop_weight
    if accumulated != 0.0:
        return premultiplied / accumulated
    return 0.0


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def unit_clip(value: cython.double) -> cython.double:
    # numpy.clip(value, 0.0, 1.0), NaN passing through.
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def channel_byte(value: cython.double) -> cython.uchar:
    if value < 0.0:
        value = 0.0
    elif value > 255.0:
        value = 255.0
    return cython.cast(cython.uchar, value)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def blended(
    source: cython.double,
    backdrop: cython.double,
    dst_a: cython.double,
    mode: cython.int,
    revised: cython.bint,
) -> cython.double:
    # backdrop is the destination channel already divided by 255.0.
    if mode == MODE_MULTIPLY:
        return source * (1.0 - dst_a) + dst_a * (source * backdrop)
    if mode == MODE_SCREEN:
        return source * (1.0 - dst_a) + dst_a * (1.0 - (1.0 - source) * (1.0 - backdrop))
    if mode == MODE_COLOR_DODGE:
        return source * (1.0 - dst_a) + dst_a * color_dodge(backdrop, source, revised)
    if mode == MODE_COLOR_BURN:
        return source * (1.0 - dst_a) + dst_a * color_burn(backdrop, source, revised)
    return source


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def float_at(row: cython.p_const_char, offset: cython.Py_ssize_t) -> cython.double:
    return cython.cast(cython.pointer[cython.const[cython.float]], row + offset)[0]


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def load_pixel(pixel: cython.p_const_uchar, channel: cython.Py_ssize_t) -> uint32_t:
    # The four channels as one word, in memory order; only ever compared,
    # stored back with store_pixel, or split back into bytes with memcpy.
    packed = cython.declare(uint32_t)
    gathered = cython.declare(cython.uchar[4])
    if channel == 1:
        memcpy(cython.address(packed), pixel, 4)
        return packed
    gathered[0] = pixel[0]
    gathered[1] = pixel[channel]
    gathered[2] = pixel[2 * channel]
    gathered[3] = pixel[3 * channel]
    memcpy(cython.address(packed), gathered, 4)
    return packed


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def store_pixel(pixel: cython.p_uchar, channel: cython.Py_ssize_t, packed: uint32_t) -> cython.void:
    scattered = cython.declare(cython.uchar[4])
    if channel == 1:
        memcpy(pixel, cython.address(packed), 4)
        return
    memcpy(scattered, cython.address(packed), 4)
    pixel[0] = scattered[0]
    pixel[channel] = scattered[1]
    pixel[2 * channel] = scattered[2]
    pixel[3 * channel] = scattered[3]


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def composite_pixel(
    source_pixel: uint32_t,
    backdrop_pixel: uint32_t,
    level: cython.uchar,
    accumulated: cython.double,
    unit: cython.p_const_double,
    mode: cython.int,
    revised: cython.bint,
) -> uint32_t:
    """One visible pixel: the group backdrop removed, clipped, and blended over
    the destination in blend_visible_pixels' order. unit[x] is x / 255.0."""
    s = cython.declare(cython.uchar[4])
    b = cython.declare(cython.uchar[4])
    o = cython.declare(cython.uchar[4])
    memcpy(s, cython.address(source_pixel), 4)
    memcpy(b, cython.address(backdrop_pixel), 4)
    dst_a: cython.double = unit[b[3]]
    complete: cython.double = unit[s[3]]
    src_a: cython.double = unit[level]
    one_minus_src_a: cython.double = 1.0 - src_a
    out_a: cython.double = src_a + dst_a * one_minus_src_a
    color: cython.double
    k: cython.int
    if out_a <= 0.0:
        return 0
    for k in range(3):
        color = unit_clip(backdrop_removed(unit[s[k]], complete, unit[b[k]], dst_a, accumulated))
        color = blended(color, unit[b[k]], dst_a, mode, revised)
        o[k] = channel_byte(
            rint(((color * 255.0) * src_a + b[k] * dst_a * one_minus_src_a) / out_a)
        )
    o[3] = channel_byte(rint(out_a * 255.0))
    packed = cython.declare(uint32_t)
    memcpy(cython.address(packed), o, 4)
    return packed


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def quantize(level: cython.p_uchar, scaled: cython.double) -> cython.int:
    """Store numpy's uint8 cast of rint(scaled) and report on it: VISIBLE if
    it is not zero, OUT_OF_RANGE if rint(scaled) does not fit a C int, where
    that cast is left to the platform. A NaN casts to 0 on every platform
    numpy runs on. Branch-free, so that the row loops vectorize."""
    limit: cython.double = 2147483648.0
    rounded: cython.double = rint(scaled)
    fits: cython.bint = (rounded >= -limit) & (rounded < limit)
    value: cython.uchar = cython.cast(
        cython.uchar, cython.cast(cython.int, rounded if fits else 0.0)
    )
    level[0] = value
    return (VISIBLE if value != 0 else 0) | (
        OUT_OF_RANGE if (not fits) & (rounded == rounded) else 0
    )


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def quantize_row(
    out: cython.p_uchar,
    alpha: cython.p_const_float,
    width: cython.Py_ssize_t,
    opacity: cython.double,
) -> cython.int:
    flags: cython.int = 0
    x: cython.Py_ssize_t
    for x in range(width):
        flags |= quantize(out + x, cython.cast(cython.double, alpha[x]) * opacity * 255.0)
    return flags


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def quantize_masked_row(
    out: cython.p_uchar,
    alpha: cython.p_const_float,
    mask: cython.p_const_float,
    width: cython.Py_ssize_t,
    opacity: cython.double,
) -> cython.int:
    flags: cython.int = 0
    x: cython.Py_ssize_t
    for x in range(width):
        flags |= quantize(
            out + x,
            cython.cast(cython.double, alpha[x])
            * opacity
            * 255.0
            * cython.cast(cython.double, mask[x]),
        )
    return flags


def composite_nonisolated_blend(
    destination: cython.uchar[:, :, :],
    rendered: cython.const[cython.uchar][:, :, :],
    source_alpha: cython.const[cython.float][:, :],
    opacity: cython.double,
    mask_alpha: typing.Optional[cython.const[cython.float][:, :]],
    mode: cython.int,
    copy_unmasked: cython.bint,
    revised: cython.bint = True,
):
    """Composite a non-isolated group over its parent, removing the backdrop.

    The general branch of core_pdf.impl.render_target.composite_nonisolated_group:
    the effective alpha rint(source_alpha * opacity * 255 [* mask]) cast to
    uint8, then for each visible pixel the group backdrop removed as
    core_pdf_spec's remove_group_backdrop does, clipped to [0, 1], and blended
    over the destination as render_blend.blend_visible_pixels does. mode is a
    kernel blend code (0 normal, 1 multiply, 2 screen, 3 color dodge, 4 color
    burn); revised selects the PDF 2.0 dodge and burn endpoints. With
    copy_unmasked, a visible pixel whose mask is exactly 1 takes the rendered
    pixel unchanged. Returns the effective alpha plane.
    """
    height: cython.Py_ssize_t = source_alpha.shape[0]
    width: cython.Py_ssize_t = source_alpha.shape[1]
    if destination.shape[0] != height or destination.shape[1] != width or destination.shape[2] != 4:
        raise ValueError("destination must be source_alpha.shape + (4,)")
    if rendered.shape[0] != height or rendered.shape[1] != width or rendered.shape[2] != 4:
        raise ValueError("rendered must be source_alpha.shape + (4,)")
    masked: cython.bint = mask_alpha is not None
    if masked and (mask_alpha.shape[0] != height or mask_alpha.shape[1] != width):
        raise ValueError("mask_alpha differs from source_alpha in shape")
    if mode < MODE_NORMAL or mode > MODE_COLOR_BURN:
        raise ValueError("unsupported blend mode")

    effective = numpy.empty((height, width), dtype=numpy.uint8)
    if height == 0 or width == 0:
        return effective
    out: cython.uchar[:, ::1] = effective

    # Row pointers and strides held in locals: a byte store through a
    # memoryview may alias the memoryview structs, so indexing through them
    # would reload every stride after every store.
    d_pixel: cython.Py_ssize_t = destination.strides[1]
    d_channel: cython.Py_ssize_t = destination.strides[2]
    r_pixel: cython.Py_ssize_t = rendered.strides[1]
    r_channel: cython.Py_ssize_t = rendered.strides[2]
    a_pixel: cython.Py_ssize_t = source_alpha.strides[1]
    m_pixel: cython.Py_ssize_t = mask_alpha.strides[1] if masked else 0
    out_row: cython.p_uchar
    d_row: cython.p_uchar
    r_row: cython.p_const_uchar
    a_row: cython.p_const_char
    m_row: cython.p_const_char = cython.NULL

    y: cython.Py_ssize_t
    x: cython.Py_ssize_t
    c: cython.Py_ssize_t
    scaled: cython.double
    level: cython.uchar
    flags: cython.int = 0
    with cython.nogil:
        for y in range(height):
            out_row = cython.address(out[y, 0])
            a_row = cython.cast(cython.p_const_char, cython.address(source_alpha[y, 0]))
            if masked:
                m_row = cython.cast(cython.p_const_char, cython.address(mask_alpha[y, 0]))
            if a_pixel == 4 and not masked:
                flags |= quantize_row(
                    out_row, cython.cast(cython.p_const_float, a_row), width, opacity
                )
            elif a_pixel == 4 and m_pixel == 4:
                flags |= quantize_masked_row(
                    out_row,
                    cython.cast(cython.p_const_float, a_row),
                    cython.cast(cython.p_const_float, m_row),
                    width,
                    opacity,
                )
            else:
                for x in range(width):
                    scaled = float_at(a_row, x * a_pixel) * opacity * 255.0
                    if masked:
                        scaled = scaled * float_at(m_row, x * m_pixel)
                    flags |= quantize(cython.address(out_row[x]), scaled)
    if flags & OUT_OF_RANGE:
        raise ValueError("effective alpha out of range")
    if not flags & VISIBLE:
        return effective

    # x / 255.0 for every byte x: the same division, done once per value.
    unit = cython.declare(cython.double[256])
    for c in range(256):
        unit[c] = cython.cast(cython.double, c) / 255.0

    d: cython.p_uchar
    r: cython.p_const_uchar
    source_pixel: uint32_t
    backdrop_pixel: uint32_t
    alpha_bits = cython.declare(uint32_t)
    accumulated = cython.declare(cython.float)
    slot: uint32_t
    # The output is a function of the two pixels, the source alpha and the
    # effective alpha alone, and groups repeat a few of those combinations
    # over many pixels, so results are kept in a small direct-mapped cache.
    # A key's second word holds the effective alpha, never 0 for a pixel
    # composited, so the zeroed table starts empty.
    first_key: uint64_t
    second_key: uint64_t
    result: uint32_t
    cached_first = cython.declare(uint64_t[1024])
    cached_second = cython.declare(uint64_t[1024])
    cached_out = cython.declare(uint32_t[1024])
    memset(cached_second, 0, cython.sizeof(cached_second))
    hash_source: uint32_t = HASH_SOURCE
    hash_backdrop: uint32_t = HASH_BACKDROP
    hash_alpha: uint32_t = HASH_ALPHA
    with cython.nogil:
        for y in range(height):
            out_row = cython.address(out[y, 0])
            d_row = cython.address(destination[y, 0, 0])
            r_row = cython.address(rendered[y, 0, 0])
            a_row = cython.cast(cython.p_const_char, cython.address(source_alpha[y, 0]))
            if masked:
                m_row = cython.cast(cython.p_const_char, cython.address(mask_alpha[y, 0]))
            for x in range(width):
                level = out_row[x]
                if level == 0:
                    continue
                d = d_row + x * d_pixel
                r = r_row + x * r_pixel
                source_pixel = load_pixel(r, r_channel)
                if copy_unmasked and masked and float_at(m_row, x * m_pixel) == 1.0:
                    store_pixel(d, d_channel, source_pixel)
                    continue
                memcpy(cython.address(alpha_bits), a_row + x * a_pixel, 4)
                backdrop_pixel = load_pixel(d, d_channel)
                slot = (
                    (source_pixel * hash_source)
                    ^ (backdrop_pixel * hash_backdrop)
                    ^ (alpha_bits * hash_alpha)
                    ^ level
                ) >> 22
                first_key = (cython.cast(uint64_t, source_pixel) << 32) | backdrop_pixel
                second_key = (cython.cast(uint64_t, alpha_bits) << 8) | level
                if (cached_first[slot] == first_key) & (cached_second[slot] == second_key):
                    result = cached_out[slot]
                else:
                    memcpy(cython.address(accumulated), cython.address(alpha_bits), 4)
                    result = composite_pixel(
                        source_pixel, backdrop_pixel, level, accumulated, unit, mode, revised
                    )
                    cached_first[slot] = first_key
                    cached_second[slot] = second_key
                    cached_out[slot] = result
                store_pixel(d, d_channel, result)
    return effective


def remove_group_backdrop_samples(
    components: cython.const[cython.double][:, :],
    alpha: cython.const[cython.double][:],
    backdrop_components: cython.const[cython.double][:, :],
    backdrop_alpha: cython.const[cython.double][:],
    group_alpha: cython.const[cython.double][:],
):
    """The backdrop removal composite_nonisolated_blend applies per pixel, over
    rows of samples: core_pdf_spec's remove_group_backdrop without validation,
    returning the source components only. It exists to be checked against it."""
    count: cython.Py_ssize_t = components.shape[0]
    channels: cython.Py_ssize_t = components.shape[1]
    if (
        backdrop_components.shape[0] != count
        or backdrop_components.shape[1] != channels
        or alpha.shape[0] != count
        or backdrop_alpha.shape[0] != count
        or group_alpha.shape[0] != count
    ):
        raise ValueError("invalid transparency group samples")
    source = numpy.empty((count, channels), dtype=numpy.float64)
    out: cython.double[:, ::1] = source
    i: cython.Py_ssize_t
    k: cython.Py_ssize_t
    with cython.nogil:
        for i in range(count):
            for k in range(channels):
                out[i, k] = backdrop_removed(
                    components[i, k],
                    alpha[i],
                    backdrop_components[i, k],
                    backdrop_alpha[i],
                    group_alpha[i],
                )
    return source
