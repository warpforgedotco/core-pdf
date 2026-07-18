# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from functools import lru_cache
from typing import Any

from core_jpeg.impl.codecs.jpx.params import (
    JPX_CODEBLOCK_STYLE_RESET,
    JPX_CODEBLOCK_STYLE_SEGSYM,
    JPX_CODEBLOCK_STYLE_VSC,
)
from core_jpeg.impl.codecs.jpx.tier1_entropy import RawBitDecoder
from core_jpeg.impl.codecs.jpx.tier1_state import (
    T1_CTXNO_AGG,
    T1_CTXNO_MAG,
    T1_CTXNO_UNI,
    JpxTier1State,
    t1_zero_coding_context,
    t1_zero_coding_context_from_counts,
)
from core_jpeg.impl.errors import JpegParseError


def t1_significance_magnitude(bitplane: int) -> int:
    one = 1 << (bitplane + 1)
    return one | (one >> 1)


def t1_refinement_delta(bitplane: int) -> int:
    return 1 << bitplane


def t1_decode_sign(
    decoder: Any,
    state: JpxTier1State,
    x: int,
    y: int,
    magnitude: int,
    vertical_stripe_causal: bool = False,
) -> None:
    context, predicted_sign = state.sign_context(x, y, vertical_stripe_causal)
    sign = decoder.decode(context) ^ predicted_sign
    state.update_significance(x, y, -magnitude if sign else magnitude)


def t1_decode_sigpass(
    decoder: Any,
    state: JpxTier1State,
    bitplane: int,
    orientation: int,
    vertical_stripe_causal: bool = False,
) -> None:
    magnitude = t1_significance_magnitude(bitplane)
    significant = state.significant
    visited = state.visited
    for x, y, index in t1_scan_positions(state.width, state.height):
        if significant[index]:
            continue
        horizontal, vertical, diagonal = state.neighbor_counts(
            x,
            y,
            vertical_stripe_causal,
        )
        if horizontal + vertical + diagonal == 0:
            continue
        context = t1_zero_coding_context_from_counts(
            horizontal,
            vertical,
            diagonal,
            orientation,
        )
        if decoder.decode(context):
            t1_decode_sign(
                decoder,
                state,
                x,
                y,
                magnitude,
                vertical_stripe_causal,
            )
        visited[index] = True


def t1_decode_sigpass_raw(
    decoder: RawBitDecoder,
    state: JpxTier1State,
    bitplane: int,
    vertical_stripe_causal: bool = False,
) -> None:
    magnitude = t1_significance_magnitude(bitplane)
    significant = state.significant
    visited = state.visited
    for x, y, index in t1_scan_positions(state.width, state.height):
        if significant[index]:
            continue
        if not state.has_significant_neighbor(x, y, vertical_stripe_causal):
            continue
        if decoder.decode_bit():
            sign = decoder.decode_bit()
            state.update_significance(x, y, -magnitude if sign else magnitude)
        visited[index] = True


def t1_decode_refpass(
    decoder: Any,
    state: JpxTier1State,
    bitplane: int,
    vertical_stripe_causal: bool = False,
) -> None:
    delta = t1_refinement_delta(bitplane)
    data = state.data
    significant = state.significant
    visited = state.visited
    refined = state.refined
    for x, y, index in t1_scan_positions(state.width, state.height):
        if not significant[index] or visited[index]:
            continue
        if refined[index]:
            context = T1_CTXNO_MAG + 2
        elif state.has_significant_neighbor(x, y, vertical_stripe_causal):
            context = T1_CTXNO_MAG + 1
        else:
            context = T1_CTXNO_MAG
        bit = decoder.decode(context)
        if data[index] < 0:
            data[index] += -delta if bit else delta
        else:
            data[index] += delta if bit else -delta
        refined[index] = True


def t1_decode_refpass_raw(
    decoder: RawBitDecoder,
    state: JpxTier1State,
    bitplane: int,
) -> None:
    delta = t1_refinement_delta(bitplane)
    data = state.data
    significant = state.significant
    visited = state.visited
    refined = state.refined
    for ignored_x, ignored_y, index in t1_scan_positions(state.width, state.height):
        if not significant[index] or visited[index]:
            continue
        bit = decoder.decode_bit()
        if data[index] < 0:
            data[index] += -delta if bit else delta
        else:
            data[index] += delta if bit else -delta
        refined[index] = True


def t1_decode_cleanup_pass(
    decoder: Any,
    state: JpxTier1State,
    bitplane: int,
    orientation: int,
    *,
    vertical_stripe_causal: bool = False,
    segmentation_symbols: bool = False,
) -> None:
    magnitude = t1_significance_magnitude(bitplane)
    width = state.width
    height = state.height
    significant = state.significant
    visited = state.visited
    for stripe_y in range(0, height, 4):
        stripe_end = min(stripe_y + 4, height)
        for x in range(width):
            y = stripe_y
            while y < stripe_end:
                run_length_ok = stripe_end - y == 4
                if run_length_ok:
                    for offset in range(4):
                        index = (y + offset) * width + x
                        if (
                            significant[index]
                            or visited[index]
                            or state.has_significant_neighbor(
                                x,
                                y + offset,
                                vertical_stripe_causal,
                            )
                        ):
                            run_length_ok = False
                            break
                if run_length_ok and decoder.decode(T1_CTXNO_AGG):
                    offset = decoder.decode(T1_CTXNO_UNI) * 2 + decoder.decode(T1_CTXNO_UNI)
                    y += offset
                    t1_decode_sign(
                        decoder,
                        state,
                        x,
                        y,
                        magnitude,
                        vertical_stripe_causal,
                    )
                    visited[y * width + x] = True
                    y += 1
                    continue
                if run_length_ok:
                    y += 4
                    continue
                index = y * width + x
                if not significant[index] and not visited[index]:
                    context = t1_zero_coding_context(
                        state,
                        x,
                        y,
                        orientation,
                        vertical_stripe_causal,
                    )
                    if decoder.decode(context):
                        t1_decode_sign(
                            decoder,
                            state,
                            x,
                            y,
                            magnitude,
                            vertical_stripe_causal,
                        )
                y += 1
    if segmentation_symbols:
        t1_decode_segmentation_symbol(decoder)
    state.reset_pass_flags()


def t1_decode_segmentation_symbol(decoder: Any) -> int:
    value = 0
    for ignored in range(4):
        value = (value << 1) | decoder.decode(T1_CTXNO_UNI)
    return value


def t1_decode_codeblock_passes(
    decoder: Any,
    state: JpxTier1State,
    bitplane_plus_one: int,
    pass_type: int,
    num_passes: int,
    orientation: int,
    *,
    codeblock_style: int = 0,
    raw_bypass: bool = False,
) -> tuple[int, int]:
    vertical_stripe_causal = bool(codeblock_style & JPX_CODEBLOCK_STYLE_VSC)
    for ignored in range(num_passes):
        if bitplane_plus_one <= 0:
            break
        bitplane = bitplane_plus_one - 1
        if raw_bypass and pass_type == 2:
            raise JpegParseError("JPX raw bypass segment cannot contain cleanup pass")
        if raw_bypass and pass_type == 0:
            t1_decode_sigpass_raw(
                decoder,
                state,
                bitplane,
                vertical_stripe_causal,
            )
        elif raw_bypass and pass_type == 1:
            t1_decode_refpass_raw(decoder, state, bitplane)
        elif pass_type == 0:
            t1_decode_sigpass(
                decoder,
                state,
                bitplane,
                orientation,
                vertical_stripe_causal,
            )
        elif pass_type == 1:
            t1_decode_refpass(
                decoder,
                state,
                bitplane,
                vertical_stripe_causal,
            )
        else:
            t1_decode_cleanup_pass(
                decoder,
                state,
                bitplane,
                orientation,
                vertical_stripe_causal=vertical_stripe_causal,
                segmentation_symbols=bool(codeblock_style & JPX_CODEBLOCK_STYLE_SEGSYM),
            )
        if (
            codeblock_style & JPX_CODEBLOCK_STYLE_RESET
            and not raw_bypass
            and hasattr(decoder, "reset_contexts")
        ):
            decoder.reset_contexts()
        pass_type += 1
        if pass_type == 3:
            pass_type = 0
            bitplane_plus_one -= 1
    return bitplane_plus_one, pass_type


@lru_cache(maxsize=128)
def t1_scan_order(width: int, height: int) -> tuple[tuple[int, int], ...]:
    coords: list[tuple[int, int]] = []
    for stripe_y in range(0, height, 4):
        for x in range(width):
            for y in range(stripe_y, min(stripe_y + 4, height)):
                coords.append((x, y))
    return tuple(coords)


@lru_cache(maxsize=128)
def t1_scan_positions(width: int, height: int) -> tuple[tuple[int, int, int], ...]:
    coords: list[tuple[int, int, int]] = []
    for stripe_y in range(0, height, 4):
        for x in range(width):
            for y in range(stripe_y, min(stripe_y + 4, height)):
                coords.append((x, y, y * width + x))
    return tuple(coords)
