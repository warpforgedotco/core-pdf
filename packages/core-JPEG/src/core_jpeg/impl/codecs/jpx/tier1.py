# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Any, cast

from core_jpeg.impl.codecs.jpx.tier1_entropy import (
    MQDecoder,
    RawBitDecoder,
    default_t1_contexts_split,
)
from core_jpeg.impl.codecs.jpx.tier1_passes import t1_decode_codeblock_passes
from core_jpeg.impl.codecs.jpx.tier1_plan import build_tier1_codeblock_plan
from core_jpeg.impl.codecs.jpx.tier1_state import JpxTier1State


def t1_decode_codeblock(
    data: bytes,
    width: int,
    height: int,
    num_bitplanes: int,
    zero_bit_planes: int,
    num_passes: int,
    orientation: int,
    decoder: Any | None = None,
    codeblock_style: int = 0,
) -> list[int]:
    if width <= 0 or height <= 0:
        return []
    state = JpxTier1State(width, height)
    bitplane_plus_one = num_bitplanes - zero_bit_planes
    if bitplane_plus_one <= 0 or num_passes <= 0:
        return state.data
    entropy_decoder = decoder if decoder is not None else MQDecoder(data)
    t1_decode_codeblock_passes(
        entropy_decoder,
        state,
        bitplane_plus_one,
        pass_type=2,
        num_passes=num_passes,
        orientation=orientation,
        codeblock_style=codeblock_style,
    )
    return state.data


def decode_tier1_codeblock_segments(
    block: Any,
    width: int,
    height: int,
    num_bitplanes: int,
    orientation: int,
    codeblock_style: int = 0,
) -> list[int]:
    if width <= 0 or height <= 0:
        block.data = []
        return block.data
    if not block.segments:
        return [0] * (width * height)
    plan = build_tier1_codeblock_plan(
        block.segments,
        num_bitplanes=num_bitplanes,
        codeblock_style=codeblock_style,
    )
    state = JpxTier1State(width, height)
    if plan.initial_bitplane_plus_one <= 0:
        block.data = state.data
        return block.data
    mq_contexts: tuple[list[int], list[int]] = default_t1_contexts_split()
    for segment_plan in plan.segments:
        segment = block.segments[segment_plan.segment_index]
        decoder = (
            RawBitDecoder(segment.payload)
            if segment_plan.raw_bypass
            else MQDecoder(segment.payload, mq_contexts)
        )
        t1_decode_codeblock_passes(
            decoder,
            state,
            segment_plan.bitplane_plus_one,
            segment_plan.pass_type,
            segment_plan.num_passes,
            orientation,
            codeblock_style=codeblock_style,
            raw_bypass=segment_plan.raw_bypass,
        )
        if not segment_plan.raw_bypass:
            mq_contexts = cast(MQDecoder, decoder).contexts()
    block.data = state.data
    return block.data
