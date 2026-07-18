# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from core_jpeg.impl.codecs.jpx.params import (
    JPX_CODEBLOCK_STYLE_LAZY,
    JPX_CODEBLOCK_STYLE_RESET,
    JPX_CODEBLOCK_STYLE_SEGSYM,
    JPX_CODEBLOCK_STYLE_VSC,
)
from core_jpeg.impl.errors import JpegParseError

T1_PASS_SIGNIFICANCE = 0
T1_PASS_REFINEMENT = 1
T1_PASS_CLEANUP = 2


class Tier1SegmentMetadata(Protocol):
    num_passes: int
    zero_bit_planes: int


@dataclass(frozen=True, slots=True)
class JpxTier1SegmentPlan:
    segment_index: int
    num_passes: int
    bitplane_plus_one: int
    pass_type: int
    raw_bypass: bool
    reset_contexts: bool
    segmentation_symbols: bool
    vertical_stripe_causal: bool

    @property
    def entropy_mode(self) -> str:
        return "raw" if self.raw_bypass else "mq"


@dataclass(frozen=True, slots=True)
class JpxTier1CodeblockPlan:
    zero_bit_planes: int
    initial_bitplane_plus_one: int
    coded_bitplanes: int
    codeblock_style: int
    segments: tuple[JpxTier1SegmentPlan, ...]

    @property
    def uses_raw_bypass(self) -> bool:
        return any(segment.raw_bypass for segment in self.segments)

    @property
    def uses_context_reset(self) -> bool:
        return bool(self.codeblock_style & JPX_CODEBLOCK_STYLE_RESET)

    @property
    def uses_segmentation_symbols(self) -> bool:
        return bool(self.codeblock_style & JPX_CODEBLOCK_STYLE_SEGSYM)

    @property
    def uses_vertical_stripe_causal(self) -> bool:
        return bool(self.codeblock_style & JPX_CODEBLOCK_STYLE_VSC)


def build_tier1_codeblock_plan(
    segments: Sequence[Tier1SegmentMetadata],
    *,
    num_bitplanes: int,
    codeblock_style: int = 0,
) -> JpxTier1CodeblockPlan:
    zero_bit_planes = segments[0].zero_bit_planes if segments else 0
    for segment in segments:
        if segment.zero_bit_planes != zero_bit_planes:
            raise JpegParseError("JPX code-block zero bit-plane count changed")

    bitplane_plus_one = num_bitplanes - max(zero_bit_planes, 0)
    coded_bitplanes = bitplane_plus_one
    pass_type = T1_PASS_CLEANUP
    segment_plans: list[JpxTier1SegmentPlan] = []
    for segment_index, segment in enumerate(segments):
        if bitplane_plus_one <= 0:
            break
        if segment.num_passes <= 0:
            continue
        raw_bypass = tier1_segment_uses_raw_bypass(
            bitplane_plus_one,
            coded_bitplanes,
            pass_type,
            codeblock_style,
        )
        effective_passes, next_bitplane_plus_one, next_pass_type = advance_tier1_pass_state(
            bitplane_plus_one,
            pass_type,
            segment.num_passes,
        )
        if effective_passes <= 0:
            break
        segment_plans.append(
            JpxTier1SegmentPlan(
                segment_index=segment_index,
                num_passes=effective_passes,
                bitplane_plus_one=bitplane_plus_one,
                pass_type=pass_type,
                raw_bypass=raw_bypass,
                reset_contexts=bool(codeblock_style & JPX_CODEBLOCK_STYLE_RESET and not raw_bypass),
                segmentation_symbols=bool(codeblock_style & JPX_CODEBLOCK_STYLE_SEGSYM),
                vertical_stripe_causal=bool(codeblock_style & JPX_CODEBLOCK_STYLE_VSC),
            )
        )
        bitplane_plus_one = next_bitplane_plus_one
        pass_type = next_pass_type
    return JpxTier1CodeblockPlan(
        zero_bit_planes=zero_bit_planes,
        initial_bitplane_plus_one=coded_bitplanes,
        coded_bitplanes=coded_bitplanes,
        codeblock_style=codeblock_style,
        segments=tuple(segment_plans),
    )


def advance_tier1_pass_state(
    bitplane_plus_one: int,
    pass_type: int,
    num_passes: int,
) -> tuple[int, int, int]:
    effective_passes = 0
    for ignored in range(num_passes):
        if bitplane_plus_one <= 0:
            break
        effective_passes += 1
        pass_type += 1
        if pass_type == 3:
            pass_type = T1_PASS_SIGNIFICANCE
            bitplane_plus_one -= 1
    return effective_passes, bitplane_plus_one, pass_type


def tier1_segment_uses_raw_bypass(
    bitplane_plus_one: int,
    coded_bitplanes: int,
    pass_type: int,
    codeblock_style: int,
) -> bool:
    return bool(
        codeblock_style & JPX_CODEBLOCK_STYLE_LAZY
        and bitplane_plus_one <= coded_bitplanes - 4
        and pass_type < T1_PASS_CLEANUP
    )
