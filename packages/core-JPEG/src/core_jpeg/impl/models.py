# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from dataclasses import dataclass, field

from core_jpeg.impl.types import CodecKind


@dataclass(frozen=True, slots=True)
class DecodedJpxComponent:
    index: int
    width: int
    height: int
    precision: int
    is_signed: bool
    data: bytes
    # Exact, level-shift-corrected component samples. ``data`` remains the
    # historical display-oriented 8-bit plane for compatibility.
    samples: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Jp2Resolution:
    vertical: float
    horizontal: float


@dataclass(frozen=True, slots=True)
class DecodedJpxImage:
    width: int
    height: int
    color_space: str | None
    components: tuple[DecodedJpxComponent, ...]
    interleaved: bytes
    native_components: tuple[DecodedJpxComponent, ...] = ()
    capture_resolution: Jp2Resolution | None = None
    display_resolution: Jp2Resolution | None = None


@dataclass(frozen=True, slots=True)
class DecodeWorkload:
    codec: CodecKind
    encoded_bytes: int | None = None
    width: int | None = None
    height: int | None = None
    components: int | None = None
    tiles: int | None = None
    tile_parts: int | None = None
    levels: int | None = None
    codeblocks: int | None = None
    codeblock_width: int | None = None
    codeblock_height: int | None = None
    codeblock_style: int | None = None
    progression_order: int | None = None
    reversible: bool | None = None
    progressive: bool | None = None
    restart_interval: int | None = None
    packet_uses_sop: bool = False
    packet_uses_eph: bool = False
    uses_mct: bool = False
    has_icc_profile: bool = False
    has_palette: bool = False
    has_channel_definitions: bool = False
    apply_embedded_color: bool = True
    native_component_output: bool = False
    features: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", frozenset(self.features))
