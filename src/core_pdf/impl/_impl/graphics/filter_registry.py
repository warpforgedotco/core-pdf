# SPDX-License-Identifier: AGPL-3.0-only
"""Canonical metadata for supported PDF stream filters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from core_pdf.impl.spec.s_07_filters.registry import (
    FILTER_DESCRIPTORS as PDF_FILTER_DESCRIPTORS,
)
from core_pdf.impl.spec.s_07_filters.registry import (
    FilterDecoder as FilterDecoder,
)
from core_pdf.impl.spec.s_07_filters.registry import (
    FilterDescriptor,
)
from core_pdf.impl.spec.s_07_filters.registry import (
    declared_filter_names as declared_filter_names,
)

FILTER_DESCRIPTORS = (
    *PDF_FILTER_DESCRIPTORS,
    FilterDescriptor("Identity", None),
    FilterDescriptor("None", None),
)
FILTER_DESCRIPTOR_BY_NAME = {descriptor.name: descriptor for descriptor in FILTER_DESCRIPTORS}
FILTER_NAME_ALIASES = {
    descriptor.name.lower(): descriptor.name for descriptor in FILTER_DESCRIPTORS
}
FILTER_NAME_ALIASES.update(
    platedecode="FlateDecode", runlength="RunLengthDecode", ccitt="CCITTFaxDecode"
)
CCITT_FILTERS = frozenset(descriptor.name for descriptor in FILTER_DESCRIPTORS if descriptor.ccitt)
PREDICTOR_FILTERS = frozenset(
    descriptor.name for descriptor in FILTER_DESCRIPTORS if descriptor.predictor
)


@dataclass(frozen=True, slots=True)
class NativeImageSpec:
    """What a decoder's array fast path accepts, and how wide its output is.

    `channels` maps a normalized colour-space name to its component count and
    doubles as the preallocation table: a name absent from it means "decode
    without a preallocated buffer". `color_names` and `bits` are the wider set
    the fast path will accept at all; `bits=None` means any depth.
    """

    channels: Mapping[str | None, int]
    color_names: frozenset[str | None]
    bits: frozenset[int | None] | None = None


internal_RAW_SAMPLE_IMAGE = NativeImageSpec(
    channels={None: 1, "DeviceGray": 1, "DeviceRGB": 3},
    color_names=frozenset({None, "DeviceGray", "DeviceRGB"}),
    bits=frozenset({None, 8}),
)

NATIVE_IMAGE_SPECS: Mapping[str, NativeImageSpec] = {
    "jpeg": NativeImageSpec(
        channels={"DeviceGray": 1, "DeviceRGB": 3, "DeviceCMYK": 4},
        color_names=frozenset({None, "DeviceGray", "DeviceRGB", "DeviceCMYK"}),
        bits=frozenset({None, 8}),
    ),
    "jpx": NativeImageSpec(
        channels={"DeviceGray": 1, "DeviceRGB": 3},
        color_names=frozenset({None, "DeviceGray", "DeviceRGB"}),
    ),
    "ccitt": NativeImageSpec(
        channels={},
        color_names=frozenset({None, "DeviceGray"}),
        bits=frozenset({None, 1}),
    ),
    # flate and lzw share a spec: both decode to raw samples the caller reshapes.
    "flate": internal_RAW_SAMPLE_IMAGE,
    "lzw": internal_RAW_SAMPLE_IMAGE,
}
