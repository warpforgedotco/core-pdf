# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from core_pdf.impl._impl.pdf_names import recover_pdf_name
from core_pdf_spec.s_07_filters.registry import (
    FILTER_DESCRIPTORS as PDF_FILTER_DESCRIPTORS,
)
from core_pdf_spec.s_07_filters.registry import (
    FilterDecoder as FilterDecoder,
)
from core_pdf_spec.s_07_filters.registry import (
    FilterDescriptor,
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
    "flate": internal_RAW_SAMPLE_IMAGE,
    "lzw": internal_RAW_SAMPLE_IMAGE,
}


def declared_filter_names(value: object) -> list[str]:
    values = value if isinstance(value, (list, tuple)) else (value,)
    return [name for item in values if (name := recover_pdf_name(item))]
