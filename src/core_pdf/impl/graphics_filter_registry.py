# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy

from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.types import GeneratedRecord
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


class NativeImageSpec(GeneratedRecord):
    channels: Mapping[str | None, int]
    color_names: frozenset[str | None]
    bits: frozenset[int | None] | None = None


RAW_SAMPLE_IMAGE = NativeImageSpec(
    channels={None: 1, "DeviceGray": 1, "DeviceRGB": 3},
    color_names=frozenset({None, "DeviceGray", "DeviceRGB"}),
    bits=frozenset({None, 8}),
)

JPEG_IMAGE = NativeImageSpec(
    channels={"DeviceGray": 1, "DeviceRGB": 3, "DeviceCMYK": 4},
    color_names=frozenset({None, "DeviceGray", "DeviceRGB", "DeviceCMYK"}),
    bits=frozenset({None, 8}),
)
JPX_IMAGE = NativeImageSpec(
    channels={"DeviceGray": 1, "DeviceRGB": 3},
    color_names=frozenset({None, "DeviceGray", "DeviceRGB"}),
)
CCITT_IMAGE = NativeImageSpec(
    channels={},
    color_names=frozenset({None, "DeviceGray"}),
    bits=frozenset({None, 1}),
)

type FilterFn = Callable[[bytes, object], bytes]
type PassthroughFilterFn = Callable[[bytes], tuple[bytes, bool]]
type NativeDecodeFn = Callable[
    [bytes | memoryview, object, tuple[int, ...] | None, Callable[[], bytes]],
    numpy.ndarray[Any, Any] | None,
]


class NativeImageCodec(GeneratedRecord):
    spec: NativeImageSpec
    decode: NativeDecodeFn
    requires_identity_decode: bool = True
    after_filters: bool = False


class TolerantFilter(GeneratedRecord):
    decoder: FilterDecoder
    decode: FilterFn
    passthrough: PassthroughFilterFn | None = None
    native: NativeImageCodec | None = None


def declared_filter_names(value: object) -> list[str]:
    values = value if isinstance(value, (list, tuple)) else (value,)
    return [name for item in values if (name := recover_pdf_name(item))]
