# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.records import Record
from core_pdf_spec.s_07_filters.registry import (
    FILTER_DESCRIPTORS as PDF_FILTER_DESCRIPTORS,
)
from core_pdf_spec.s_07_filters.registry import (
    FilterDecoder as FilterDecoder,
)
from core_pdf_spec.s_07_filters.registry import (
    FilterDescriptor,
)

frozen_setattr = object.__setattr__


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


class NativeImageSpec(Record):
    __slots__ = ("channels", "color_names", "bits")

    channels: Mapping[str | None, int]
    color_names: frozenset[str | None]
    bits: frozenset[int | None] | None

    __fields__: ClassVar[tuple[str, ...]] = ("channels", "color_names", "bits")
    __match_args__ = ("channels", "color_names", "bits")

    def __init__(
        self,
        channels: Mapping[str | None, int],
        color_names: frozenset[str | None],
        bits: frozenset[int | None] | None = None,
    ) -> None:
        frozen_setattr(self, "channels", channels)
        frozen_setattr(self, "color_names", color_names)
        frozen_setattr(self, "bits", bits)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.channels == other.channels
            and self.color_names == other.color_names
            and self.bits == other.bits
        )

    def __hash__(self) -> int:
        return hash((self.channels, self.color_names, self.bits))


RAW_SAMPLE_IMAGE = NativeImageSpec(
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
    "flate": RAW_SAMPLE_IMAGE,
    "lzw": RAW_SAMPLE_IMAGE,
}


def declared_filter_names(value: object) -> list[str]:
    values = value if isinstance(value, (list, tuple)) else (value,)
    return [name for item in values if (name := recover_pdf_name(item))]
