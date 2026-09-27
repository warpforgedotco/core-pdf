# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import numpy

from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf.impl.types import Record, frozen_setattr
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


class NativeImageCodec(Record):
    __slots__ = ("spec", "decode", "requires_identity_decode", "after_filters")

    spec: NativeImageSpec
    decode: NativeDecodeFn
    requires_identity_decode: bool
    after_filters: bool

    __fields__: ClassVar[tuple[str, ...]] = (
        "spec",
        "decode",
        "requires_identity_decode",
        "after_filters",
    )
    __match_args__ = ("spec", "decode", "requires_identity_decode", "after_filters")

    def __init__(
        self,
        spec: NativeImageSpec,
        decode: NativeDecodeFn,
        requires_identity_decode: bool = True,
        after_filters: bool = False,
    ) -> None:
        frozen_setattr(self, "spec", spec)
        frozen_setattr(self, "decode", decode)
        frozen_setattr(self, "requires_identity_decode", requires_identity_decode)
        frozen_setattr(self, "after_filters", after_filters)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.spec == other.spec
            and self.decode == other.decode
            and self.requires_identity_decode == other.requires_identity_decode
            and self.after_filters == other.after_filters
        )

    def __hash__(self) -> int:
        return hash((self.spec, self.decode, self.requires_identity_decode, self.after_filters))


class TolerantFilter(Record):
    __slots__ = ("decoder", "decode", "passthrough", "native")

    decoder: FilterDecoder
    decode: FilterFn
    passthrough: PassthroughFilterFn | None
    native: NativeImageCodec | None

    __fields__: ClassVar[tuple[str, ...]] = ("decoder", "decode", "passthrough", "native")
    __match_args__ = ("decoder", "decode", "passthrough", "native")

    def __init__(
        self,
        decoder: FilterDecoder,
        decode: FilterFn,
        passthrough: PassthroughFilterFn | None = None,
        native: NativeImageCodec | None = None,
    ) -> None:
        frozen_setattr(self, "decoder", decoder)
        frozen_setattr(self, "decode", decode)
        frozen_setattr(self, "passthrough", passthrough)
        frozen_setattr(self, "native", native)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.decoder == other.decoder
            and self.decode == other.decode
            and self.passthrough == other.passthrough
            and self.native == other.native
        )

    def __hash__(self) -> int:
        return hash((self.decoder, self.decode, self.passthrough, self.native))


def declared_filter_names(value: object) -> list[str]:
    values = value if isinstance(value, (list, tuple)) else (value,)
    return [name for item in values if (name := recover_pdf_name(item))]
