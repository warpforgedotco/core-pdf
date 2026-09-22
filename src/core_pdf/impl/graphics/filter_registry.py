# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, NoReturn, Self

from core_pdf.impl.pdf_names import recover_pdf_name
from core_pdf_spec.s_07_filters.registry import (
    FILTER_DESCRIPTORS as PDF_FILTER_DESCRIPTORS,
)
from core_pdf_spec.s_07_filters.registry import (
    FilterDecoder as FilterDecoder,
)
from core_pdf_spec.s_07_filters.registry import (
    FilterDescriptor,
)

internal_frozen_setattr = object.__setattr__


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


class NativeImageSpec:
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
        internal_frozen_setattr(self, "channels", channels)
        internal_frozen_setattr(self, "color_names", color_names)
        internal_frozen_setattr(self, "bits", bits)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"channels={self.channels!r}, "
            f"color_names={self.color_names!r}, "
            f"bits={self.bits!r}"
            ")"
        )

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

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            internal_frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        channels = changes.pop("channels", self.channels)
        color_names = changes.pop("color_names", self.color_names)
        bits = changes.pop("bits", self.bits)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(channels, color_names, bits)


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
