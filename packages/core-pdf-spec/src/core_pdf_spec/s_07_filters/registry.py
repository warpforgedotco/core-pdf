# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import Any, ClassVar, Literal, NoReturn, Self, TypeAlias

frozen_setattr = object.__setattr__


FilterDecoder: TypeAlias = Literal[
    "ascii85",
    "ascii_hex",
    "ccitt",
    "crypt",
    "flate",
    "jbig2",
    "jpeg",
    "jpx",
    "lzw",
    "run_length",
]


class FilterDescriptor:
    __slots__ = ("name", "decoder", "predictor", "ccitt", "wants_image_dictionary")

    name: str
    decoder: FilterDecoder | None
    predictor: bool
    ccitt: bool
    wants_image_dictionary: bool

    __fields__: ClassVar[tuple[str, ...]] = (
        "name",
        "decoder",
        "predictor",
        "ccitt",
        "wants_image_dictionary",
    )
    __match_args__ = ("name", "decoder", "predictor", "ccitt", "wants_image_dictionary")

    def __init__(
        self,
        name: str,
        decoder: FilterDecoder | None,
        predictor: bool = False,
        ccitt: bool = False,
        wants_image_dictionary: bool = False,
    ) -> None:
        frozen_setattr(self, "name", name)
        frozen_setattr(self, "decoder", decoder)
        frozen_setattr(self, "predictor", predictor)
        frozen_setattr(self, "ccitt", ccitt)
        frozen_setattr(self, "wants_image_dictionary", wants_image_dictionary)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"name={self.name!r}, "
            f"decoder={self.decoder!r}, "
            f"predictor={self.predictor!r}, "
            f"ccitt={self.ccitt!r}, "
            f"wants_image_dictionary={self.wants_image_dictionary!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.name == other.name
            and self.decoder == other.decoder
            and self.predictor == other.predictor
            and self.ccitt == other.ccitt
            and self.wants_image_dictionary == other.wants_image_dictionary
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.name,
                self.decoder,
                self.predictor,
                self.ccitt,
                self.wants_image_dictionary,
            )
        )

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        name = changes.pop("name", self.name)
        decoder = changes.pop("decoder", self.decoder)
        predictor = changes.pop("predictor", self.predictor)
        ccitt = changes.pop("ccitt", self.ccitt)
        wants_image_dictionary = changes.pop("wants_image_dictionary", self.wants_image_dictionary)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(name, decoder, predictor, ccitt, wants_image_dictionary)


FILTER_DESCRIPTORS = (
    FilterDescriptor(
        "FlateDecode",
        "flate",
        predictor=True,
    ),
    FilterDescriptor("Fl", "flate", predictor=True),
    FilterDescriptor("ASCIIHexDecode", "ascii_hex"),
    FilterDescriptor("AHx", "ascii_hex"),
    FilterDescriptor("ASCII85Decode", "ascii85"),
    FilterDescriptor("A85", "ascii85"),
    FilterDescriptor("RunLengthDecode", "run_length"),
    FilterDescriptor("RL", "run_length"),
    FilterDescriptor(
        "LZWDecode",
        "lzw",
        predictor=True,
    ),
    FilterDescriptor("LZW", "lzw", predictor=True),
    FilterDescriptor("DCT", "jpeg"),
    FilterDescriptor("DCTDecode", "jpeg"),
    FilterDescriptor(
        "CCITTFaxDecode",
        "ccitt",
        ccitt=True,
    ),
    FilterDescriptor("CCF", "ccitt", ccitt=True),
    FilterDescriptor("Crypt", "crypt"),
    FilterDescriptor(
        "JPXDecode",
        "jpx",
        wants_image_dictionary=True,
    ),
    FilterDescriptor("JBIG2Decode", "jbig2"),
)

FILTER_DESCRIPTOR_BY_NAME = {descriptor.name: descriptor for descriptor in FILTER_DESCRIPTORS}
CCITT_FILTERS = frozenset(descriptor.name for descriptor in FILTER_DESCRIPTORS if descriptor.ccitt)
PREDICTOR_FILTERS = frozenset(
    descriptor.name for descriptor in FILTER_DESCRIPTORS if descriptor.predictor
)


__all__ = (
    "FilterDecoder",
    "FilterDescriptor",
    "FILTER_DESCRIPTORS",
    "FILTER_DESCRIPTOR_BY_NAME",
    "CCITT_FILTERS",
    "PREDICTOR_FILTERS",
)
