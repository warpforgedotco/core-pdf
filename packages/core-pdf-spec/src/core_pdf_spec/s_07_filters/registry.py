# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from typing import ClassVar, Literal, TypeAlias

from core_records import Record, frozen_setattr

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


class FilterDescriptor(Record):
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
