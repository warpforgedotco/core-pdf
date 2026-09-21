# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

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


@dataclass(frozen=True, slots=True)
class FilterDescriptor:
    name: str
    decoder: FilterDecoder | None
    predictor: bool = False
    ccitt: bool = False
    wants_image_dictionary: bool = False


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
