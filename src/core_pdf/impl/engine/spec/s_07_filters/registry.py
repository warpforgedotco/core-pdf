# SPDX-License-Identifier: AGPL-3.0-only
"""Canonical metadata for supported PDF stream filters."""

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
    """Normalization and decoding properties for one accepted filter name."""

    name: str
    decoder: FilterDecoder | None
    aliases: tuple[str, ...]
    predictor: bool = False
    expensive_decode: bool = False
    ccitt: bool = False


FILTER_DESCRIPTORS = (
    FilterDescriptor(
        "FlateDecode",
        "flate",
        ("flatedecode", "platedecode"),
        predictor=True,
        expensive_decode=True,
    ),
    FilterDescriptor("Fl", "flate", ("fl",), predictor=True, expensive_decode=True),
    FilterDescriptor("ASCIIHexDecode", "ascii_hex", ("asciihexdecode",)),
    FilterDescriptor("AHx", "ascii_hex", ("ahx",)),
    FilterDescriptor("ASCII85Decode", "ascii85", ("ascii85decode",)),
    FilterDescriptor("A85", "ascii85", ("a85",)),
    FilterDescriptor("RunLengthDecode", "run_length", ("runlengthdecode", "runlength")),
    FilterDescriptor("RL", "run_length", ("rl",)),
    FilterDescriptor(
        "LZWDecode",
        "lzw",
        ("lzwdecode",),
        predictor=True,
        expensive_decode=True,
    ),
    FilterDescriptor("LZW", "lzw", ("lzw",), predictor=True, expensive_decode=True),
    FilterDescriptor("DCT", "jpeg", ("dct",), expensive_decode=True),
    FilterDescriptor("DCTDecode", "jpeg", ("dctdecode",), expensive_decode=True),
    FilterDescriptor(
        "CCITTFaxDecode",
        "ccitt",
        ("ccitt", "ccittfaxdecode"),
        expensive_decode=True,
        ccitt=True,
    ),
    FilterDescriptor("CCF", "ccitt", ("ccf",), expensive_decode=True, ccitt=True),
    FilterDescriptor("Crypt", "crypt", ("crypt",)),
    FilterDescriptor("JPXDecode", "jpx", ("jpxdecode",), expensive_decode=True),
    FilterDescriptor("JBIG2Decode", "jbig2", ("jbig2decode",), expensive_decode=True),
    FilterDescriptor("Identity", None, ("identity",)),
    FilterDescriptor("None", None, ("none",)),
)

FILTER_DESCRIPTOR_BY_NAME = {descriptor.name: descriptor for descriptor in FILTER_DESCRIPTORS}
FILTER_NAME_ALIASES = {
    alias: descriptor.name for descriptor in FILTER_DESCRIPTORS for alias in descriptor.aliases
}
CCITT_FILTERS = frozenset(descriptor.name for descriptor in FILTER_DESCRIPTORS if descriptor.ccitt)
PREDICTOR_FILTERS = frozenset(
    descriptor.name for descriptor in FILTER_DESCRIPTORS if descriptor.predictor
)
EXPENSIVE_DECODE_CACHE_FILTERS = frozenset(
    descriptor.name for descriptor in FILTER_DESCRIPTORS if descriptor.expensive_decode
)
