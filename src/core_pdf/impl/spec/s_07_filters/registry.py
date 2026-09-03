# SPDX-License-Identifier: AGPL-3.0-only
"""Canonical metadata for supported PDF stream filters."""

from __future__ import annotations

from collections.abc import Mapping
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
    ccitt: bool = False
    # JPX reads the image dictionary (for /SMaskInData and colour) rather than
    # the DecodeParms every other filter is handed.
    wants_image_dictionary: bool = False


FILTER_DESCRIPTORS = (
    FilterDescriptor(
        "FlateDecode",
        "flate",
        ("flatedecode", "platedecode"),
        predictor=True,
    ),
    FilterDescriptor("Fl", "flate", ("fl",), predictor=True),
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
    ),
    FilterDescriptor("LZW", "lzw", ("lzw",), predictor=True),
    FilterDescriptor("DCT", "jpeg", ("dct",)),
    FilterDescriptor("DCTDecode", "jpeg", ("dctdecode",)),
    FilterDescriptor(
        "CCITTFaxDecode",
        "ccitt",
        ("ccitt", "ccittfaxdecode"),
        ccitt=True,
    ),
    FilterDescriptor("CCF", "ccitt", ("ccf",), ccitt=True),
    FilterDescriptor("Crypt", "crypt", ("crypt",)),
    FilterDescriptor(
        "JPXDecode",
        "jpx",
        ("jpxdecode",),
        wants_image_dictionary=True,
    ),
    FilterDescriptor("JBIG2Decode", "jbig2", ("jbig2decode",)),
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
