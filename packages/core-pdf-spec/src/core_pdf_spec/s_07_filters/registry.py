# SPDX-License-Identifier: AGPL-3.0-only
"""Canonical metadata for supported PDF stream filters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from core_pdf_spec.s_07_filters.errors import FilterParseError
from core_pdf_spec.s_07_syntax_primitives.coercion import normalize_pdf_name

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


def declared_filter_names(value: object) -> list[str]:
    """Return the PDF names declared by one ``Filter`` entry."""
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else (value,)
    names: list[str] = []
    for item in values:
        name = normalize_pdf_name(item)
        if name is None:
            raise FilterParseError("invalid stream filter name")
        names.append(name)
    return names


@dataclass(frozen=True, slots=True)
class FilterDescriptor:
    """Normalization and decoding properties for one accepted filter name."""

    name: str
    decoder: FilterDecoder | None
    predictor: bool = False
    ccitt: bool = False
    # JPX reads the image dictionary (for /SMaskInData and colour) rather than
    # the DecodeParms every other filter is handed.
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
    "declared_filter_names",
    "FilterDescriptor",
    "FILTER_DESCRIPTORS",
    "FILTER_DESCRIPTOR_BY_NAME",
    "CCITT_FILTERS",
    "PREDICTOR_FILTERS",
)
