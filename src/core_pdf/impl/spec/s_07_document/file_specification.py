# SPDX-License-Identifier: AGPL-3.0-only

"""ISO 32000-2 7.11: file specification dictionaries and their embedded file streams."""

from __future__ import annotations

from core_pdf.impl.spec.citations import cites
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict, PdfObject, PdfValueResolver
from core_pdf.impl.spec.types import PdfString
from core_records import GeneratedRecord

# The EF keys naming an embedded file stream, in the order a reader takes them.
EMBEDDED_FILE_KEYS = ("F", "UF")


class FileSpecificationStrings(GeneratedRecord):
    """A file specification dictionary's names and description, as the strings it holds."""

    file_name: bytes | None = None
    unicode_file_name: bytes | None = None
    description: bytes | None = None


class EmbeddedFileParameters(GeneratedRecord):
    """An embedded file stream's Subtype and the entries of its Params dictionary."""

    subtype: str | None = None
    size: int | None = None
    creation_date: bytes | None = None
    modification_date: bytes | None = None
    checksum: bytes | None = None


def string_bytes(value: PdfObject) -> bytes | None:
    if isinstance(value, PdfString):
        return value.data
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    return None


def string_entry(resolver: PdfValueResolver, container: PdfDict, key: str) -> bytes | None:
    value = resolver.resolve(container.get(key))
    if value is None:
        return None
    data = string_bytes(value)
    if data is None:
        raise ValueError(f"invalid file specification {key} string")
    return data


@cites("ISO 32000-2", "7.11.3", rule="a file specification's F file name")
@cites("ISO 32000-2", "7.11.3", introduced="1.7", rule="UF, a file name as a text string")
@cites("ISO 32000-2", "7.11.3", introduced="1.6", rule="Desc, a file specification's description")
def file_specification_strings(
    resolver: PdfValueResolver, specification: PdfDict
) -> FileSpecificationStrings:
    """The F, UF and Desc strings, undecoded: F is a byte string, UF and Desc text strings."""
    return FileSpecificationStrings(
        file_name=string_entry(resolver, specification, "F"),
        unicode_file_name=string_entry(resolver, specification, "UF"),
        description=string_entry(resolver, specification, "Desc"),
    )


@cites("ISO 32000-2", "7.11.4", introduced="1.3", rule="the embedded file streams of EF")
def embedded_file_stream(resolver: PdfValueResolver, specification: PdfDict) -> PdfStream | None:
    """The stream EF names under F, or else UF; None when the file is not embedded."""
    embedded = resolver.resolve(specification.get("EF"))
    if embedded is None:
        return None
    if not isinstance(embedded, dict):
        raise ValueError("invalid file specification EF dictionary")
    for key in EMBEDDED_FILE_KEYS:
        stream = resolver.resolve(embedded.get(key))
        if stream is None:
            continue
        if not isinstance(stream, PdfStream):
            raise ValueError(f"invalid embedded file stream under EF {key}")
        return stream
    return None


@cites("ISO 32000-2", "7.11.4", introduced="1.3", rule="an embedded file's Subtype and Params")
def embedded_file_parameters(
    resolver: PdfValueResolver, stream: PdfStream
) -> EmbeddedFileParameters:
    """The MIME type and the Size, CreationDate, ModDate and CheckSum an embedded file states."""
    dictionary = stream.dictionary
    subtype_value = dictionary.get("Subtype")
    subtype = resolver.resolve_name(subtype_value)
    if subtype_value is not None and subtype is None:
        raise ValueError("invalid embedded file Subtype")
    params = resolver.resolve(dictionary.get("Params"))
    if params is None:
        return EmbeddedFileParameters(subtype=subtype)
    if not isinstance(params, dict):
        raise ValueError("invalid embedded file Params dictionary")
    size_value = params.get("Size")
    size = resolver.resolve_int(size_value)
    if size_value is not None and (size is None or size < 0):
        raise ValueError("invalid embedded file Size")
    return EmbeddedFileParameters(
        subtype=subtype,
        size=size,
        creation_date=string_entry(resolver, params, "CreationDate"),
        modification_date=string_entry(resolver, params, "ModDate"),
        checksum=string_entry(resolver, params, "CheckSum"),
    )


__all__ = (
    "EMBEDDED_FILE_KEYS",
    "EmbeddedFileParameters",
    "FileSpecificationStrings",
    "embedded_file_parameters",
    "embedded_file_stream",
    "file_specification_strings",
    "string_bytes",
)
