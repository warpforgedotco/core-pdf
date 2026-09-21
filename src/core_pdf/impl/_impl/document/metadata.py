# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TypeAlias, TypedDict, cast

from defusedxml.common import DefusedXmlException
from defusedxml.ElementTree import fromstring as defused_fromstring

from core_pdf.impl._impl.document.recovery.text_strings import decode_pdf_text_string
from core_pdf.impl._impl.document.standards import internal_resolve_catalog
from core_pdf.impl._impl.model.pdf_values import coerce_value
from core_pdf.impl._impl.pdf_names import recover_pdf_name
from core_pdf.impl.exceptions import PdfError
from core_pdf.impl.types import PdfName, PdfReference, PdfString
from core_pdf_spec.s_07_document.metadata import catalog_metadata_stream, info_dictionary
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict, PdfValueResolver

MetadataScalar: TypeAlias = (
    str
    | bytes
    | bytearray
    | memoryview
    | int
    | float
    | bool
    | None
    | PdfName
    | PdfReference
    | PdfStream
    | PdfString
)
MetadataList: TypeAlias = list["MetadataValue"]
MetadataMap: TypeAlias = dict[str, "MetadataValue"]
MetadataValue: TypeAlias = MetadataScalar | MetadataList | MetadataMap
InfoMetadataRecord: TypeAlias = dict[str, MetadataValue]


class XmpNodeRecord(TypedDict, total=False):
    tag: str
    attributes: dict[str, str]
    text: str
    children: list[XmpNodeRecord]
    parse_error: str


class MetadataRecord(TypedDict):
    info: InfoMetadataRecord
    xmp: XmpNodeRecord | None


def resolve_metadata(
    resolver: PdfValueResolver, trailer: PdfDict, *, recover: bool = False
) -> MetadataRecord:
    xmp: XmpNodeRecord | None
    try:
        xmp = resolve_metadata_stream(resolver, trailer, recover=True)
    except PdfError, RecursionError, ValueError:
        xmp = {"parse_error": "invalid XMP metadata"}
    return {
        "info": resolve_info_metadata(resolver, trailer, recover=recover),
        "xmp": xmp,
    }


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def resolve_info_metadata(
    resolver: PdfValueResolver, trailer: PdfDict, *, recover: bool = False
) -> InfoMetadataRecord:
    try:
        info = info_dictionary(resolver, trailer)
        if info is None:
            return {}
        coerced = cast(dict[object, object], coerce_value(info, decode_pdf_text_string))
        return {
            str(recover_pdf_name(key) or key): cast(MetadataValue, value)
            for key, value in coerced.items()
        }
    except PdfError, RecursionError, ValueError:
        if recover:
            return {}
        raise


def xml_node_shell(node: ET.Element) -> XmpNodeRecord:
    attrs = {str(local_name(key)): value for key, value in node.attrib.items()}
    text = (node.text or "").strip()
    result: XmpNodeRecord = {"tag": local_name(node.tag)}
    if attrs:
        result["attributes"] = attrs
    if text:
        result["text"] = text
    return result


def xml_node_to_value(node: ET.Element) -> XmpNodeRecord:
    root = xml_node_shell(node)
    stack: list[tuple[ET.Element, XmpNodeRecord]] = [(node, root)]
    while stack:
        current, result = stack.pop()
        child_nodes = list(current)
        if child_nodes:
            children = [xml_node_shell(child) for child in child_nodes]
            result["children"] = children
            stack.extend(zip(child_nodes, children))
    return root


def parse_xmp_metadata(stream: object, *, recover: bool = False) -> XmpNodeRecord | None:
    if not isinstance(stream, PdfStream):
        return None
    raw = stream.data
    if not raw:
        return None
    try:
        root = defused_fromstring(raw)
    except (ET.ParseError, DefusedXmlException) as error:
        if recover:
            return {"parse_error": "invalid XMP metadata"}
        raise ValueError("invalid XMP metadata") from error

    packet: XmpNodeRecord = {
        "tag": local_name(root.tag),
        "attributes": {str(local_name(key)): value for key, value in root.attrib.items()},
    }

    children = list(root)
    if children:
        packet["children"] = [xml_node_to_value(child) for child in children]
    text = (root.text or "").strip()
    if text:
        packet["text"] = text
    return packet


def resolve_metadata_stream(
    resolver: PdfValueResolver, trailer: PdfDict, *, recover: bool = False
) -> XmpNodeRecord | None:
    try:
        catalog = internal_resolve_catalog(resolver, trailer)
        if catalog is None:
            return None
        metadata = catalog_metadata_stream(resolver, catalog)
    except ValueError:
        if recover:
            return None
        raise
    return parse_xmp_metadata(metadata, recover=recover) if metadata is not None else None
