# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Protocol, cast

from core_pdf.impl.engine.spec.s_07_document.metadata_types import (
    InfoMetadataRecord,
    MetadataRecord,
    MetadataValue,
    XmpNodeRecord,
)
from core_pdf.impl.engine.spec.s_07_objects.coercion import (
    coerce_value,
    normalize_pdf_name,
)
from core_pdf.impl.engine.spec.s_07_objects.pdfdict import lookup_dict_key
from core_pdf.impl.engine.spec.s_09_fonts.encoding import decode_pdf_text_string
from core_pdf.impl.objects import PdfStream
from core_pdf.impl.types import PdfDict


class MetadataResolver(Protocol):
    def resolve(self, ref: object) -> object: ...

    def resolve_dict(self, value: object) -> PdfDict | None: ...


def resolve_metadata(
    resolver: MetadataResolver, trailer: PdfDict, *, recover: bool = False
) -> MetadataRecord:
    return {
        "info": resolve_info_metadata(resolver, trailer, recover=recover),
        # XMP is optional metadata and should not block document extraction.
        "xmp": resolve_metadata_stream(resolver, trailer, recover=True),
    }


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def resolve_info_metadata(
    resolver: MetadataResolver, trailer: PdfDict, *, recover: bool = False
) -> InfoMetadataRecord:
    info_ref = lookup_dict_key(trailer, "Info")
    if info_ref is None:
        return {}
    info = resolver.resolve_dict(info_ref)
    if not isinstance(info, dict):
        if recover:
            return {}
        raise ValueError("invalid trailer Info dictionary")
    return {
        str(normalize_pdf_name(key) or key): cast(
            MetadataValue, coerce_value(value, decode_pdf_text_string)
        )
        for key, value in info.items()
    }


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
        root = ET.fromstring(raw)
    except ET.ParseError:
        if recover:
            return {"parse_error": "invalid XMP metadata"}
        raise ValueError("invalid XMP metadata")

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
    resolver: MetadataResolver, trailer: PdfDict, *, recover: bool = False
) -> XmpNodeRecord | None:
    root = resolver.resolve_dict(lookup_dict_key(trailer, "Root"))
    if root is None:
        return None
    if not isinstance(root, dict):
        if recover:
            return None
        raise ValueError("invalid trailer Root dictionary")
    metadata = resolver.resolve(lookup_dict_key(root, "Metadata"))
    if isinstance(metadata, PdfStream):
        return parse_xmp_metadata(metadata, recover=recover)
    if metadata is not None:
        if recover:
            return None
        raise ValueError("invalid Metadata stream")
    return None
