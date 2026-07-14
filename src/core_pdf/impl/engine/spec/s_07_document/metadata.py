# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TypeAlias, TypeGuard

from core_pdf.impl.engine.spec.s_07_objects.resolver import ObjectResolver
from core_pdf.impl.engine.spec.s_07_syntax.primitives import (
    PdfDictLike,
    PdfObject,
    PdfStream,
    coerce_value,
)
from core_pdf.impl.engine.spec.s_09_fonts.encoding import decode_pdf_text_string

MetadataValue: TypeAlias = (
    None | bool | int | float | str | list["MetadataValue"] | dict[str, "MetadataValue"]
)
MetadataDict: TypeAlias = dict[str, MetadataValue]


def is_metadata_value(value: object) -> TypeGuard[MetadataValue]:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, list):
        return all(is_metadata_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and is_metadata_value(item) for key, item in value.items())
    return False


def resolve_metadata(resolver: ObjectResolver, trailer: PdfDictLike) -> MetadataDict:
    return {
        "info": resolve_info_metadata(resolver, trailer),
        "xmp": resolve_metadata_stream(resolver, trailer),
    }


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def resolve_info_metadata(resolver: ObjectResolver, trailer: PdfDictLike) -> MetadataDict:
    info_ref = trailer.get("Info")
    if info_ref is None:
        return {}
    info = resolver.resolve_dict(info_ref)
    if info is None:
        raise ValueError("invalid trailer Info dictionary")
    result: MetadataDict = {}
    for key, value in info.items():
        coerced = coerce_value(value, decode_pdf_text_string)
        if not is_metadata_value(coerced):
            raise ValueError("invalid metadata value")
        result[str(key)] = coerced
    return result


def xml_node_shell(node: ET.Element) -> MetadataDict:
    attrs: MetadataDict = {str(local_name(key)): value for key, value in node.attrib.items()}
    text = (node.text or "").strip()
    result: MetadataDict = {"tag": local_name(node.tag)}
    if attrs:
        result["attributes"] = attrs
    if text:
        result["text"] = text
    return result


def xml_node_to_value(node: ET.Element) -> MetadataDict:
    root = xml_node_shell(node)
    stack: list[tuple[ET.Element, MetadataDict]] = [(node, root)]
    while stack:
        current, result = stack.pop()
        child_nodes = list(current)
        if child_nodes:
            child_results: list[MetadataDict] = [xml_node_shell(child) for child in child_nodes]
            result["children"] = list(child_results)
            stack.extend(zip(child_nodes, child_results))
    return root


def parse_xmp_metadata(stream: PdfObject) -> MetadataDict | None:
    if not isinstance(stream, PdfStream):
        return None
    raw = stream.data
    if not raw:
        return None
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        raise ValueError("invalid XMP metadata")

    packet: MetadataDict = {
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


def resolve_metadata_stream(resolver: ObjectResolver, trailer: PdfDictLike) -> MetadataDict | None:
    root = resolver.resolve_dict(trailer.get("Root"))
    if root is None:
        return None
    metadata = resolver.resolve(root.get("Metadata"))
    if isinstance(metadata, PdfStream):
        return parse_xmp_metadata(metadata)
    if metadata is not None:
        raise ValueError("invalid Metadata stream")
    return None
