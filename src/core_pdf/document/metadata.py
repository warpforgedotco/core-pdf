from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from core_pdf.fonts.encoding import decode_pdf_text_string
from core_pdf.syntax.primitives import PdfStream, coerce_value


def resolve_metadata(resolver: Any, trailer: dict[str, Any]) -> dict[str, Any]:
    return {
        "info": resolve_info_metadata(resolver, trailer),
        "xmp": resolve_metadata_stream(resolver, trailer),
    }


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def resolve_info_metadata(resolver: Any, trailer: dict[str, Any]) -> dict[str, Any]:
    info_ref = trailer.get("Info")
    if info_ref is None:
        return {}
    info = resolver.resolve_dict(info_ref)
    if info is None:
        raise ValueError("invalid trailer Info dictionary")
    return {str(key): coerce_value(value, decode_pdf_text_string) for key, value in info.items()}


def xml_node_shell(node: ET.Element) -> dict[str, Any]:
    attrs = {str(local_name(key)): value for key, value in node.attrib.items()}
    text = (node.text or "").strip()
    result: dict[str, Any] = {"tag": local_name(node.tag)}
    if attrs:
        result["attributes"] = attrs
    if text:
        result["text"] = text
    return result


def xml_node_to_value(node: ET.Element) -> dict[str, Any]:
    root = xml_node_shell(node)
    stack: list[tuple[ET.Element, dict[str, Any]]] = [(node, root)]
    while stack:
        current, result = stack.pop()
        child_nodes = list(current)
        if child_nodes:
            children = [xml_node_shell(child) for child in child_nodes]
            result["children"] = children
            stack.extend(zip(child_nodes, children))
    return root


def parse_xmp_metadata(stream: Any) -> dict[str, Any] | None:
    if not isinstance(stream, PdfStream):
        return None
    raw = stream.data
    if not raw:
        return None
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        raise ValueError("invalid XMP metadata")

    packet: dict[str, Any] = {
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


def resolve_metadata_stream(resolver: Any, trailer: dict[str, Any]) -> dict[str, Any] | None:
    root = resolver.resolve_dict(trailer.get("Root"))
    if root is None:
        return None
    metadata = resolver.resolve(root.get("Metadata"))
    if isinstance(metadata, PdfStream):
        return parse_xmp_metadata(metadata)
    if metadata is not None:
        raise ValueError("invalid Metadata stream")
    return None
