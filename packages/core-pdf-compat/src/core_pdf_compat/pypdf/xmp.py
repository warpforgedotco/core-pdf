# Follows pypdf's XMP handling, Copyright (c) 2006-2008, Mathieu Fenniak,
# under its BSD-style license.
"""pypdf's XMP metadata: an XML packet read and edited as a DOM."""

from __future__ import annotations

import datetime
import decimal
import re
import warnings
from collections.abc import Callable, Iterator
from string import hexdigits
from typing import Any
from xml.dom.expatbuilder import ExpatBuilderNS
from xml.dom.minidom import Document
from xml.dom.minidom import Element as XmlElement
from xml.parsers.expat import ExpatError

from .errors import LimitReachedError, PdfReadError, XmpDocumentError
from .generic import ContentStream, PdfObject

XMP_MAX_INPUT_LENGTH = 5_000_000
XMP_MAX_ELEMENT_COUNT = 100_000

RDF_NAMESPACE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
DC_NAMESPACE = "http://purl.org/dc/elements/1.1/"
XMP_NAMESPACE = "http://ns.adobe.com/xap/1.0/"
PDF_NAMESPACE = "http://ns.adobe.com/pdf/1.3/"
XMPMM_NAMESPACE = "http://ns.adobe.com/xap/1.0/mm/"
# Keys in this namespace escape invalid XML name characters as U+2182 and four hex digits.
PDFX_NAMESPACE = "http://ns.adobe.com/pdfx/1.3/"
PDFAID_NAMESPACE = "http://www.aiim.org/pdfa/ns/id/"

_NAMESPACE_PREFIX_MAP = {
    DC_NAMESPACE: "dc",
    XMP_NAMESPACE: "xmp",
    PDF_NAMESPACE: "pdf",
    XMPMM_NAMESPACE: "xmpMM",
    PDFAID_NAMESPACE: "pdfaid",
    PDFX_NAMESPACE: "pdfx",
}

iso8601 = re.compile(
    """
        (?P<year>[0-9]{4})
        (-
            (?P<month>[0-9]{2})
            (-
                (?P<day>[0-9]+)
                (T
                    (?P<hour>[0-9]{2}):
                    (?P<minute>[0-9]{2})
                    (:(?P<second>[0-9]{2}(.[0-9]+)?))?
                    (?P<tzd>Z|[-+][0-9]{2}:[0-9]{2})
                )?
            )?
        )?
    """,
    re.VERBOSE,
)

_MINIMAL_XMP = f"""<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="pypdf">
    <rdf:RDF xmlns:rdf="{RDF_NAMESPACE}">
        <rdf:Description rdf:about=""
            xmlns:dc="{DC_NAMESPACE}"
            xmlns:xmp="{XMP_NAMESPACE}"
            xmlns:pdf="{PDF_NAMESPACE}"
            xmlns:xmpMM="{XMPMM_NAMESPACE}"
            xmlns:pdfaid="{PDFAID_NAMESPACE}"
            xmlns:pdfx="{PDFX_NAMESPACE}">
        </rdf:Description>
    </rdf:RDF>
</x:xmpmeta>
<?xpacket end="r"?>"""


def _identity(value: Any) -> Any:
    return value


def _converter_date(value: str) -> datetime.datetime:
    matches = iso8601.match(value)
    if matches is None:
        raise ValueError(f"Invalid date format: {value}")
    second = decimal.Decimal(matches.group("second") or "0")
    whole_seconds = second.to_integral(decimal.ROUND_FLOOR)
    date = datetime.datetime(  # noqa: DTZ001
        int(matches.group("year")),
        int(matches.group("month") or "1"),
        int(matches.group("day") or "1"),
        int(matches.group("hour") or "0"),
        int(matches.group("minute") or "0"),
        int(whole_seconds),
        int((second - whole_seconds) * 1_000_000),
    )
    offset = matches.group("tzd") or "Z"
    if offset != "Z":
        hours, minutes = (int(part) for part in offset.split(":"))
        hours *= -1
        if hours < 0:
            minutes *= -1
        date += datetime.timedelta(hours=hours, minutes=minutes)
    return date


def _format_datetime_utc(value: datetime.datetime) -> str:
    """A UTC time with a trailing Z. A naive value is taken to be UTC already."""
    if value.tzinfo is not None and value.utcoffset() is not None:
        value = value.astimezone(datetime.UTC)
    return value.replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _format_date(value: Any) -> str:
    return _format_datetime_utc(value) if isinstance(value, datetime.datetime) else str(value)


class _XmpBuilder(ExpatBuilderNS):
    """Parses XML, refusing entity declarations and capping the element count.

    libexpat's own limits stop exponential entity expansion, not quadratic expansion.
    """

    def __init__(self, options: Any = None) -> None:
        super().__init__(options=options)
        self._element_count = 0

    def _deny_entity(self, entity_name: str, *_: object) -> None:
        raise ExpatError(f"Forbidden entities: {entity_name!r}")

    def start_element_handler(self, name: str, attributes: list[str]) -> None:
        self._element_count += 1
        if self._element_count > XMP_MAX_ELEMENT_COUNT:
            raise LimitReachedError(
                f"XMP metadata exceeds limit of {XMP_MAX_ELEMENT_COUNT} elements."
            )
        super().start_element_handler(name=name, attributes=attributes)

    def install(self, parser: Any) -> None:
        super().install(parser)
        parser.EntityDeclHandler = self._deny_entity
        parser.StartElementHandler = self.start_element_handler


def _container_items(element: XmlElement, kind: str) -> list[XmlElement] | None:
    containers = element.getElementsByTagNameNS(RDF_NAMESPACE, kind)
    if not len(containers):
        return None
    return [
        item
        for container in containers
        for item in container.getElementsByTagNameNS(RDF_NAMESPACE, "li")
    ]


def _unescape_key(key: str) -> str:
    start = 0
    while (index := key.find("ↂ", start)) != -1:
        digits = key[index + 1 : index + 5]
        if len(digits) == 4 and all(digit in hexdigits for digit in digits):
            key = key[:index] + chr(int(digits, base=16)) + key[index + 5 :]
        start = index + 1
    return key


class XmpInformation(PdfObject):
    def __init__(self, stream: ContentStream) -> None:
        self.stream = stream
        try:
            data = self.stream.get_data()
            if (length := len(data)) > XMP_MAX_INPUT_LENGTH:
                raise LimitReachedError(
                    f"XMP stream size {length} exceeds limit of {XMP_MAX_INPUT_LENGTH}."
                )
            document: Document = _XmpBuilder().parseString(data)
        except (AttributeError, ExpatError) as error:
            raise PdfReadError(f"XML in XmpInformation was invalid: {error}") from error
        roots = document.getElementsByTagNameNS(RDF_NAMESPACE, "RDF")
        if not roots:
            raise PdfReadError("XML in XmpInformation was invalid: Missing rdf:RDF root element")
        self.rdf_root: XmlElement = roots[0]
        self.cache: dict[Any, Any] = {}

    @classmethod
    def create(cls) -> XmpInformation:
        stream = ContentStream(None, None)
        stream.set_data(_MINIMAL_XMP.encode("utf-8"))
        return cls(stream)

    def write_to_stream(self, stream: Any, encryption_key: object = None) -> None:
        warnings.warn(
            "XmpInformation.write_to_stream is deprecated and will be removed in pypdf "
            "6.0.0. Use PdfWriter.xmp_metadata instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if encryption_key is not None:
            warnings.warn(
                "the encryption_key parameter of write_to_stream is deprecated and was "
                "removed in pypdf 5.0.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.stream.write_to_stream(stream)

    def _descriptions(self, about_uri: str) -> Iterator[XmlElement]:
        for description in self.rdf_root.getElementsByTagNameNS(RDF_NAMESPACE, "Description"):
            if description.getAttributeNS(RDF_NAMESPACE, "about") == about_uri:
                yield description

    def get_element(self, about_uri: str, namespace: str, name: str) -> Iterator[Any]:
        for description in self._descriptions(about_uri):
            attribute = description.getAttributeNodeNS(namespace, name)
            if attribute is not None:
                yield attribute
            yield from description.getElementsByTagNameNS(namespace, name)

    def get_nodes_in_namespace(self, about_uri: str, namespace: str) -> Iterator[Any]:
        for description in self._descriptions(about_uri):
            for index in range(description.attributes.length):
                attribute = description.attributes.item(index)
                if attribute and attribute.namespaceURI == namespace:
                    yield attribute
            for child in description.childNodes:
                if child.namespaceURI == namespace:
                    yield child

    def _get_text(self, element: XmlElement) -> str:
        nodes = element.childNodes
        return "".join(node.data for node in nodes if node.nodeType == node.TEXT_NODE)

    def _node_text(self, node: Any) -> str:
        return node.nodeValue if node.nodeType == node.ATTRIBUTE_NODE else self._get_text(node)

    def _cached(self, namespace: str, name: str, compute: Callable[[], Any]) -> Any:
        cached = self.cache.get(namespace, {}).get(name)
        if cached:
            return cached
        value = compute()
        self.cache.setdefault(namespace, {})[name] = value
        return value

    def _get_single_value(
        self, namespace: str, name: str, converter: Callable[[str], Any] = _identity
    ) -> Any:
        def compute() -> Any:
            for node in self.get_element("", namespace, name):
                return converter(self._node_text(node))
            return None

        return self._cached(namespace, name, compute)

    def _getter_bag(self, namespace: str, name: str) -> list[str]:
        def compute() -> list[str]:
            values: list[str] = []
            for node in self.get_element("", namespace, name):
                # An attribute has no children, and holds one plain value.
                items = None if node.nodeType == node.ATTRIBUTE_NODE else self._items(node, "Bag")
                values.extend([self._node_text(node)] if items is None else items)
            return values

        return self._cached(namespace, name, compute)

    def _items(self, node: XmlElement, kind: str) -> list[str] | None:
        items = _container_items(node, kind)
        return None if items is None else [self._get_text(item) for item in items]

    def _get_seq_values(
        self, namespace: str, name: str, converter: Callable[[Any], Any] = _identity
    ) -> list[Any]:
        def compute() -> list[Any]:
            values: list[Any] = []
            for node in self.get_element("", namespace, name):
                if node.nodeType == node.ATTRIBUTE_NODE:
                    values.append(converter(self._node_text(node)))
                elif (seq := self._items(node, "Seq")) is not None:
                    values.extend(map(converter, seq))
                elif (bag := self._items(node, "Bag")) is not None:
                    # Some producers write a bag for dc:creator, which should be a sequence.
                    values.extend(bag)
                else:
                    values.append(converter(self._get_text(node)))
            return values

        return self._cached(namespace, name, compute)

    def _get_langalt_values(self, namespace: str, name: str) -> dict[Any, Any]:
        def compute() -> dict[Any, Any]:
            values: dict[Any, Any] = {}
            for node in self.get_element("", namespace, name):
                is_attribute = node.nodeType == node.ATTRIBUTE_NODE
                items = None if is_attribute else _container_items(node, "Alt")
                if items is None:
                    values["x-default"] = self._node_text(node)
                else:
                    for item in items:
                        values[item.getAttribute("xml:lang")] = self._get_text(item)
            return values

        return self._cached(namespace, name, compute)

    @property
    def custom_properties(self) -> dict[Any, Any]:
        if not hasattr(self, "_custom_properties"):
            self._custom_properties = {
                _unescape_key(node.localName): self._node_text(node)
                for node in self.get_nodes_in_namespace("", PDFX_NAMESPACE)
            }
        return self._custom_properties

    def _document(self) -> Document:
        document = self.rdf_root.ownerDocument
        if document is None:
            raise XmpDocumentError("XMP Document is None")
        return document

    def _get_or_create_description(self, about_uri: str = "") -> XmlElement:
        for description in self._descriptions(about_uri):
            return description
        description = self._document().createElementNS(RDF_NAMESPACE, "rdf:Description")
        description.setAttributeNS(RDF_NAMESPACE, "rdf:about", about_uri)
        self.rdf_root.appendChild(description)
        return description

    def _clear_cache_entry(self, namespace: str, name: str) -> None:
        namespace_cache = self.cache.get(namespace)
        if namespace_cache and name in namespace_cache:
            del namespace_cache[name]

    def _get_namespace_prefix(self, namespace: str) -> str:
        return _NAMESPACE_PREFIX_MAP.get(namespace, "unknown")

    def _update_stream(self) -> None:
        self.stream.set_data(self._document().toxml(encoding="utf-8"))

    def _clear_property(self, namespace: str, name: str) -> XmlElement:
        self._clear_cache_entry(namespace, name)
        description = self._get_or_create_description()
        for element in list(description.getElementsByTagNameNS(namespace, name)):
            if (parent := element.parentNode) is not None:
                parent.removeChild(element)
        if attribute := description.getAttributeNodeNS(namespace, name):
            description.removeAttributeNode(attribute)
        return description

    def _new_property(self, namespace: str, name: str) -> tuple[Document, XmlElement]:
        document = self._document()
        prefix = self._get_namespace_prefix(namespace)
        return document, document.createElementNS(namespace, f"{prefix}:{name}")

    def _set_single_value(self, namespace: str, name: str, value: str | None) -> None:
        description = self._clear_property(namespace, name)
        if value is not None:
            document, element = self._new_property(namespace, name)
            element.appendChild(document.createTextNode(str(value)))
            description.appendChild(element)
        self._update_stream()

    def _set_items(
        self, namespace: str, name: str, kind: str, items: list[tuple[str | None, Any]]
    ) -> None:
        description = self._clear_property(namespace, name)
        if items:
            document, element = self._new_property(namespace, name)
            container = document.createElementNS(RDF_NAMESPACE, f"rdf:{kind}")
            for lang, value in items:
                item = document.createElementNS(RDF_NAMESPACE, "rdf:li")
                if lang is not None:
                    item.setAttribute("xml:lang", lang)
                item.appendChild(document.createTextNode(str(value)))
                container.appendChild(item)
            element.appendChild(container)
            description.appendChild(element)
        self._update_stream()

    def _set_bag_values(self, namespace: str, name: str, values: list[str] | None) -> None:
        self._set_items(namespace, name, "Bag", [(None, value) for value in values or ()])

    def _set_seq_values(self, namespace: str, name: str, values: list[str] | None) -> None:
        self._set_items(namespace, name, "Seq", [(None, value) for value in values or ()])

    def _set_langalt_values(self, namespace: str, name: str, values: dict[str, str] | None) -> None:
        self._set_items(namespace, name, "Alt", list((values or {}).items()))


def _single(namespace: str, name: str) -> property:
    return property(
        lambda self: self._get_single_value(namespace, name),
        lambda self, value: self._set_single_value(namespace, name, value),
    )


def _date(namespace: str, name: str) -> property:
    return property(
        lambda self: self._get_single_value(namespace, name, _converter_date),
        lambda self, value: self._set_single_value(
            namespace, name, _format_datetime_utc(value) if value else None
        ),
    )


def _bag(namespace: str, name: str) -> property:
    return property(
        lambda self: self._getter_bag(namespace, name),
        lambda self, values: self._set_bag_values(namespace, name, values),
    )


def _langalt(namespace: str, name: str) -> property:
    return property(
        lambda self: self._get_langalt_values(namespace, name),
        lambda self, values: self._set_langalt_values(namespace, name, values),
    )


_PROPERTIES = {
    "dc_contributor": _bag(DC_NAMESPACE, "contributor"),
    "dc_coverage": _single(DC_NAMESPACE, "coverage"),
    "dc_creator": property(
        lambda self: self._get_seq_values(DC_NAMESPACE, "creator"),
        lambda self, values: self._set_seq_values(DC_NAMESPACE, "creator", values),
    ),
    "dc_date": property(
        lambda self: self._get_seq_values(DC_NAMESPACE, "date", _converter_date),
        lambda self, values: self._set_seq_values(
            DC_NAMESPACE, "date", None if values is None else [_format_date(v) for v in values]
        ),
    ),
    "dc_description": _langalt(DC_NAMESPACE, "description"),
    "dc_format": _single(DC_NAMESPACE, "format"),
    "dc_identifier": _single(DC_NAMESPACE, "identifier"),
    "dc_language": _bag(DC_NAMESPACE, "language"),
    "dc_publisher": _bag(DC_NAMESPACE, "publisher"),
    "dc_relation": _bag(DC_NAMESPACE, "relation"),
    "dc_rights": _langalt(DC_NAMESPACE, "rights"),
    "dc_source": _single(DC_NAMESPACE, "source"),
    "dc_subject": _bag(DC_NAMESPACE, "subject"),
    "dc_title": _langalt(DC_NAMESPACE, "title"),
    "dc_type": _bag(DC_NAMESPACE, "type"),
    "pdf_keywords": _single(PDF_NAMESPACE, "Keywords"),
    "pdf_pdfversion": _single(PDF_NAMESPACE, "PDFVersion"),
    "pdf_producer": _single(PDF_NAMESPACE, "Producer"),
    "xmp_create_date": _date(XMP_NAMESPACE, "CreateDate"),
    "xmp_modify_date": _date(XMP_NAMESPACE, "ModifyDate"),
    "xmp_metadata_date": _date(XMP_NAMESPACE, "MetadataDate"),
    "xmp_creator_tool": _single(XMP_NAMESPACE, "CreatorTool"),
    "xmpmm_document_id": _single(XMPMM_NAMESPACE, "DocumentID"),
    "xmpmm_instance_id": _single(XMPMM_NAMESPACE, "InstanceID"),
    "pdfaid_part": _single(PDFAID_NAMESPACE, "part"),
    "pdfaid_conformance": _single(PDFAID_NAMESPACE, "conformance"),
}
for _name, _property in _PROPERTIES.items():
    setattr(XmpInformation, _name, _property)
