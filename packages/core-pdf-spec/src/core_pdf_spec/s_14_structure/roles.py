# SPDX-License-Identifier: AGPL-3.0-only
"""Namespace-aware structure role resolution (ISO 32000-2, 14.7 and 14.8.6)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from core_pdf_spec.exceptions import PdfUnsupportedError
from core_pdf_spec.s_07_syntax_primitives.text_string import decode_pdf_text_string
from core_pdf_spec.standards import PdfVersion, SemanticContext
from core_pdf_spec.types import PdfName, PdfReference, PdfString

PDF_1_7_NAMESPACE = "http://iso.org/pdf/ssn"
PDF_2_0_NAMESPACE = "http://iso.org/pdf2/ssn"
MATHML_NAMESPACE = "http://www.w3.org/1998/Math/MathML"

# ISO 32000-1:2008, 14.8.4, Tables 333-340. These identify the namespace's
# vocabulary, not the introduction version or permitted children of each tag.
PDF_1_7_STRUCTURE_TYPES = frozenset(
    [
        "Document",
        "Part",
        "Art",
        "Sect",
        "Div",
        "BlockQuote",
        "Caption",
        "TOC",
        "TOCI",
        "Index",
        "NonStruct",
        "Private",
        "P",
        "H",
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
        "L",
        "LI",
        "Lbl",
        "LBody",
        "Table",
        "TR",
        "TH",
        "TD",
        "THead",
        "TBody",
        "TFoot",
        "Span",
        "Quote",
        "Note",
        "Reference",
        "BibEntry",
        "Code",
        "Link",
        "Annot",
        "Ruby",
        "RB",
        "RT",
        "RP",
        "Warichu",
        "WT",
        "WP",
        "Figure",
        "Formula",
        "Form",
    ]
)
# ISO 32000-2:2020, Annex M; Hn for every positive integer is handled below.
PDF_2_0_STRUCTURE_TYPES = PDF_1_7_STRUCTURE_TYPES - frozenset(
    [
        "Art",
        "BlockQuote",
        "TOC",
        "TOCI",
        "Index",
        "Private",
        "Quote",
        "Note",
        "Reference",
        "BibEntry",
        "Code",
    ]
) | frozenset(["DocumentFragment", "Aside", "Title", "FENote", "Sub", "Em", "Strong", "Artifact"])


@dataclass(frozen=True, slots=True)
class StructureType:
    """A type name plus namespace identity; None means an undefined namespace."""

    name: str
    namespace: str | None


@dataclass(frozen=True, slots=True)
class StructureRole:
    """The terminal role and followed associations, without a conformance claim.

    Cycles are permitted in logical role maps (14.7.3, NOTE 2), so they terminate
    with ``cycle`` instead of a parsing error. ``domain`` identifies MathML's
    namespace; it does not validate its vocabulary or hierarchy.
    """

    name: str
    namespace: str | None
    status: Literal["standard", "domain", "unmapped", "cycle"]
    path: tuple[StructureType, ...]


def is_standard_structure_type(name: str, namespace: str) -> bool:
    if namespace == PDF_1_7_NAMESPACE:
        return name in PDF_1_7_STRUCTURE_TYPES
    if namespace == PDF_2_0_NAMESPACE:
        return name in PDF_2_0_STRUCTURE_TYPES or (
            name.startswith("H")
            and bool(name[1:])
            and name[1:].isascii()
            and name[1:].isdecimal()
            and name[1] != "0"
        )
    return False


def internal_identity(value: object) -> object:
    return value


def internal_name(value: object) -> str | None:
    return value.value if isinstance(value, PdfName) else None


def internal_namespace(
    value: object,
    resolve: Callable[[object], object],
    context: SemanticContext | None,
    decode_text: Callable[[bytes], str] | None,
) -> tuple[str, object]:
    dictionary = resolve(value)
    if not isinstance(dictionary, dict):
        raise ValueError("invalid structure namespace dictionary")
    kind = resolve(dictionary.get("Type"))
    if kind is not None and (not isinstance(kind, PdfName) or kind.value != "Namespace"):
        raise ValueError("invalid structure namespace Type")
    name = resolve(dictionary.get("NS"))
    if not isinstance(name, PdfString):
        raise ValueError("structure namespace NS must be a text string")
    namespace = (
        decode_pdf_text_string(name.data, context=context)
        if decode_text is None
        else decode_text(bytes(name.data))
    )
    return namespace, dictionary.get("RoleMapNS")


def internal_mapping_value(
    mapping: object, name: str, resolve: Callable[[object], object]
) -> object:
    dictionary = resolve(mapping)
    if dictionary is None:
        return None
    if not isinstance(dictionary, dict):
        raise ValueError("invalid structure role map dictionary")
    # The object model permits str, PdfName, and byte representations of decoded
    # dictionary keys. Do not strip a slash belonging to an already decoded key.
    for key, value in dictionary.items():
        decoded = (
            key.value
            if isinstance(key, PdfName)
            else key.decode("latin-1")
            if isinstance(key, bytes)
            else key
        )
        if decoded == name:
            return resolve(value)
    return None


def resolve_structure_role(
    name: object,
    *,
    namespace: object = None,
    role_map: object = None,
    resolve: Callable[[object], object] = internal_identity,
    resolve_name: Callable[[object], str | None] = internal_name,
    decode_text: Callable[[bytes], str] | None = None,
    context: SemanticContext | None = None,
) -> StructureRole:
    """Follow root RoleMap or namespace RoleMapNS associations transitively.

    ``namespace`` is the structure element's NS value, optionally already
    resolved. Array targets in RoleMapNS must contain an indirect namespace
    reference (14.7.4.2, Table 356). A single-name target enters the default
    standard namespace, and does not reapply the root RoleMap (14.8.6.2).
    The root map applies to an undefined namespace only. Its final standard
    type uses the PDF 1.7 namespace even in PDF 2.0 (14.8.6.1, corrected).

    Before PDF 1.5, standard types are not remapped (14.7.3, NOTE 3). Otherwise
    the starting type is mapped even if standard; subsequent associations stop
    upon a recognized type or a cycle (14.7.3). Namespace declarations are not
    rejected solely because a document under-declares their introduction.
    The caller owns malformed-input recovery and Namespaces-array validation.
    ``resolve_name`` and ``decode_text`` are parsing extension points for readers.
    """
    if context is not None and (context.version is None or not context.version.recognized):
        raise PdfUnsupportedError("structure role semantics require a recognized PDF version")
    type_name = resolve_name(resolve(name))
    if type_name is None:
        raise ValueError("structure type must be a PDF name")
    namespace_name: str | None = None
    mapping = role_map
    # Optional dictionary entries whose value resolves to null are absent
    # (7.3.7 / 7.3.9). Required RoleMapNS array targets below stay strict.
    namespace = resolve(namespace)
    if namespace is not None:
        namespace_name, mapping = internal_namespace(namespace, resolve, context, decode_text)
    current = StructureType(type_name, namespace_name)
    path = [current]
    seen = {current}
    first = True
    while True:
        effective_namespace = PDF_1_7_NAMESPACE if current.namespace is None else current.namespace
        standard = is_standard_structure_type(current.name, effective_namespace)
        if not first and standard:
            return StructureRole(current.name, effective_namespace, "standard", tuple(path))
        old_standard = (
            context is not None
            and context.version is not None
            and context.version < PdfVersion(1, 5)
            and current.namespace is None
            and standard
        )
        target = None if old_standard else internal_mapping_value(mapping, current.name, resolve)
        if target is None:
            status: Literal["standard", "domain", "unmapped", "cycle"] = (
                "standard"
                if standard
                else "domain"
                if current.namespace == MATHML_NAMESPACE
                else "unmapped"
            )
            return StructureRole(current.name, effective_namespace, status, tuple(path))
        first = False
        target_name = resolve_name(target)
        if target_name is not None:
            if current.namespace is None:
                next_namespace = None
                next_mapping = role_map
            else:
                next_namespace = PDF_1_7_NAMESPACE
                next_mapping = None
        else:
            if current.namespace is None or not isinstance(target, list) or len(target) != 2:
                raise ValueError("invalid structure role map target")
            target_name = resolve_name(resolve(target[0]))
            if target_name is None or not isinstance(target[1], PdfReference):
                raise ValueError("namespace role target requires a name and namespace reference")
            next_namespace, next_mapping = internal_namespace(
                target[1], resolve, context, decode_text
            )
        # RoleMapNS maps into another namespace; root RoleMap chains are in an
        # undefined namespace and therefore do not have this restriction.
        if current.namespace is not None and next_namespace == current.namespace:
            raise ValueError("namespace role mapping must target another namespace")
        current = StructureType(target_name, next_namespace)
        path.append(current)
        effective_namespace = PDF_1_7_NAMESPACE if current.namespace is None else current.namespace
        if is_standard_structure_type(current.name, effective_namespace):
            return StructureRole(current.name, effective_namespace, "standard", tuple(path))
        if current in seen:
            return StructureRole(current.name, effective_namespace, "cycle", tuple(path))
        seen.add(current)
        mapping = next_mapping


__all__ = (
    "MATHML_NAMESPACE",
    "PDF_1_7_NAMESPACE",
    "PDF_1_7_STRUCTURE_TYPES",
    "PDF_2_0_NAMESPACE",
    "PDF_2_0_STRUCTURE_TYPES",
    "StructureRole",
    "StructureType",
    "is_standard_structure_type",
    "resolve_structure_role",
)
