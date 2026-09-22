# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, Literal, NoReturn, Self

from core_pdf_spec.exceptions import PdfUnsupportedError
from core_pdf_spec.s_07_syntax_primitives.text_string import decode_pdf_text_string
from core_pdf_spec.standards import PdfVersion, SemanticContext
from core_pdf_spec.types import PdfName, PdfReference, PdfString

frozen_setattr = object.__setattr__


PDF_1_7_NAMESPACE = "http://iso.org/pdf/ssn"
PDF_2_0_NAMESPACE = "http://iso.org/pdf2/ssn"
MATHML_NAMESPACE = "http://www.w3.org/1998/Math/MathML"

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


class StructureType:
    __slots__ = ("name", "namespace")

    name: str
    namespace: str | None

    __fields__: ClassVar[tuple[str, ...]] = ("name", "namespace")
    __match_args__ = ("name", "namespace")

    def __init__(self, name: str, namespace: str | None) -> None:
        frozen_setattr(self, "name", name)
        frozen_setattr(self, "namespace", namespace)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(name={self.name!r}, namespace={self.namespace!r})"

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.name == other.name and self.namespace == other.namespace

    def __hash__(self) -> int:
        return hash((self.name, self.namespace))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        name = changes.pop("name", self.name)
        namespace = changes.pop("namespace", self.namespace)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(name, namespace)


class StructureRole:
    __slots__ = ("name", "namespace", "status", "path")

    name: str
    namespace: str | None
    status: Literal["standard", "domain", "unmapped", "cycle"]
    path: tuple[StructureType, ...]

    __fields__: ClassVar[tuple[str, ...]] = ("name", "namespace", "status", "path")
    __match_args__ = ("name", "namespace", "status", "path")

    def __init__(
        self,
        name: str,
        namespace: str | None,
        status: Literal["standard", "domain", "unmapped", "cycle"],
        path: tuple[StructureType, ...],
    ) -> None:
        frozen_setattr(self, "name", name)
        frozen_setattr(self, "namespace", namespace)
        frozen_setattr(self, "status", status)
        frozen_setattr(self, "path", path)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"name={self.name!r}, "
            f"namespace={self.namespace!r}, "
            f"status={self.status!r}, "
            f"path={self.path!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.name == other.name
            and self.namespace == other.namespace
            and self.status == other.status
            and self.path == other.path
        )

    def __hash__(self) -> int:
        return hash((self.name, self.namespace, self.status, self.path))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        name = changes.pop("name", self.name)
        namespace = changes.pop("namespace", self.namespace)
        status = changes.pop("status", self.status)
        path = changes.pop("path", self.path)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(name, namespace, status, path)


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


def identity(value: object) -> object:
    return value


def name(value: object) -> str | None:
    return value.value if isinstance(value, PdfName) else None


def resolve_namespace(
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


def mapping_value(mapping: object, name: str, resolve: Callable[[object], object]) -> object:
    dictionary = resolve(mapping)
    if dictionary is None:
        return None
    if not isinstance(dictionary, dict):
        raise ValueError("invalid structure role map dictionary")
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
    resolve: Callable[[object], object] = identity,
    resolve_name: Callable[[object], str | None] = name,
    decode_text: Callable[[bytes], str] | None = None,
    context: SemanticContext | None = None,
) -> StructureRole:
    if context is not None and (context.version is None or not context.version.recognized):
        raise PdfUnsupportedError("structure role semantics require a recognized PDF version")
    type_name = resolve_name(resolve(name))
    if type_name is None:
        raise ValueError("structure type must be a PDF name")
    namespace_name: str | None = None
    mapping = role_map
    namespace = resolve(namespace)
    if namespace is not None:
        namespace_name, mapping = resolve_namespace(namespace, resolve, context, decode_text)
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
        target = None if old_standard else mapping_value(mapping, current.name, resolve)
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
            next_namespace, next_mapping = resolve_namespace(
                target[1], resolve, context, decode_text
            )
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
