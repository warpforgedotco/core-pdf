"""High-level pikepdf-shaped facade backed by core-pdf."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, MutableMapping, MutableSequence
from decimal import Decimal
from os import PathLike
from typing import Any, cast, overload

from core_pdf import PdfDocument
from core_pdf.api.compat.pypdf import (
    PdfPageObject,
    PdfReader,
    StructuredState,
)
from core_pdf.impl._impl.output.model import Document
from core_pdf.impl._impl.output.model import Page as StructuredPage
from core_pdf.impl.exceptions import PdfParseError, PdfUnsupportedError
from core_pdf.impl.spec.s_07_document.document_labels import resolve_page_tree_node_type
from core_pdf.impl.spec.s_07_document.metadata import resolve_info_metadata
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.types import PdfName, PdfReference

from .._strict_page_tree import internal_has_malformed_shadowed_definition


class Rectangle(tuple[Decimal, Decimal, Decimal, Decimal]):
    """pikepdf-shaped numeric array preserving PDF decimal spelling semantics."""

    def __new__(cls, *values: object) -> "Rectangle":
        if len(values) != 4:
            raise ValueError("rectangle must contain four coordinates")
        coordinates = cast(
            tuple[Decimal, Decimal, Decimal, Decimal],
            tuple(Decimal(str(value)) for value in values),
        )
        return super().__new__(cls, coordinates)

    @property
    def width(self) -> Decimal:
        return self[2] - self[0]

    @property
    def height(self) -> Decimal:
        return self[3] - self[1]


class Array(list[Any]):
    def __repr__(self) -> str:
        values = ", ".join(_pike_repr(value) for value in self)
        return f"pikepdf.Array([ {values} ])"

    __str__ = __repr__


def _pike_repr(value: object) -> str:
    if isinstance(value, str):
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, dict):
        rows = [
            f'  "/{str(key).lstrip("/")}": {_pike_repr(item)}'
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        ]
        return "{\n" + ",\n".join(rows) + "\n}"
    return repr(value)


def _pike_value(value: object) -> object:
    if isinstance(value, PdfName):
        return f"/{value}"
    if isinstance(value, list):
        return Array(_pike_value(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _pike_value(item) for key, item in value.items()}
    if isinstance(value, str):
        return value.replace("\xad", "\ufffd")
    return value


def _pikepdf_info_metadata(pdf: PdfDocument) -> dict[str, Any]:
    info = pdf.trailer_dict.get("Info")
    if isinstance(info, PdfReference):
        entry = pdf.xref.get((info.object_number << 16) | info.generation_number)
        if entry is not None and entry.object_stream is None:
            lexer = PdfLexer(
                pdf.raw_data,
                reference_resolver=pdf.resolver.resolve,
                decipher=pdf.decipher,
                recover_dictionary_structure=False,
            )
            try:
                lexer.rewind(entry.offset)
                lexer.parse_indirect_object()
            except PdfParseError:
                return {}
            finally:
                lexer.close()
    return dict(resolve_info_metadata(pdf.resolver, pdf.trailer_dict, recover=True))


class Page(PdfPageObject):
    def __init__(
        self,
        document: StructuredState,
        page: Any,
        media_box: tuple[Decimal, Decimal, Decimal, Decimal] | None = None,
    ) -> None:
        super().__init__(document, page)
        engine_box = None
        if document.pdf is not None and media_box is None:
            engine_box = document.source_pdf.pages[page.page_number - 1].media_box
        self.mediabox = cast(Any, Rectangle(*(media_box or engine_box or self.mediabox)))
        self.cropbox = cast(Any, Rectangle(*self.cropbox))


def _validate_pikepdf_object_graph(document: StructuredState) -> None:
    if document.pdf is None:
        return
    pdf = document.source_pdf
    raw_data = bytes(pdf.raw_data)
    trailer_at = raw_data.rfind(b"trailer")
    if trailer_at >= 0:
        trailer = raw_data[trailer_at + 7 :]
        if raw_data.rfind(b"%%EOF", 0, trailer_at) < 0 and (
            not trailer.lstrip().startswith(b"<<")
            or (b"/Root" in trailer and re.search(rb"/Root\s+\d+\s+\d+\s+R\b", trailer) is None)
        ):
            raise PdfUnsupportedError("unable to find trailer dictionary")
    pages = pdf.catalog().get("Pages")
    if isinstance(pages, PdfReference) and internal_has_malformed_shadowed_definition(pdf, pages):
        raise PdfUnsupportedError("shadowed page tree root")
    if isinstance(pages, PdfReference) and pdf.xref_was_recovered:
        entry = pdf.xref.get((pages.object_number << 16) | pages.generation_number)
        if (
            pdf.strict_xref_validation_error() is not None
            and entry is not None
            and entry.object_stream is not None
        ):
            # qpdf's damaged-file reconstruction scans indirect objects but
            # cannot use a malformed xref stream to locate compressed objects.
            # Core's recovery can salvage that index, so preserve pikepdf's
            # stricter boundary when the page-tree root exists only there.
            raise PdfUnsupportedError("root of pages tree has no /Kids array")
    seen: set[tuple[str, int, int] | tuple[str, int]] = set()

    def visit(value: object) -> None:
        if isinstance(value, PdfReference):
            key = (value.object_number << 16) | value.generation_number
            if key not in pdf.xref:
                raise PdfUnsupportedError("invalid page tree reference")
        marker = (
            ("reference", value.object_number, value.generation_number)
            if isinstance(value, PdfReference)
            else ("object", id(value))
        )
        if marker in seen:
            raise PdfUnsupportedError("loop detected in page tree")
        resolved = pdf.resolver.resolve(value)
        if not isinstance(resolved, dict):
            return
        if resolve_page_tree_node_type(pdf.resolver, cast(PdfDict, resolved)) != "Pages":
            return
        kids = pdf.resolver.resolve(resolved.get("Kids"))
        if not isinstance(kids, list):
            return
        seen.add(marker)
        for kid in kids:
            visit(kid)

    visit(pages)


def _raw_indirect_object(pdf: PdfDocument, reference: PdfReference) -> bytes:
    entry = pdf.xref.get((reference.object_number << 16) | reference.generation_number)
    if entry is None:
        return b""
    if entry.object_stream is None:
        start = entry.offset
        data = bytes(pdf.raw_data)
        end = data.find(b"endobj", start)
        return data[start : end if end >= 0 else len(data)]
    pdf.resolver.resolve(reference)
    container = pdf.resolver.object_streams.get(entry.object_stream)
    if container is None or reference.object_number not in container.index:
        return b""
    start = container.index[reference.object_number]
    following = sorted(offset for offset in container.index.values() if offset > start)
    end = following[0] if following else len(container.raw_body)
    return bytes(container.raw_body[start:end])


def _raw_media_box(
    pdf: PdfDocument, reference: PdfReference
) -> tuple[Decimal, Decimal, Decimal, Decimal] | None:
    number = rb"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    match = re.search(
        rb"/MediaBox\s*\[\s*("
        + number
        + rb")\s+("
        + number
        + rb")\s+("
        + number
        + rb")\s+("
        + number
        + rb")\s*\]",
        _raw_indirect_object(pdf, reference),
    )
    if match is None:
        return None
    return cast(
        tuple[Decimal, Decimal, Decimal, Decimal],
        tuple(Decimal(value.decode("ascii")) for value in match.groups()),
    )


def _pikepdf_page_boxes(
    pdf: PdfDocument,
) -> tuple[tuple[Decimal, Decimal, Decimal, Decimal] | None, ...]:
    boxes: list[tuple[Decimal, Decimal, Decimal, Decimal] | None] = []

    def visit(
        value: object,
        inherited: tuple[Decimal, Decimal, Decimal, Decimal] | None = None,
    ) -> None:
        if not isinstance(value, PdfReference):
            return
        box = _raw_media_box(pdf, value) or inherited
        resolved = pdf.resolver.resolve(value)
        if not isinstance(resolved, dict):
            return
        node_type = resolve_page_tree_node_type(pdf.resolver, cast(PdfDict, resolved))
        if node_type == "Page":
            boxes.append(box)
            return
        if node_type != "Pages":
            return
        kids = pdf.resolver.resolve(resolved.get("Kids"))
        if isinstance(kids, list):
            for kid in kids:
                visit(kid, box)

    visit(pdf.catalog().get("Pages"))
    return tuple(boxes)


class Attachments(MutableMapping[str, bytes]):
    def __init__(self, values: dict[str, bytes] | None = None) -> None:
        self._values = dict(values or {})

    def __getitem__(self, key: str) -> bytes:
        return self._values[key]

    def __setitem__(self, key: str, value: bytes) -> None:
        self._values[str(key)] = bytes(value)

    def __delitem__(self, key: str) -> None:
        del self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


class DocumentInfo(MutableMapping[str, Any]):
    def __init__(self, owner: "Pdf", values: dict[str, Any]) -> None:
        self._owner = owner
        self._values = dict(values)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._values[str(key)] = value
        self._owner.metadata[str(key).lstrip("/")] = value

    def __delitem__(self, key: str) -> None:
        del self._values[key]
        self._owner.metadata.pop(key.lstrip("/"), None)

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


class Pages(MutableSequence[PdfPageObject]):
    def __init__(self, owner: "Pdf", values: tuple[PdfPageObject, ...] = ()) -> None:
        self._owner = owner
        self._values: list[PdfPageObject] = list(values)

    @overload
    def __getitem__(self, index: int) -> PdfPageObject: ...

    @overload
    def __getitem__(self, index: slice) -> list[PdfPageObject]: ...

    def __getitem__(self, index: int | slice) -> Any:
        return self._values[index]

    @overload
    def __setitem__(self, index: int, value: PdfPageObject) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[PdfPageObject]) -> None: ...

    def __setitem__(self, index: int | slice, value: Any) -> None:
        self._values[index] = value
        self._sync()

    def __delitem__(self, index: int | slice) -> None:
        del self._values[index]
        self._sync()

    def __len__(self) -> int:
        return len(self._values)

    def insert(self, index: int, value: PdfPageObject) -> None:
        self._values.insert(index, value)
        self._sync()

    def _sync(self) -> None:
        pages = tuple(item._page for item in self._values)
        self._owner._document = StructuredState.synthetic(
            Document(pages=pages, metadata=self._owner._document.structured.metadata)
        )
        self._values = [Page(self._owner._document, page) for page in pages]


class Pdf(PdfReader):
    pages: Any

    @classmethod
    def new(cls) -> Pdf:
        instance = cls.__new__(cls)
        instance._document = StructuredState.synthetic(Document())
        instance.__dict__["pages"] = Pages(instance)
        instance.metadata = {}
        instance.trailer = {}
        instance._attachments = Attachments()
        instance._outlines = []
        instance._source_name = None
        return instance

    @classmethod
    def open(cls, stream: object, *args: object, **kwargs: object) -> Pdf:
        del args
        password = kwargs.pop("password", None)
        if kwargs:
            raise TypeError(f"unsupported open options: {', '.join(kwargs)}")
        return cls(cast(str | PathLike[str] | bytes, stream), cast(str | None, password))

    def __init__(self, stream: object, password: str | None = None) -> None:
        self._source_name = str(stream) if isinstance(stream, (str, PathLike)) else None
        pdf = PdfDocument.open(cast(str | PathLike[str] | bytes, stream), password=password or "")
        try:
            pages = tuple(
                StructuredPage(
                    page_number=index,
                    width=page.width or 612.0,
                    height=page.height or 792.0,
                    rotation=page.rotation,
                    cropbox=page.crop_box,
                )
                for index, page in enumerate(pdf.pages, 1)
            )
            metadata = _pikepdf_info_metadata(pdf)
            self._document = StructuredState(pdf, Document(pages=pages, metadata=metadata))
            self.metadata = dict(metadata)
            self.trailer = {}
            _validate_pikepdf_object_graph(self._document)
            media_boxes = _pikepdf_page_boxes(pdf)
            self.__dict__["pages"] = Pages(
                self,
                tuple(
                    Page(
                        self._document,
                        page,
                        media_boxes[index] if index < len(media_boxes) else None,
                    )
                    for index, page in enumerate(pages)
                ),
            )
            try:
                outline = self.outline
            except (KeyError, TypeError, ValueError):
                outline = []
            self._outlines = [
                [item.level + 1, item.title, item.page + 1]
                for item in outline
                if item.page is not None
            ]
            self._attachments = Attachments(
                {
                    embedded.filename: embedded.data
                    for embedded in self._document.source_pdf.embedded_files()
                }
            )
        except Exception:
            pdf.close()
            raise

    @property
    def docinfo(self) -> DocumentInfo:
        values = {key: _pike_value(value) for key, value in self.metadata.items()}
        return DocumentInfo(self, values)

    @property
    def filename(self) -> str | None:
        return self._source_name

    @property
    def is_linearized(self) -> bool:
        return False

    @property
    def attachments(self) -> Attachments:
        return self._attachments


__all__ = ("Attachments", "DocumentInfo", "Pdf")
