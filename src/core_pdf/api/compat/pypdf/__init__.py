"""Supported high-level pypdf-shaped APIs backed by core-pdf."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from io import BytesIO
from os import PathLike
from typing import Any, cast

from core_pdf import PdfDocument
from core_pdf.api.compat._shared import ClosingMixin, coerce_bbox
from core_pdf.api.compat.pypdf._text import extract_legacy_text
from core_pdf.impl.exceptions import PdfUnsupportedError
from core_pdf.impl.output.model import (
    Document,
    Page,
)
from core_pdf.impl.primitives import PdfReference

PdfInput = str | PathLike[str] | bytes | bytearray | BytesIO


def internal_validate_pypdf_page_tree(pdf: PdfDocument) -> None:
    """Preserve pypdf's rejection of repeated/cyclic intermediate page nodes."""
    literal_trailers = tuple(pdf.iter_literal_trailer_dictionaries())
    if literal_trailers:
        latest_root = literal_trailers[-1].get("Root")
        if isinstance(latest_root, PdfReference):
            root_key = (latest_root.object_number << 16) | latest_root.generation_number
            if root_key not in pdf.xref:
                raise ValueError("catalog root references a missing object generation")
        elif latest_root is not None and not isinstance(latest_root, dict):
            raise ValueError("invalid catalog root")

    seen: set[tuple[int, int]] = set()

    def visit(value: object) -> None:
        if isinstance(value, PdfReference):
            key = (value.object_number, value.generation_number)
            if key in seen:
                raise ValueError("detected cyclic page references")
            seen.add(key)
        node = pdf.resolver.resolve(value)
        if not isinstance(node, dict):
            raise ValueError("invalid object in page tree")
        kids = pdf.resolver.resolve(node.get("Kids"))
        if isinstance(kids, (list, tuple)):
            for kid in kids:
                visit(kid)

    visit(pdf.catalog().get("Pages"))


class StructuredState(ClosingMixin):
    """Facade-local ownership of an engine document or synthetic structured snapshot."""

    def __init__(self, pdf: PdfDocument | None, structured: Document | None = None) -> None:
        self.pdf = pdf
        self._structured = structured

    @property
    def structured(self) -> Document:
        """Full structured snapshot, extracted lazily from the source document."""
        if self._structured is None:
            self._structured = self.source_pdf.structured_document
        return self._structured

    @classmethod
    def open(cls, source: PdfInput, *, password: str = "") -> "StructuredState":
        pdf = PdfDocument.open(source, password=password)
        try:
            if pdf.raw_data.find(b"startxref") < 0:
                raise ValueError("startxref not found")
            internal_validate_pypdf_page_tree(pdf)
        except Exception:
            pdf.close()
            raise
        return cls(pdf)

    @classmethod
    def synthetic(cls, structured: Document) -> "StructuredState":
        return cls(None, structured)

    @property
    def source_pdf(self) -> PdfDocument:
        if self.pdf is None:
            raise ValueError("synthetic snapshots do not have a source PDF")
        return self.pdf

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.structured.metadata

    @property
    def pages(self) -> tuple[Page, ...]:
        return self.structured.pages

    @property
    def form_fields(self) -> tuple[Any, ...]:
        return tuple(field for page in self.pages for field in page.form_fields)

    def capability_document(self) -> PdfDocument:
        if self.pdf is None:
            raise PdfUnsupportedError("synthetic snapshots do not have PDF capabilities")
        return self.pdf

    def capability_page(self, page_number: int) -> Any:
        return self.capability_document().pages[page_number - 1]

    def replace_pages(self, pages: Sequence[Page]) -> "StructuredState":
        return StructuredState.synthetic(replace(self.structured, pages=tuple(pages)))

    def delete_page(self, page_number: int) -> "StructuredState":
        return self.replace_pages(
            tuple(page for index, page in enumerate(self.pages, 1) if index != page_number)
        )

    def update_metadata(self, values: Mapping[str, Any]) -> "StructuredState":
        return StructuredState.synthetic(
            replace(self.structured, metadata={**self.structured.metadata, **values})
        )

    def apply_redactions(self) -> "StructuredState":
        redactions = tuple(
            annotation.bbox
            for page in self.pages
            for annotation in page.annotations
            if (annotation.subtype or "").casefold() == "redact" and annotation.bbox
        )

        def covered(box: tuple[float, float, float, float] | None) -> bool:
            return bool(
                box
                and any(
                    box[0] >= redaction[0]
                    and box[1] >= redaction[1]
                    and box[2] <= redaction[2]
                    and box[3] <= redaction[3]
                    for redaction in redactions
                )
            )

        return self.replace_pages(
            tuple(
                replace(
                    page,
                    blocks=tuple(block for block in page.blocks if not covered(block.bbox)),
                )
                for page in self.pages
            )
        )

    def close(self) -> None:
        if self.pdf is not None:
            self.pdf.close()


class Destination:
    """Small pypdf-shaped bookmark destination."""

    def __init__(self, title: str, page: int | None, level: int = 0) -> None:
        self.title = title
        self.page = page
        self.level = level

    @property
    def typ(self) -> str:
        return "/Fit"


class Rectangle(tuple[float, float, float, float]):
    """Tuple-compatible pypdf rectangle with the common geometry accessors."""

    def __new__(cls, left: float, bottom: float, right: float, top: float) -> "Rectangle":
        return super().__new__(cls, (float(left), float(bottom), float(right), float(top)))

    @property
    def left(self) -> float:
        return self[0]

    @property
    def bottom(self) -> float:
        return self[1]

    @property
    def right(self) -> float:
        return self[2]

    @property
    def top(self) -> float:
        return self[3]

    @property
    def lower_left(self) -> tuple[float, float]:
        return (self.left, self.bottom)

    @property
    def lower_right(self) -> tuple[float, float]:
        return (self.right, self.bottom)

    @property
    def upper_left(self) -> tuple[float, float]:
        return (self.left, self.top)

    @property
    def upper_right(self) -> tuple[float, float]:
        return (self.right, self.top)

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom


class PdfPageObject:
    def __init__(self, document: StructuredState, page: Page) -> None:
        self._document = document
        self._page = page
        self.internal_text_override: str | None = None
        if document.pdf is not None and 0 < page.page_number <= len(document.pdf.pages):
            source_page = document.pdf.pages[page.page_number - 1]
            media_box = source_page.media_box or (0.0, 0.0, page.width, page.height)
            crop_box = source_page.crop_box or media_box
            self.mediabox = Rectangle(*media_box)
            self.cropbox = Rectangle(*crop_box)
            raw_rotation = source_page.inherited_values.get("Rotate")
            self.rotation = (
                int(raw_rotation)
                if isinstance(raw_rotation, (int, float))
                else source_page.rotation
            )
        else:
            self.mediabox = Rectangle(0, 0, page.width, page.height)
            self.cropbox = Rectangle(*(page.cropbox or self.mediabox))
            self.rotation = page.rotation

    def _capability_view(self) -> Any:
        return self._document.capability_page(self._page.page_number).structured_view

    def extract_text(self, *args: object, **kwargs: object) -> str:  # noqa: C901
        del args, kwargs
        if self.internal_text_override is not None:
            return self.internal_text_override
        # Preserve the legacy text-operator projection for untouched pages.
        source_page = self._document.pages[self._page.page_number - 1]
        if self._document.pdf is not None and self._page is source_page:
            page = self._document.capability_page(self._page.page_number)
            return extract_legacy_text(page)
        return "\n".join(line.text for block in self._page.blocks for line in block.lines)

    def rotate(self, angle: int) -> PdfPageObject:
        self._page = replace(self._page, rotation=(self.rotation + angle) % 360)
        self.rotation = self._page.rotation
        return self

    def scale_to(self, width: float, height: float) -> PdfPageObject:
        self._page = replace(self._page, width=width, height=height)
        self.mediabox = Rectangle(0, 0, width, height)
        self.cropbox = self.mediabox
        return self

    def set_cropbox(self, box: tuple[float, float, float, float]) -> PdfPageObject:
        rectangle = Rectangle(*box)
        self._page = replace(self._page, cropbox=coerce_bbox(tuple(rectangle)))
        self.cropbox = rectangle
        return self

    def transfer_rotation_to_content(self) -> None:
        """Apply the page rotation to its geometry and clear the page rotation."""
        if self.rotation % 180:
            self.mediabox = Rectangle(0, 0, self.mediabox.height, self.mediabox.width)
            self.cropbox = Rectangle(0, 0, self.cropbox.height, self.cropbox.width)
        self._page = replace(self._page, rotation=0)
        self.rotation = 0

    def merge_page(self, page: PdfPageObject | Page) -> PdfPageObject:
        base_text = self.extract_text()
        overlay_text = page.extract_text() if isinstance(page, PdfPageObject) else page.text
        overlay = page._page if isinstance(page, PdfPageObject) else page
        order = max((item.order for item in self._page.elements), default=-1) + 1
        blocks = tuple(replace(item, order=order + item.order) for item in overlay.blocks)
        tables = tuple(replace(item, order=order + item.order) for item in overlay.tables)
        figures = tuple(replace(item, order=order + item.order) for item in overlay.figures)
        self._page = replace(
            self._page,
            blocks=(*self._page.blocks, *blocks),
            tables=(*self._page.tables, *tables),
            figures=(*self._page.figures, *figures),
            links=(*self._page.links, *overlay.links),
            annotations=(*self._page.annotations, *overlay.annotations),
            form_fields=(*self._page.form_fields, *overlay.form_fields),
        )
        self.internal_text_override = "\n".join(filter(None, (base_text, overlay_text)))
        return self

    @property
    def annotations(self) -> tuple[Any, ...]:
        return self._page.annotations or self._capability_view().annotations

    @property
    def links(self) -> tuple[Any, ...]:
        return self._page.links or self._capability_view().links

    @property
    def form_fields(self) -> tuple[Any, ...]:
        return self._page.form_fields or self._capability_view().form_fields

    @property
    def images(self) -> tuple[Any, ...]:
        return tuple(self._document.capability_page(self._page.page_number).extract_images())


class internal_LockedPages:
    def __iter__(self) -> Any:
        raise PdfUnsupportedError("file has not been decrypted")

    def __len__(self) -> int:
        raise PdfUnsupportedError("file has not been decrypted")

    def __getitem__(self, index: object) -> Any:
        del index
        raise PdfUnsupportedError("file has not been decrypted")


class PdfReader(ClosingMixin):
    """Reader exposing the most-used pypdf page and metadata APIs."""

    pages: Any
    metadata: dict[str, Any]

    def __init__(
        self,
        stream: PdfInput,
        password: str | None = None,
        strict: bool = True,
        **kwargs: object,
    ) -> None:
        del strict, kwargs
        self._encrypted_source = stream.getvalue() if isinstance(stream, BytesIO) else stream
        try:
            document = StructuredState.open(self._encrypted_source, password=password or "")
        except PdfUnsupportedError:
            if password:
                raise
            self._document = cast(Any, None)
            self.pages = cast(Any, internal_LockedPages())
            self.metadata: dict[str, Any] = {}
            self.trailer: dict[str, Any] = {}
            self._decryption_pending = True
            return
        self._decryption_pending = False
        self._bind(document)
        self.trailer = {}

    def _bind(self, document: StructuredState) -> None:
        """Adopt ``document`` and drop any previously materialized pages/metadata."""
        self._document = document
        self.__dict__.pop("pages", None)
        self.__dict__.pop("metadata", None)

    def __getattr__(self, name: str) -> Any:
        if name in {"pages", "metadata"} and self.__dict__.get("_document") is not None:
            self._materialize()
            return self.__dict__[name]
        raise AttributeError(name)

    def _materialize(self) -> None:
        """Materialize page objects and merge engine info metadata lazily."""
        document = self._document
        self.pages = tuple(PdfPageObject(document, page) for page in document.pages)
        raw_metadata = document.source_pdf.get_metadata()
        metadata: dict[str, Any] = (
            dict(cast(Any, raw_metadata.get("info", {}))) if isinstance(raw_metadata, dict) else {}
        )
        metadata.update(document.metadata)
        self.metadata = metadata

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    def get_page(self, page_number: int) -> PdfPageObject:
        return self.pages[page_number]

    def get_page_number(self, page: PdfPageObject | Page) -> int:
        """Return the zero-based index for a page owned by this reader."""
        value = page._page if isinstance(page, PdfPageObject) else page
        candidate: PdfPageObject
        for index, candidate in enumerate(self.pages):
            if candidate._page is value or candidate._page == value:
                return index
        raise ValueError("page is not part of this reader")

    @property
    def outline(self) -> list[Destination]:
        if self._document is None:
            return []
        return [
            Destination(item.title, item.page_index, item.level)
            for item in self._document.source_pdf.iter_outlines()
        ]

    @property
    def outlines(self) -> list[Destination]:
        return self.outline

    def get_destination_page_number(self, destination: object) -> int:
        if isinstance(destination, Destination) and destination.page is not None:
            return destination.page
        raise ValueError("destination does not resolve to a page")

    def get_fields(self) -> dict[str, Any]:
        if self._document is None:
            return {}
        return {
            item.record.name: item.record
            for item in self._document.source_pdf.extract_form_fields()
        }

    def get_form_text_fields(self) -> dict[str, str]:
        if self._document is None:
            return {}
        return {
            field.name: field.value_text
            for item in self._document.source_pdf.extract_form_fields()
            for field in (item.record,)
            if field.type.casefold() in {"text", "tx"}
        }

    def get_form_xfa(self) -> None:
        return None

    def close(self) -> None:
        if self._document is not None:
            self._document.close()

    @property
    def is_encrypted(self) -> bool:
        return self._decryption_pending or bool(
            getattr(getattr(self._document, "pdf", None), "decipher", None)
        )

    def decrypt(self, password: str) -> bool:
        if not self._decryption_pending:
            return not self.is_encrypted
        try:
            document = StructuredState.open(self._encrypted_source, password=password)
        except PdfUnsupportedError:
            return False
        self._decryption_pending = False
        self._bind(document)
        return True


__all__ = (
    "Destination",
    "PdfPageObject",
    "PdfReader",
    "Rectangle",
)
