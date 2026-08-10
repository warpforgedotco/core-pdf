# SPDX-License-Identifier: AGPL-3.0-only
"""Typed normalized elements and retrieval chunks derived from the structured IR."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core_pdf.impl.engine.structured.model import BBox, Document

if TYPE_CHECKING:
    from core_pdf.impl.models import ImageRecord, PageScoped


@dataclass(frozen=True, slots=True)
class ElementRecord:
    """One normalized reading-order element with its page context."""

    element_id: str
    kind: str
    text: str
    page_number: int
    bbox: BBox | None = None
    order: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return the stable dictionary shape consumed by the OSS-style facades."""
        return {
            "element_id": self.element_id,
            "type": self.kind,
            "text": self.text,
            "page_number": self.page_number,
            "bbox": self.bbox,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """A bounded, page-aware retrieval chunk assembled from element records."""

    text: str
    page_numbers: tuple[int, ...]
    element_ids: tuple[str, ...]
    element_types: tuple[str, ...]
    section_path: tuple[str, ...] = ()
    element_geometry: tuple[tuple[int, BBox], ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return the stable dictionary shape consumed by the OSS-style facades."""
        return {
            "text": self.text,
            "page_numbers": self.page_numbers,
            "element_ids": self.element_ids,
            "metadata": {
                "element_types": self.element_types,
                "element_geometry": self.element_geometry,
            },
        }


def document_elements(
    document: Document,
    images: Iterable[PageScoped[ImageRecord]] = (),
) -> tuple[ElementRecord, ...]:
    """Flatten a structured document (plus scoped images) into element records."""
    records: list[ElementRecord] = []
    for page in document.pages:
        for index, item in enumerate(page.elements):
            text = str(getattr(item, "text", ""))
            if not text and hasattr(item, "rows"):
                text = "\n".join(
                    " | ".join(str(cell.text) for cell in row) for row in getattr(item, "rows", ())
                )
            order = getattr(item, "order", index)
            records.append(
                ElementRecord(
                    element_id=f"p{page.page_number}-e{index}",
                    kind=type(item).__name__.casefold(),
                    text=text,
                    page_number=page.page_number,
                    bbox=getattr(item, "bbox", None),
                    order=order,
                    metadata={
                        "order": order,
                        "kind": getattr(getattr(item, "kind", None), "value", None),
                    },
                )
            )
        records.extend(
            ElementRecord(
                element_id=f"p{page.page_number}-annotation-{index}",
                kind="annotation",
                text=annotation.contents,
                page_number=page.page_number,
                bbox=annotation.bbox,
                metadata={
                    "subtype": annotation.subtype,
                    "destination": annotation.destination,
                },
            )
            for index, annotation in enumerate(page.annotations)
        )
        records.extend(
            ElementRecord(
                element_id=f"p{page.page_number}-form-{index}",
                kind="form_field",
                text=form_field.value_text,
                page_number=page.page_number,
                bbox=form_field.bbox,
                metadata={"name": form_field.name, "field_type": form_field.field_type},
            )
            for index, form_field in enumerate(page.form_fields)
        )
        records.extend(
            ElementRecord(
                element_id=f"p{page.page_number}-link-{index}",
                kind="link",
                text=link.text,
                page_number=page.page_number,
                bbox=link.bbox,
                metadata={"url": link.url, "link_type": link.link_type},
            )
            for index, link in enumerate(page.links)
        )
    records.extend(
        ElementRecord(
            element_id=f"p{item.page_number}-image-{index}",
            kind="image",
            text="",
            page_number=item.page_number,
            bbox=item.record.rect,
            metadata={
                "kind": item.record.kind,
                "data": item.record.data,
                "image_metadata": item.record.image_metadata,
            },
        )
        for index, item in enumerate(images)
    )
    return tuple(records)


def document_section_paths(document: Document) -> dict[int, tuple[str, ...]]:
    """Return the cumulative heading path in effect at the end of each page."""
    paths: dict[int, tuple[str, ...]] = {}
    current: list[str] = []
    for page in document.pages:
        for block in page.blocks:
            kind = str(getattr(getattr(block, "kind", None), "value", "")).casefold()
            if kind != "heading":
                continue
            level = getattr(block, "level", None)
            depth = int(level) if isinstance(level, int) and level > 0 else 1
            del current[depth - 1 :]
            current.append(str(getattr(block, "text", "")).strip())
        paths[page.page_number] = tuple(current)
    return paths


def chunk_elements(
    elements: Iterable[ElementRecord],
    *,
    max_characters: int = 2000,
    section_paths: Mapping[int, tuple[str, ...]] | None = None,
) -> tuple[ChunkRecord, ...]:
    """Pack element records into deterministic page-aware chunks."""
    if max_characters < 1:
        raise ValueError("max_characters must be positive")
    chunks: list[ChunkRecord] = []
    current: list[ElementRecord] = []
    size = 0
    for element in elements:
        addition = len(element.text) + (2 if current else 0)
        if current and size + addition > max_characters:
            chunks.append(_make_chunk(current, section_paths))
            current, size = [], 0
        current.append(element)
        size += len(element.text) + (2 if len(current) > 1 else 0)
    if current:
        chunks.append(_make_chunk(current, section_paths))
    return tuple(chunks)


def _make_chunk(
    items: list[ElementRecord],
    section_paths: Mapping[int, tuple[str, ...]] | None,
) -> ChunkRecord:
    pages = tuple(dict.fromkeys(item.page_number for item in items))
    section_path: tuple[str, ...] = ()
    if section_paths and pages:
        section_path = section_paths.get(pages[-1], ())
    return ChunkRecord(
        text="\n\n".join(item.text for item in items if item.text),
        page_numbers=pages,
        element_ids=tuple(item.element_id for item in items),
        element_types=tuple(item.kind for item in items),
        section_path=section_path,
        element_geometry=tuple(
            (item.page_number, item.bbox) for item in items if item.bbox is not None
        ),
    )


__all__ = (
    "ChunkRecord",
    "ElementRecord",
    "chunk_elements",
    "document_elements",
    "document_section_paths",
)
