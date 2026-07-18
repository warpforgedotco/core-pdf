# SPDX-License-Identifier: AGPL-3.0-only
"""Immutable, format-neutral document records."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

BBox: TypeAlias = tuple[float, float, float, float]
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class BlockKind(StrEnum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    QUOTE = "quote"
    CODE = "code"
    FOOTNOTE = "footnote"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TextLine:
    text: str
    break_before: int = 1
    bbox: BBox | None = None
    advance_bbox: BBox | None = None
    ink_bbox: BBox | None = None
    kind: str = "text-line"
    source: str = "unknown"
    confidence: float | None = None
    baseline: BBox | None = None
    contributing_sources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Block:
    order: int
    kind: BlockKind
    lines: tuple[TextLine, ...] = ()
    bbox: BBox | None = None
    column_index: int | None = None
    rotation: int = 0
    confidence: float | None = None
    level: int | None = None
    provenance: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        parts: list[str] = []
        for line in self.lines:
            if parts:
                parts.append("\n" * max(1, line.break_before))
            parts.append(line.text)
        return "".join(parts)


@dataclass(frozen=True, slots=True)
class Page:
    page_number: int
    page_label: str | None = None
    width: float = 0.0
    height: float = 0.0
    rotation: int = 0
    blocks: tuple[Block, ...] = ()
    page_class: str = "unknown"
    base_route: str = "unknown"
    confidence: float | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks)


@dataclass(frozen=True, slots=True)
class Diagnostic:
    code: str
    message: str
    severity: str = "warning"
    page_number: int | None = None


@dataclass(frozen=True, slots=True)
class Document:
    pages: tuple[Page, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: tuple[Diagnostic, ...] = ()
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def text(self) -> str:
        return "\f".join(page.text for page in self.pages) + "\f"

    def to_json_dict(self) -> dict[str, JsonValue]:
        from core_document.serialization import document_to_json_dict

        return document_to_json_dict(self)

    def to_json(self, *, indent: int | None = 2, sort_keys: bool = True) -> str:
        from core_document.serialization import document_to_json

        return document_to_json(self, indent=indent, sort_keys=sort_keys)

    def to_markdown(self) -> str:
        from core_document.serialization import document_to_markdown

        return document_to_markdown(self)

    def to_html(self) -> str:
        from core_document.serialization import document_to_html

        return document_to_html(self)
