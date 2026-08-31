# SPDX-License-Identifier: AGPL-3.0-only
"""Public records returned by extraction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Generic, Self, TypeVar

from core_pdf.impl.types import Rectangle

RecordT = TypeVar("RecordT")


@dataclass(frozen=True, slots=True)
class PageScoped(Generic[RecordT]):
    """A page-level extraction record with its document-level context."""

    page_index: int
    page_number: int
    page_label: str | None
    record: RecordT


@dataclass(frozen=True, slots=True)
class TextWord:
    """One canonical word record shared by layout and structured extraction."""

    text: str
    bbox: Rectangle | None = None
    line_index: int = 0
    word_index: int = 0
    block_index: int = 0
    page_number: int | None = None
    source: str = "unknown"


def internal_text_word_tokens(text: str) -> tuple[str, ...]:
    """Return the non-whitespace word vocabulary used throughout extraction."""
    return tuple(text.split())


def internal_reconcile_text_words(
    text: str,
    words: tuple[TextWord, ...],
) -> tuple[TextWord, ...]:
    """Keep word text and geometry consistent after line-level normalization.

    Unchanged and one-to-one substituted words retain their boxes. If normalization
    changes word boundaries, geometry cannot be mapped safely and is left unknown.
    """
    tokens = internal_text_word_tokens(text)
    if not tokens:
        return ()
    if not words:
        return tuple(TextWord(token) for token in tokens)
    if tuple(word.text for word in words) == tokens:
        return words
    if len(words) == len(tokens):
        return tuple(replace(word, text=token) for word, token in zip(words, tokens, strict=True))
    return tuple(TextWord(token) for token in tokens)


@dataclass(frozen=True, slots=True)
class DrawingRecord:
    kind: str
    seqno: int
    fill: tuple[float, ...] | None
    fill_pattern: Mapping[object, object] | None
    fill_opacity: float | None
    stroke_color: tuple[float, ...] | None
    stroke_pattern: Mapping[object, object] | None
    stroke_opacity: float | None
    line_width: float
    line_cap: int
    line_join: int
    dash_pattern: tuple[list[float], float] | None
    fill_rule: str
    blend_mode: str | None
    soft_mask_alpha: float | None
    raw_data: bytes | memoryview | None
    dictionary: Mapping[object, object] | None
    image_source: object | None
    image_clip: Rectangle | None
    path: object | None
    items: tuple[object, ...]
    rect: Rectangle | None

    @classmethod
    def from_captured(cls, source: object, **overrides: object) -> Self:
        """Build a record from an object exposing the same fields."""
        values = {field.name: getattr(source, field.name) for field in fields(DrawingRecord)}
        values.update(overrides)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    width: int
    height: int
    channels: int
    color_model: str
    alpha: bool
    stride: int
    source_rect: Rectangle
    transform: object | None
    clipping: Rectangle | None


@dataclass(frozen=True, slots=True)
class ImageRecord(DrawingRecord):
    data: object | None = None
    image_metadata: ImageMetadata | None = None


__all__ = ("DrawingRecord", "ImageMetadata", "ImageRecord", "PageScoped", "TextWord")
