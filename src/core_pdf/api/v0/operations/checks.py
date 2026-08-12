"""Shared deterministic predicates and patterns used by multiple operations."""

from __future__ import annotations

import re

from ...models import AnnotationRecord

INTERACTIVE_ANNOTATION_SUBTYPES = frozenset({"link", "screen", "widget"})

DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)

AUTHOR_YEAR_PATTERN = re.compile(r"\b([A-Z][A-Za-z'’-]+(?:\s+et\s+al\.)?),\s*(\d{4}[a-z]?)\b")

REFERENCE_SECTION_HEADING_PATTERN = re.compile(
    r"^(?:references|bibliography|works cited|literature cited)\s*:?$", re.I
)


def is_interactive_annotation(annotation: AnnotationRecord) -> bool:
    """Whether the annotation subtype implies user interaction."""
    return annotation.subtype.strip().casefold() in INTERACTIVE_ANNOTATION_SUBTYPES


__all__ = (
    "AUTHOR_YEAR_PATTERN",
    "DOI_PATTERN",
    "INTERACTIVE_ANNOTATION_SUBTYPES",
    "REFERENCE_SECTION_HEADING_PATTERN",
    "is_interactive_annotation",
)
