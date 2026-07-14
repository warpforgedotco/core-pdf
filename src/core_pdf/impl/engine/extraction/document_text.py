# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Protocol, cast

from core_pdf.impl.engine.extraction.page_text.engine import DocumentExtractionResult

if TYPE_CHECKING:
    from core_pdf.impl.engine.spec.s_07_document.metadata_types import (
        MetadataRecord,
        MetadataValue,
    )


class _DocumentTextPage(Protocol):
    def extract_text(self) -> str: ...

    def to_markdown(self) -> str: ...


class _DocumentTextOutline(Protocol):
    level: int
    title: str


class _DocumentTextHost(Protocol):
    @property
    def pages(self) -> Iterable[object]: ...

    def extract(self) -> DocumentExtractionResult: ...

    def get_metadata(self) -> MetadataRecord: ...

    def iter_outlines(self) -> Iterable[_DocumentTextOutline]: ...


def _metadata_markdown_value(value: MetadataValue) -> str:
    if isinstance(value, memoryview):
        return value.tobytes().decode("utf-8", errors="replace")
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


class DocumentTextMixin:
    def extract_text(self: _DocumentTextHost) -> str:
        return self.extract().text

    def to_markdown(self: _DocumentTextHost) -> str:
        md_parts: list[str] = []

        metadata = self.get_metadata()
        info = metadata.get("info", {})
        if info:
            md_parts.append("# Document Metadata")
            for k, v in info.items():
                if v:
                    md_parts.append(f"- **{k}**: {_metadata_markdown_value(v)}")
            md_parts.append("")

        try:
            outlines = self.iter_outlines()
            if outlines:
                md_parts.append("# Table of Contents")
                for item in outlines:
                    indent = "  " * item.level
                    md_parts.append(f"{indent}- {item.title}")
                md_parts.append("")
        except ValueError, KeyError, StopIteration:
            pass

        md_parts.append(
            "\f".join(
                cast(_DocumentTextPage, page_obj).to_markdown()
                for page_obj in self.pages
            )
        )

        return "\n".join(md_parts) + "\f"

    def iter_text(self: _DocumentTextHost) -> Iterator[str]:
        result = self.extract()
        if result.pages:
            for page_result in result.pages:
                yield page_result.text
            return
        for page_obj in self.pages:
            yield cast(_DocumentTextPage, page_obj).extract_text()


__all__ = ("DocumentTextMixin",)
