from __future__ import annotations

import contextlib
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from core_pdf import PdfDocument
from core_pdf.impl.recovery_xref import XRefScanner
from core_pdf.impl.types import Record, frozen_setattr
from core_pdf_spec.s_07_content.inline_images import validate_inline_images
from core_pdf_spec.s_07_filters.errors import FilterParseError

from ..pypdf import validate_pypdf_page_tree
from ._operator_text import OperatorTextProjection


class MetadataMode(StrEnum):
    NONE = "none"
    LLM = "llm"
    EMBED = "embed"
    ALL = "all"


def _metadata_text(metadata: dict[str, Any], excluded: frozenset[str], template: str) -> str:
    values = [(key, value) for key, value in metadata.items() if key not in excluded]
    if not values:
        return ""
    rendered = "\n".join(f"{key}: {value}" for key, value in values)
    return template.format(metadata=rendered)


class _MetadataMixin:
    __slots__ = ()

    text: str
    metadata: dict[str, Any]
    excluded_llm_metadata_keys: frozenset[str]
    excluded_embed_metadata_keys: frozenset[str]
    metadata_template: str
    text_template: str

    def get_content(self, metadata_mode: object = None) -> str:
        mode = str(getattr(metadata_mode, "value", metadata_mode or MetadataMode.NONE)).casefold()
        if mode in {"none", "metadata_mode.none"}:
            return self.text
        excluded = (
            self.excluded_llm_metadata_keys
            if mode.endswith("llm")
            else self.excluded_embed_metadata_keys
        )
        metadata = _metadata_text(self.metadata, excluded, self.metadata_template)
        return self.text_template.format(metadata_str=metadata, content=self.text).strip()

    def get_metadata_str(self, mode: object = None) -> str:
        value = str(getattr(mode, "value", mode or MetadataMode.ALL)).casefold()
        excluded = (
            self.excluded_llm_metadata_keys
            if value.endswith("llm")
            else (self.excluded_embed_metadata_keys if value.endswith("embed") else frozenset())
        )
        return _metadata_text(self.metadata, excluded, self.metadata_template)


class Document(Record, _MetadataMixin):
    __slots__ = (
        "text",
        "metadata",
        "id_",
        "excluded_llm_metadata_keys",
        "excluded_embed_metadata_keys",
        "metadata_template",
        "text_template",
    )

    text: str
    metadata: dict[str, Any]
    id_: str
    excluded_llm_metadata_keys: frozenset[str]
    excluded_embed_metadata_keys: frozenset[str]
    metadata_template: str
    text_template: str

    __fields__: ClassVar[tuple[str, ...]] = (
        "text",
        "metadata",
        "id_",
        "excluded_llm_metadata_keys",
        "excluded_embed_metadata_keys",
        "metadata_template",
        "text_template",
    )
    __match_args__ = (
        "text",
        "metadata",
        "id_",
        "excluded_llm_metadata_keys",
        "excluded_embed_metadata_keys",
        "metadata_template",
        "text_template",
    )

    def __init__(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        id_: str = "",
        excluded_llm_metadata_keys: frozenset[str] = frozenset(),
        excluded_embed_metadata_keys: frozenset[str] = frozenset(),
        metadata_template: str = "{metadata}",
        text_template: str = "{metadata_str}\n\n{content}",
    ) -> None:
        frozen_setattr(self, "text", text)
        frozen_setattr(self, "metadata", {} if metadata is None else metadata)
        frozen_setattr(self, "id_", id_)
        frozen_setattr(self, "excluded_llm_metadata_keys", excluded_llm_metadata_keys)
        frozen_setattr(self, "excluded_embed_metadata_keys", excluded_embed_metadata_keys)
        frozen_setattr(self, "metadata_template", metadata_template)
        frozen_setattr(self, "text_template", text_template)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.text == other.text
            and self.metadata == other.metadata
            and self.id_ == other.id_
            and self.excluded_llm_metadata_keys == other.excluded_llm_metadata_keys
            and self.excluded_embed_metadata_keys == other.excluded_embed_metadata_keys
            and self.metadata_template == other.metadata_template
            and self.text_template == other.text_template
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.text,
                self.metadata,
                self.id_,
                self.excluded_llm_metadata_keys,
                self.excluded_embed_metadata_keys,
                self.metadata_template,
                self.text_template,
            )
        )

    @property
    def doc_id(self) -> str:
        return self.id_ or str(self.metadata.get("doc_id", ""))

    def to_dict(self) -> dict[str, Any]:
        return {"id_": self.doc_id, "text": self.text, "metadata": dict(self.metadata)}


class Node(Record, _MetadataMixin):
    __slots__ = (
        "text",
        "node_id",
        "metadata",
        "relationships",
        "id_",
        "excluded_llm_metadata_keys",
        "excluded_embed_metadata_keys",
        "metadata_template",
        "text_template",
    )

    text: str
    node_id: str
    metadata: dict[str, Any]
    relationships: dict[str, str]
    id_: str
    excluded_llm_metadata_keys: frozenset[str]
    excluded_embed_metadata_keys: frozenset[str]
    metadata_template: str
    text_template: str

    __fields__: ClassVar[tuple[str, ...]] = (
        "text",
        "node_id",
        "metadata",
        "relationships",
        "id_",
        "excluded_llm_metadata_keys",
        "excluded_embed_metadata_keys",
        "metadata_template",
        "text_template",
    )
    __match_args__ = (
        "text",
        "node_id",
        "metadata",
        "relationships",
        "id_",
        "excluded_llm_metadata_keys",
        "excluded_embed_metadata_keys",
        "metadata_template",
        "text_template",
    )

    def __init__(
        self,
        text: str,
        node_id: str = "",
        metadata: dict[str, Any] | None = None,
        relationships: dict[str, str] | None = None,
        id_: str = "",
        excluded_llm_metadata_keys: frozenset[str] = frozenset(),
        excluded_embed_metadata_keys: frozenset[str] = frozenset(),
        metadata_template: str = "{metadata}",
        text_template: str = "{metadata_str}\n\n{content}",
    ) -> None:
        frozen_setattr(self, "text", text)
        frozen_setattr(self, "node_id", node_id)
        frozen_setattr(self, "metadata", {} if metadata is None else metadata)
        frozen_setattr(self, "relationships", {} if relationships is None else relationships)
        frozen_setattr(self, "id_", id_)
        frozen_setattr(self, "excluded_llm_metadata_keys", excluded_llm_metadata_keys)
        frozen_setattr(self, "excluded_embed_metadata_keys", excluded_embed_metadata_keys)
        frozen_setattr(self, "metadata_template", metadata_template)
        frozen_setattr(self, "text_template", text_template)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.text == other.text
            and self.node_id == other.node_id
            and self.metadata == other.metadata
            and self.relationships == other.relationships
            and self.id_ == other.id_
            and self.excluded_llm_metadata_keys == other.excluded_llm_metadata_keys
            and self.excluded_embed_metadata_keys == other.excluded_embed_metadata_keys
            and self.metadata_template == other.metadata_template
            and self.text_template == other.text_template
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.text,
                self.node_id,
                self.metadata,
                self.relationships,
                self.id_,
                self.excluded_llm_metadata_keys,
                self.excluded_embed_metadata_keys,
                self.metadata_template,
                self.text_template,
            )
        )

    @property
    def source_node(self) -> str | None:
        return self.relationships.get("source")

    @property
    def ref_doc_id(self) -> str | None:
        return self.source_node

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id or self.id_,
            "text": self.text,
            "metadata": dict(self.metadata),
            "relationships": dict(self.relationships),
        }


TextNode = Node


def load_data(
    source: object,
    *,
    max_characters: int = 2000,
    extra_info: Mapping[str, Any] | None = None,
    **kwargs: object,
) -> list[Document]:
    del kwargs, max_characters
    source_path = Path(source)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    source_data = source_path.read_bytes()
    if b"startxref" not in source_data or b"%%EOF" not in source_data:
        raise ValueError("incomplete PDF cross-reference terminator")
    if XRefScanner.find_startxref(source_data) is None:
        raise ValueError("PDF does not contain a startxref marker")
    with PdfDocument.open(source_data) as pdf:
        validate_pypdf_page_tree(pdf)
        pages = pdf.pages
        for page in pages:
            for stream in page.content_streams:
                with contextlib.suppress(FilterParseError):
                    validate_inline_images(stream.data)
        labels = pdf.build_page_labels(page_count=len(pages)) if pages else None
        return [
            Document(
                OperatorTextProjection(page).extract_text(),
                {
                    **(dict(extra_info) if extra_info is not None else {}),
                    "page_label": (labels[page_number - 1] if labels else None) or str(page_number),
                    "file_name": source_path.name,
                },
            )
            for page_number, page in enumerate(pages, 1)
        ]


def get_nodes_from_documents(source: object, *, max_characters: int = 2000) -> list[Node]:
    if isinstance(source, (list, tuple)) and all(isinstance(item, Document) for item in source):
        documents = list(source)
    else:
        documents = load_data(source, max_characters=max_characters)
    return [
        Node(
            text=document.text,
            node_id=f"node-{index}",
            metadata=document.metadata,
            relationships={"source": f"page-{document.metadata.get('page_label', index + 1)}"},
        )
        for index, document in enumerate(documents)
    ]


__all__ = (
    "Document",
    "MetadataMode",
    "Node",
    "TextNode",
    "get_nodes_from_documents",
    "load_data",
)
