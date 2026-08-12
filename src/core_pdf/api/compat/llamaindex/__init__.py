"""Local LlamaIndex-style document and node conversion APIs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

from core_pdf.api.compat._common import project_document

from core_pdf.api.compat.pypdf import PdfInput

from .._common import open_source


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
    """Shared ``get_content``/``get_metadata_str`` for the Document and Node facades."""

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


@dataclass(frozen=True, slots=True)
class Document(_MetadataMixin):
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id_: str = ""
    excluded_llm_metadata_keys: frozenset[str] = frozenset()
    excluded_embed_metadata_keys: frozenset[str] = frozenset()
    metadata_template: str = "{metadata}"
    text_template: str = "{metadata_str}\n\n{content}"

    @property
    def doc_id(self) -> str:
        return self.id_ or str(self.metadata.get("doc_id", ""))

    def to_dict(self) -> dict[str, Any]:
        return {"id_": self.doc_id, "text": self.text, "metadata": dict(self.metadata)}


@dataclass(frozen=True, slots=True)
class Node(_MetadataMixin):
    text: str
    node_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    relationships: dict[str, str] = field(default_factory=dict)
    id_: str = ""
    excluded_llm_metadata_keys: frozenset[str] = frozenset()
    excluded_embed_metadata_keys: frozenset[str] = frozenset()
    metadata_template: str = "{metadata}"
    text_template: str = "{metadata_str}\n\n{content}"

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
    del kwargs
    with open_source(cast(PdfInput, source)) as document:
        adapted = project_document(document)
        return [
            Document(
                chunk.text,
                {
                    **(dict(extra_info) if extra_info is not None else {}),
                    "page_numbers": chunk.page_numbers,
                    "element_ids": chunk.element_ids,
                },
            )
            for chunk in adapted.chunks(max_characters=max_characters)
        ]


def get_nodes_from_documents(source: object, *, max_characters: int = 2000) -> list[Node]:
    if isinstance(source, (list, tuple)) and all(isinstance(item, Document) for item in source):
        documents = list(cast(list[Document] | tuple[Document, ...], source))
    else:
        documents = load_data(source, max_characters=max_characters)
    return [
        Node(
            text=document.text,
            node_id=f"node-{index}",
            metadata=document.metadata,
            relationships={"source": f"page-{document.metadata['page_numbers'][0]}"},
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
