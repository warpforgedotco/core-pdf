from __future__ import annotations

import io
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

from core_pdf.impl.types import GeneratedRecord, field

from .. import pypdf


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
        # As llama-index-core's TextNode: the text alone when the mode or the metadata is empty.
        mode = _mode(metadata_mode, MetadataMode.NONE)
        metadata = self.get_metadata_str(mode)
        if mode == "none" or not metadata:
            return self.text
        return self.text_template.format(metadata_str=metadata, content=self.text).strip()

    def get_metadata_str(self, mode: object = None) -> str:
        value = _mode(mode, MetadataMode.ALL)
        if value == "none":
            return ""
        excluded = (
            self.excluded_llm_metadata_keys
            if value == "llm"
            else (self.excluded_embed_metadata_keys if value == "embed" else frozenset())
        )
        return _metadata_text(self.metadata, excluded, self.metadata_template)


def _mode(mode: object, default: MetadataMode) -> str:
    value = str(getattr(mode, "value", mode or default)).casefold()
    return value.removeprefix("metadata_mode.").removeprefix("metadatamode.")


class Document(GeneratedRecord, _MetadataMixin):
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id_: str = ""
    excluded_llm_metadata_keys: frozenset[str] = frozenset()
    excluded_embed_metadata_keys: frozenset[str] = frozenset()
    metadata_template: str = "{metadata}"
    text_template: str = "{metadata_str}\n\n{content}"

    def get_content(self, metadata_mode: object = None) -> str:
        # llama-index-core keeps no text resource for a Document made with empty text, and a
        # Document without one has empty content whatever the metadata.
        if not self.text:
            return ""
        return super().get_content(metadata_mode)

    @property
    def doc_id(self) -> str:
        return self.id_ or str(self.metadata.get("doc_id", ""))

    def to_dict(self) -> dict[str, Any]:
        return {"id_": self.doc_id, "text": self.text, "metadata": dict(self.metadata)}


class Node(GeneratedRecord, _MetadataMixin):
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
    del kwargs, max_characters
    return PDFReader().load_data(source, extra_info=extra_info)


# PDFReader is ported from llama-index-readers-file (llama_index/readers/file/docs/base.py),
# Copyright (c) Jerry Liu, under the MIT License. It reads through the pypdf facade, as
# llama_index reads through pypdf, so text, page labels and errors are pypdf's.
class PDFReader:
    def __init__(self, return_full_document: bool | None = False) -> None:
        self.return_full_document = return_full_document

    def load_data(
        self,
        file: object,
        extra_info: Mapping[str, Any] | None = None,
        fs: Any = None,
    ) -> list[Document]:
        if not isinstance(file, (Path, PurePosixPath)):
            file = (Path if fs is None else PurePosixPath)(file)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        with open(file, "rb") if fs is None else fs.open(str(file), "rb") as fp:  # noqa: PTH123
            # A file from another filesystem is read into memory first, as llama_index does.
            stream = fp if fs is None else io.BytesIO(fp.read())
            pdf = pypdf.PdfReader(stream)
            num_pages = len(pdf.pages)
            docs = []
            if self.return_full_document:
                metadata: dict[str, Any] = {"file_name": file.name}
                if extra_info is not None:
                    metadata.update(extra_info)
                text = "\n".join(pdf.pages[page].extract_text() for page in range(num_pages))
                docs.append(Document(text, metadata))
            else:
                for page in range(num_pages):
                    page_text = pdf.pages[page].extract_text()
                    page_label = pdf.page_labels[page]
                    metadata = {"page_label": page_label, "file_name": file.name}
                    if extra_info is not None:
                        metadata.update(extra_info)
                    docs.append(Document(page_text, metadata))
            return docs


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
    "PDFReader",
    "Document",
    "MetadataMode",
    "Node",
    "TextNode",
    "get_nodes_from_documents",
    "load_data",
)
