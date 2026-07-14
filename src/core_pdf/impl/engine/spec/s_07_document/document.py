# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import mmap
import struct
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Iterator, TypeAlias

from core_pdf.impl.engine.spec.s_07_document.forms import FormsMixin
from core_pdf.impl.engine.spec.s_07_document.layers import LayersMixin
from core_pdf.impl.engine.spec.s_07_document.metadata import MetadataDict, resolve_metadata
from core_pdf.impl.engine.spec.s_07_document.models import FieldRecord, NamedDestination
from core_pdf.impl.engine.spec.s_07_document.navigation import NavigationMixin
from core_pdf.impl.engine.spec.s_07_document.page import PdfPage
from core_pdf.impl.engine.spec.s_07_document.structure import StructureTree
from core_pdf.impl.engine.spec.s_07_objects.resolver import ObjectResolver
from core_pdf.impl.engine.spec.s_07_security.crypto_handlers import SECURITY_HANDLER_REGISTRY
from core_pdf.impl.engine.spec.s_07_syntax.errors import (
    PdfParseError,
    PdfSourceError,
    PdfUnsupportedError,
)
from core_pdf.impl.engine.spec.s_07_syntax.lexer import DecipherFn
from core_pdf.impl.engine.spec.s_07_syntax.primitives import (
    PdfDictLike,
    PdfObject,
    PdfReference,
    PdfSource,
    PdfStream,
)
from core_pdf.impl.engine.spec.s_07_syntax.xref import PdfXRefEntry, XRefScanner

if TYPE_CHECKING:
    from core_pdf.impl.engine.spec.s_09_fonts.decoder import FontDecoder


MAX_PAGE_TREE_DEPTH = 100
PdfDict: TypeAlias = PdfDictLike
DecoderCache: TypeAlias = dict[tuple[int, int] | int, "FontDecoder"]
PdfDataBuffer: TypeAlias = bytes | mmap.mmap


class PdfDocument(NavigationMixin, FormsMixin, LayersMixin):
    __slots__ = (
        "source",
        "password",
        "raw_data",
        "xref",
        "trailer_dict",
        "decipher",
        "resolver",
        "file_handle",
        "catalog_cache",
        "metadata_cache",
        "structure_cache",
        "structure_root_cache",
        "mark_info_cache",
        "page_dicts_cache",
        "pages_cache",
        "page_index_cache",
        "named_destinations_cache",
        "oc_layers",
        "acroform_cache",
        "fields_cache",
        "decoder_cache",
        "inherited_values_cache",
    )

    source: PdfSource
    password: str
    raw_data: PdfDataBuffer
    xref: dict[int, PdfXRefEntry]
    trailer_dict: PdfDict
    decipher: DecipherFn | None
    resolver: ObjectResolver
    file_handle: BinaryIO | None
    catalog_cache: PdfDict | None
    metadata_cache: MetadataDict | None
    structure_cache: StructureTree | None
    structure_root_cache: PdfDict | None
    mark_info_cache: PdfDict | None
    page_dicts_cache: list[PdfDict] | None
    pages_cache: list[PdfPage] | None
    page_index_cache: dict[int, int] | None
    named_destinations_cache: dict[str, NamedDestination] | None
    oc_layers: dict[str, bool] | None
    acroform_cache: PdfDict | None
    fields_cache: list[FieldRecord] | None
    decoder_cache: DecoderCache
    inherited_values_cache: dict[int, dict[str, PdfObject]]

    def __init__(self, source: PdfSource, password: str = "") -> None:
        self.source = source
        self.password = password
        self.file_handle = None
        self.decipher = None
        self.xref = {}
        self.trailer_dict = {}

        self.raw_data = self.load_data(source)
        self.scan_xref()

        self.resolver = ObjectResolver(self.raw_data, self.xref, self.trailer_dict)
        self.init_security(password)
        self.resolver.decipher = self.decipher

        self.catalog_cache = None
        self.metadata_cache = None
        self.structure_cache = None
        self.structure_root_cache = None
        self.mark_info_cache = None
        self.page_dicts_cache = None
        self.pages_cache = None
        self.page_index_cache = None
        self.named_destinations_cache = None
        self.oc_layers = None
        self.acroform_cache = None
        self.fields_cache = None
        self.decoder_cache = {}
        self.inherited_values_cache = {}

    @classmethod
    def open(cls, source: PdfSource, password: str = "") -> PdfDocument:
        """Convenience method to initialize PdfDocument."""
        return cls(source, password=password)

    @property
    def data(self) -> PdfDataBuffer:
        return self.raw_data

    @property
    def trailer(self) -> PdfDictLike:
        return self.trailer_dict

    def load_data(self, source: PdfSource) -> PdfDataBuffer:
        if isinstance(source, (str, Path)):
            if isinstance(source, str) and source.startswith("%PDF"):
                return source.encode("latin-1")
            try:
                with open(source, "rb") as file_handle:
                    return mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_READ)
            except ValueError:
                raise PdfSourceError("PDF source is empty")
        if isinstance(source, bytes):
            return source
        if isinstance(source, (memoryview, bytearray)):
            return bytes(source)

        read = getattr(source, "read", None)
        if not callable(read):
            raise PdfSourceError(f"PDF source type {type(source).__name__} is not supported")
        try:
            raw = read()
        except OSError as exc:
            raise PdfSourceError(str(exc)) from exc
        return raw if isinstance(raw, bytes) else bytes(raw)

    def scan_xref(self) -> None:
        data = self.raw_data
        start = XRefScanner.find_startxref(data)
        if start is None:
            self.xref, self.trailer_dict = XRefScanner.recover_missing_xref(data)
            self.xref = XRefScanner.repair_misaligned_entries(data, self.xref)
            return

        try:
            self.xref, self.trailer_dict = XRefScanner.load_section_chain(data, start, set())
            self.xref = XRefScanner.repair_misaligned_entries(data, self.xref)
        except (PdfParseError, ValueError, struct.error, OSError) as exc:
            try:
                self.xref, self.trailer_dict = XRefScanner.recover_missing_xref(data)
                self.xref = XRefScanner.repair_misaligned_entries(data, self.xref)
            except PdfParseError:
                raise PdfParseError("invalid xref section") from exc

    def init_security(self, password: str) -> None:
        encrypt_ref = self.trailer_dict.get("Encrypt")
        if encrypt_ref is None:
            return

        encrypt_dict = self.resolver.resolve_dict(encrypt_ref)
        if not isinstance(encrypt_dict, dict):
            raise PdfUnsupportedError("Invalid Encrypt dictionary")

        filter_name = self.resolver.resolve_name(encrypt_dict.get("Filter"))
        if filter_name in {"Adobe.PubSec", "PubSec"}:
            raise PdfUnsupportedError("Public-key encryption is not supported")
        if filter_name != "Standard":
            raise PdfUnsupportedError(f"Unsupported encryption filter: {filter_name}")

        v_opt = self.resolver.resolve_int(encrypt_dict.get("V"), 0)
        v = v_opt if v_opt is not None else 0
        handler_cls = SECURITY_HANDLER_REGISTRY.get(v)
        if handler_cls is None:
            raise PdfUnsupportedError(f"Unsupported standard encryption algorithm V={v}")

        docid = self.trailer_dict.get("ID")
        if docid is None:
            docid_list: tuple[PdfObject, ...] = (b"", b"")
        elif isinstance(docid, PdfReference):
            docid = self.resolver.resolve(docid)
            if not isinstance(docid, (list, tuple)) or len(docid) == 0:
                raise PdfUnsupportedError("Invalid trailer ID array")
            docid_list = tuple(docid)
        elif isinstance(docid, (list, tuple)) and len(docid) > 0:
            docid_list = tuple(docid)
        else:
            raise PdfUnsupportedError("Invalid trailer ID array")

        handler = handler_cls(docid_list, encrypt_dict, password)
        self.decipher = handler.decrypt

    def resolve(self, ref: PdfObject) -> PdfObject:
        return self.resolver.resolve(ref)

    def catalog(self) -> PdfDict:
        if self.catalog_cache is None:
            root_ref = self.trailer_dict.get("Root")
            if root_ref is None:
                raise ValueError("missing catalog root")
            root = self.resolver.resolve_dict(root_ref)
            if not isinstance(root, dict):
                raise ValueError("invalid catalog root")
            self.catalog_cache = root
        return self.catalog_cache

    def extract_text(self, layout: bool = True) -> str:
        """Extract text from all pages and return as a single string."""
        return "\f".join(self.iter_text(layout=layout)) + "\f"

    def to_markdown(self) -> str:
        """Extract all pages as structured Markdown with metadata and outlines."""
        md_parts = []

        # 1. Global Metadata
        metadata = self.get_metadata()
        info_value = metadata.get("info")
        if isinstance(info_value, dict) and info_value:
            md_parts.append("# Document Metadata")
            for k, v in info_value.items():
                if v:
                    md_parts.append(f"- **{k}**: {v}")
            md_parts.append("")

        # 2. Table of Contents
        try:
            outlines = self.iter_outlines()
            if outlines:
                md_parts.append("# Table of Contents")
                for item in outlines:
                    indent = "  " * item.level
                    md_parts.append(f"{indent}- {item.title}")
                md_parts.append("")
        except (ValueError, KeyError, StopIteration):
            pass

        # 3. Pages
        md_parts.append("\f".join(page.to_markdown() for page in self.pages))

        return "\n".join(md_parts) + "\f"

    def iter_text(self, layout: bool = True) -> Iterator[str]:
        """Yield extracted text page-by-page."""
        for page in self.pages:
            yield page.extract_text(layout=layout)
            # Memory safety: clear heavy page state after rendering to text
            page.state = None
            page.graphics = None
            page.grid_lines = None
            page.texttrace = None
            page.tables = {}

    def get_metadata(self) -> MetadataDict:
        if self.metadata_cache is None:
            self.metadata_cache = resolve_metadata(self.resolver, self.trailer_dict)
        return self.metadata_cache

    @property
    def structure(self) -> StructureTree | None:
        if self.structure_cache is None:
            if self.structure_root_cache is None:
                resolved_root = self.resolver.resolve(self.catalog().get("StructTreeRoot"))
                if resolved_root is not None and not isinstance(resolved_root, dict):
                    raise ValueError("invalid StructTreeRoot dictionary")
                self.structure_root_cache = resolved_root
            struct_root = self.structure_root_cache
            if struct_root is None:
                self.structure_cache = None
            elif not isinstance(struct_root, dict):
                raise ValueError("invalid StructTreeRoot dictionary")
            else:
                self.structure_cache = StructureTree(self, struct_root)
        return self.structure_cache

    @property
    def mark_info(self) -> PdfDict | None:
        if self.mark_info_cache is None:
            self.mark_info_cache = self.resolver.resolve_dict(self.catalog().get("MarkInfo"))
        mark_info = self.mark_info_cache
        if mark_info is None:
            return None
        if not isinstance(mark_info, dict):
            raise ValueError("invalid MarkInfo dictionary")
        return mark_info

    # --- Page tree ---

    def iter_page_dicts(self) -> Iterator[PdfDict]:
        """Yield page dictionaries from the PDF catalog."""
        if self.page_dicts_cache is None:
            self.page_dicts_cache, self.page_index_cache = self.build_page_cache()
        yield from self.page_dicts_cache

    def build_page_cache(self) -> tuple[list[PdfDict], dict[int, int]]:
        catalog = self.catalog()
        pages_ref = catalog.get("Pages")
        if pages_ref is None:
            raise ValueError("missing page tree root")
        pages_node = self.resolver.resolve(pages_ref)
        if not isinstance(pages_node, dict):
            raise ValueError("invalid page tree root")

        page_dicts: list[PdfDict] = []
        page_index_cache: dict[int, int] = {}

        def traverse(node: PdfObject, _depth: int = 0) -> None:
            if _depth > MAX_PAGE_TREE_DEPTH:
                raise ValueError("invalid page tree depth")
            node = self.resolver.resolve(node)
            if not isinstance(node, dict):
                raise ValueError("invalid page tree node")
            node_type = self.resolver.resolve_name(node.get("Type"))
            if node_type == "Pages":
                kids = self.resolver.resolve(node.get("Kids"))
                if kids is None:
                    raise ValueError("invalid page tree Kids array")
                if not isinstance(kids, list):
                    raise ValueError("invalid page tree Kids array")
                for kid in kids:
                    traverse(kid, _depth + 1)
            elif node_type == "Page":
                page_index_cache[id(node)] = len(page_dicts)
                page_dicts.append(node)
                prewarm_page_fonts(node)

        def prewarm_page_fonts(page_node: PdfDictLike) -> None:
            resources = self.resolver.resolve(page_node.get("Resources"))
            if resources is None:
                return
            if not isinstance(resources, dict):
                raise ValueError("invalid page Resources dictionary")
            fonts = self.resolver.resolve(resources.get("Font"))
            if fonts is None:
                return
            if not isinstance(fonts, dict):
                raise ValueError("invalid page Font resources")
            from core_pdf.impl.engine.spec.s_09_fonts.decoder import FontDecoder

            for font_ref in fonts.values():
                cache_key = (
                    (font_ref.obj_num, font_ref.gen_num)
                    if isinstance(font_ref, PdfReference)
                    else id(font_ref)
                )
                if cache_key in self.decoder_cache:
                    continue
                font_obj = self.resolver.resolve(font_ref)
                if isinstance(font_obj, PdfStream):
                    font_obj = font_obj.dictionary
                if not isinstance(font_obj, dict):
                    raise ValueError("invalid font dictionary")
                resolved = self.resolver.resolve_font_dict(font_obj)
                self.decoder_cache[cache_key] = FontDecoder(resolved)

        traverse(pages_node)
        return page_dicts, page_index_cache

    @property
    def pages(self) -> list[PdfPage]:
        if self.pages_cache is None:
            if self.page_dicts_cache is None:
                self.page_dicts_cache, self.page_index_cache = self.build_page_cache()
            self.pages_cache = [
                PdfPage(self, page_dict, i + 1) for i, page_dict in enumerate(self.page_dicts_cache)
            ]
        return self.pages_cache

    def page_count(self) -> int:
        return len(self.pages)

    def page_index_for(self, page_obj: PdfObject) -> int | None:
        if isinstance(page_obj, PdfPage):
            return page_obj.page_number - 1
        if not isinstance(page_obj, dict):
            return None
        if self.page_index_cache is None:
            self.page_dicts_cache, self.page_index_cache = self.build_page_cache()
        return self.page_index_cache.get(id(page_obj))

    def __enter__(self) -> PdfDocument:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.file_handle is not None:
            with suppress(OSError):
                self.file_handle.close()
            self.file_handle = None
