# SPDX-License-Identifier: AGPL-3.0-only

"""Associated files (ISO 32000-2 14.13) and the catalog's embedded files.

The PdfDocument methods here are functions taking the document as self; PdfDocument lists
them as its methods. A malformed AF array, file specification or embedded file stream is
skipped rather than failing the document: associated files ride along with an extraction.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from core_pdf.impl.base.exceptions import PdfError
from core_pdf.impl.base.types import PdfReference
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax.types import PdfDict, PdfObject
lazy from core_pdf.impl.base.values import decode_pdf_text_string, recover_pdf_name
lazy from core_pdf.impl.output.model import AssociatedFile, AssociatedFileOwner, EmbeddedFile
lazy from core_pdf.impl.recovery.trees import iter_name_tree_items
lazy from core_pdf.impl.spec.s_07_document import file_specification
lazy from core_pdf.impl.spec.s_07_syntax_primitives import text_string
lazy from core_pdf.impl.spec.s_14_structure import associated_files as spec_associated_files

if TYPE_CHECKING:
    from core_pdf.impl.base.pages import PageSelection
    from core_pdf.impl.base.types import PageScoped
    from core_pdf.impl.document.pdf_document import PdfDocument
    from core_pdf.impl.document.pdf_page import PdfPage

# The errors reading one file specification or owner can raise; each skips only that one.
READ_ERRORS = (PdfError, RecursionError, TypeError, ValueError)
MAX_STRUCTURE_ANCESTORS = 512


def skip_malformed(_message: str) -> None:
    return None


def reference_pair(value: object) -> tuple[int, int] | None:
    if isinstance(value, PdfReference):
        return (value.object_number, value.generation_number)
    return None


def lenient_text(data: bytes | None) -> str | None:
    if data is None:
        return None
    try:
        return decode_pdf_text_string(data)
    except ValueError:
        return data.decode("latin-1")


def decode_text(self: PdfDocument[Any], data: bytes | None) -> str | None:
    """A text string decoded for the document's version, or leniently when that fails.

    A UTF-8 byte order mark wins over a version before 2.0: writers put UTF-8 names in
    PDF 1.7 (PDF/A-3) files, and read as PDFDocEncoding the mark would show as "ï»¿".
    """
    if data is None:
        return None
    if data.startswith(b"\xef\xbb\xbf"):
        return lenient_text(data)
    try:
        return text_string.decode_pdf_text_string(data, context=self.resolver.semantic_context)
    except PdfError, ValueError:
        return lenient_text(data)


def embedded_file_record(
    self: PdfDocument[Any], specification: PdfDict, *, tree_key: str | None = None
) -> EmbeddedFile:
    resolver = self.resolver
    try:
        strings = file_specification.file_specification_strings(resolver, specification)
    except READ_ERRORS:
        strings = file_specification.FileSpecificationStrings()
    try:
        relationship = spec_associated_files.associated_file_relationship(resolver, specification)
    except READ_ERRORS:
        relationship = spec_associated_files.DEFAULT_AF_RELATIONSHIP
    stream: PdfStream | None = None
    with suppress(*READ_ERRORS):
        stream = file_specification.embedded_file_stream(resolver, specification)
    parameters = file_specification.EmbeddedFileParameters()
    stream_reference = None
    if stream is not None:
        with suppress(*READ_ERRORS):
            parameters = file_specification.embedded_file_parameters(resolver, stream)
        stream_reference = embedded_stream_reference(self, specification)
    checksum = parameters.checksum
    return EmbeddedFile(
        file_name=lenient_text(strings.file_name),
        unicode_file_name=decode_text(self, strings.unicode_file_name),
        description=decode_text(self, strings.description),
        relationship=relationship,
        mime_type=parameters.subtype,
        size=parameters.size,
        creation_date=decode_text(self, parameters.creation_date),
        modification_date=decode_text(self, parameters.modification_date),
        checksum=None if checksum is None else checksum.hex(),
        embedded=stream is not None,
        stream_reference=stream_reference,
        tree_key=tree_key,
    )


def embedded_stream_reference(
    self: PdfDocument[Any], specification: PdfDict
) -> tuple[int, int] | None:
    embedded = self.resolver.as_dict(specification.get("EF"))
    if embedded is None:
        return None
    for key in file_specification.EMBEDDED_FILE_KEYS:
        value = embedded.get(key)
        if value is not None:
            return reference_pair(value)
    return None


def owner_associated_files(
    self: PdfDocument[Any],
    owner: PdfDict,
    results: list[AssociatedFile],
    owner_kind: AssociatedFileOwner,
    **association: Any,
) -> None:
    if owner.get("AF") is None:
        return
    try:
        specifications = spec_associated_files.associated_file_specifications(
            self.resolver, owner, on_malformed=skip_malformed
        )
    except READ_ERRORS:
        return
    results.extend(
        AssociatedFile(owner_kind, self.embedded_file_record(specification), **association)
        for specification in specifications
    )


def resource_entries(
    self: PdfDocument[Any], resources: PdfDict, category: str
) -> Iterable[tuple[str, PdfDict]]:
    entries = self.resolver.dict_at(resources, category)
    if entries is None:
        return
    for key, value in entries.items():
        resolved = self.resolver.resolve(value)
        dictionary = resolved.dictionary if isinstance(resolved, PdfStream) else resolved
        if isinstance(dictionary, dict):
            yield str(recover_pdf_name(key) or key), dictionary


def page_associated_files(
    self: PdfDocument[Any],
    page: PdfPage,
    results: list[AssociatedFile],
    structure_seen: set[object] | None,
) -> None:
    page_number = int(page.page_number)
    owner_associated_files(
        self, page.page_dict, results, AssociatedFileOwner.PAGE, page_number=page_number
    )
    with suppress(*READ_ERRORS):
        for index, annotation in enumerate(page.get_annotations()):
            owner_associated_files(
                self,
                annotation.dict,
                results,
                AssociatedFileOwner.ANNOTATION,
                page_number=page_number,
                annotation_index=index,
            )
    with suppress(*READ_ERRORS):
        resources = page.resources
        for category, owner_kind in (
            ("XObject", AssociatedFileOwner.XOBJECT),
            ("Properties", AssociatedFileOwner.MARKED_CONTENT),
        ):
            for name, dictionary in resource_entries(self, resources, category):
                owner_associated_files(
                    self,
                    dictionary,
                    results,
                    owner_kind,
                    page_number=page_number,
                    resource_name=name,
                )
    if structure_seen is not None:
        with suppress(*READ_ERRORS):
            structure_associated_files(self, page, results, structure_seen)


def structure_associated_files(
    self: PdfDocument[Any], page: PdfPage, results: list[AssociatedFile], seen: set[object]
) -> None:
    """The AF of the structure elements holding the page's marked content, and their ancestors.

    Each element is reported once, with the first selected page that reaches it.
    """
    resolver = self.resolver
    key = resolver.resolve_int(page.page_dict.get("StructParents"))
    tree = self.structure
    if key is None or tree is None:
        return
    parents = resolver.resolve(tree.parent_tree.get(key))
    if not isinstance(parents, list):
        return
    page_number = int(page.page_number)
    for entry in parents:
        current: PdfObject = entry
        for _depth in range(MAX_STRUCTURE_ANCESTORS):
            # Check what was seen before resolving: pages share most of their ancestors.
            reference = reference_pair(current)
            if reference is not None:
                if reference in seen:
                    break
                element = resolver.resolve(current)
            else:
                element = current
            if not isinstance(element, dict) or id(element) in seen:
                break
            seen.add(reference if reference is not None else id(element))
            if element.get("AF") is not None:
                owner_associated_files(
                    self,
                    element,
                    results,
                    AssociatedFileOwner.STRUCTURE_ELEMENT,
                    page_number=page_number,
                    structure_type=resolver.resolve_name_like_value(element.get("S")),
                    structure_id=lenient_text(
                        file_specification.string_bytes(resolver.resolve(element.get("ID")))
                    ),
                    structure_reference=reference,
                )
            current = element.get("P")


def collect_associated_files(
    self: PdfDocument[Any], pages: Sequence[PdfPage]
) -> tuple[AssociatedFile, ...]:
    """The catalog's associated files, then those of each page and the objects on it."""
    results: list[AssociatedFile] = []
    try:
        catalog = self.catalog()
    except READ_ERRORS:
        return ()
    owner_associated_files(self, catalog, results, AssociatedFileOwner.DOCUMENT)
    structure_seen: set[object] | None = None
    root = catalog.get("StructTreeRoot")
    if root is not None:
        # The walk up each element's parents stops at the root, seen from the start.
        structure_seen = {id(self.resolver.resolve(root))}
        root_reference = reference_pair(root)
        if root_reference is not None:
            structure_seen.add(root_reference)
    for page in pages:
        page_associated_files(self, page, results, structure_seen)
    return tuple(results)


def associated_files(
    self: PdfDocument[Any], *, pages: PageSelection | None = None
) -> tuple[AssociatedFile, ...]:
    """The associated files of the document and of the selected pages (all by default)."""
    with self.acquire_operation():
        selected = tuple(page for _index, page in self.iter_selected_pages(pages))
        return self.collect_associated_files(selected)


def embedded_files(self: PdfDocument[Any]) -> tuple[EmbeddedFile, ...]:
    """The files the catalog's EmbeddedFiles name tree lists, in key order."""
    with self.acquire_operation():
        try:
            names = self.resolver.dict_at(self.catalog(), "Names")
            tree = None if names is None else self.resolver.dict_at(names, "EmbeddedFiles")
        except READ_ERRORS:
            return ()
        if tree is None:
            return ()
        records: list[EmbeddedFile] = []
        with suppress(*READ_ERRORS):
            for key, value in iter_name_tree_items(
                tree, self.resolver.resolve, self.resolver.resolve_str, on_malformed=skip_malformed
            ):
                specification = self.resolver.resolve(value)
                if isinstance(specification, dict):
                    records.append(self.embedded_file_record(specification, tree_key=key))
        return tuple(records)


def file_attachments(
    self: PdfDocument[Any], *, pages: PageSelection | None = None
) -> tuple[PageScoped[EmbeddedFile], ...]:
    """The files of the selected pages' FileAttachment annotations (ISO 32000-2 12.5.6.15)."""
    pending: list[tuple[int, EmbeddedFile]] = []
    with self.acquire_operation():
        for page_index, page in self.iter_selected_pages(pages):
            with suppress(*READ_ERRORS):
                for annotation in page.get_annotations():
                    if annotation.subtype != "FileAttachment":
                        continue
                    specification = self.resolver.as_dict(annotation.dict.get("FS"))
                    if specification is not None:
                        pending.append((page_index, self.embedded_file_record(specification)))
    return self.scoped_pending(pending)


def embedded_file_data(self: PdfDocument[Any], file: EmbeddedFile) -> bytes | None:
    """The bytes of an embedded file, decrypted and through its filters; None if not embedded."""
    reference = file.stream_reference
    if reference is None:
        return None
    with self.acquire_operation():
        stream = self.resolver.resolve(PdfReference(*reference))
        if not isinstance(stream, PdfStream):
            return None
        return stream.data
