# SPDX-License-Identifier: AGPL-3.0-only
"""Write the core-document IR as a basic, standards-compliant PDF."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import cast

from core_pdf.impl.engine.structured.model import BlockKind, Document, Page, TextLine
from core_pdf.impl.engine.structured.serialization import document_to_json_dict
from core_pdf.impl.engine.writing.document import serialize_encrypted_pdf_file
from core_pdf.impl.engine.writing.encryption import StandardPdfEncryption
from core_pdf.impl.engine.writing.fonts import (
    PdfFontProvider,
    PdfFontResource,
    StandardType1FontProvider,
)
from core_pdf.impl.engine.writing.object_graph import PdfObjectGraph
from core_pdf.impl.engine.writing.objects import (
    internal_PdfByteRangePlaceholder,
    internal_PdfSignatureContentsPlaceholder,
    serialize_pdf_string,
)
from core_pdf.impl.engine.writing.signatures import (
    PdfSignaturePlan,
    apply_signature_plan,
)
from core_pdf.impl.primitives import PdfName, PdfReference, PdfString
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream

internal_PdfDictionary = dict[PdfName, object]
internal_TaggedPageLines = tuple[tuple[str, TextLine], ...]


@dataclass(slots=True)
class internal_PdfBuildContext:
    document: Document
    graph: PdfObjectGraph
    pages_reference: PdfReference
    font_resource: PdfFontResource
    page_tagged_lines: tuple[internal_TaggedPageLines, ...]
    page_references: list[PdfReference] = field(default_factory=list)
    page_objects: list[tuple[PdfReference, internal_PdfDictionary]] = field(default_factory=list)
    form_field_references: list[PdfReference] = field(default_factory=list)


def internal_create_build_context(
    document: Document,
    font_name: str,
    font_provider: PdfFontProvider | None,
) -> internal_PdfBuildContext:
    graph = PdfObjectGraph()
    pages_reference = graph.add(None)
    font = font_provider or StandardType1FontProvider(font_name)
    page_tagged_lines = tuple(tuple(internal_tagged_lines(page)) for page in document.pages)
    font_resource = font.add_to_graph(
        graph,
        (line.text for tagged_lines in page_tagged_lines for _, line in tagged_lines),
    )
    return internal_PdfBuildContext(
        document=document,
        graph=graph,
        pages_reference=pages_reference,
        font_resource=font_resource,
        page_tagged_lines=page_tagged_lines,
    )


def internal_add_page_interactives(
    context: internal_PdfBuildContext,
    page: Page,
) -> list[PdfReference]:
    graph = context.graph
    font_resource = context.font_resource
    annotation_references: list[PdfReference] = []
    for annotation in page.annotations:
        annotation_object: internal_PdfDictionary = {
            PdfName.of("Type"): PdfName.of("Annot"),
            PdfName.of("Subtype"): PdfName.of(annotation.subtype or "Text"),
            PdfName.of("Rect"): list(annotation.bbox or (0, 0, 0, 0)),
        }
        if annotation.contents:
            annotation_object[PdfName.of("Contents")] = PdfString(
                annotation.contents.encode("utf-8")
            )
        if annotation.destination is not None:
            annotation_object[PdfName.of("A")] = {
                PdfName.of("S"): PdfName.of("URI"),
                PdfName.of("URI"): PdfString(str(annotation.destination).encode("utf-8")),
            }
        annotation_references.append(graph.add(annotation_object))
    for link in page.links:
        if link.url is None:
            continue
        annotation_references.append(
            graph.add(
                {
                    PdfName.of("Type"): PdfName.of("Annot"),
                    PdfName.of("Subtype"): PdfName.of("Link"),
                    PdfName.of("Rect"): list(link.bbox or (0, 0, 0, 0)),
                    PdfName.of("A"): {
                        PdfName.of("S"): PdfName.of("URI"),
                        PdfName.of("URI"): PdfString(link.url.encode("utf-8")),
                    },
                }
            )
        )
    for form_field in page.form_fields:
        field_type = form_field.field_type
        if field_type.casefold() in {"text", "tx"}:
            field_type = "Tx"
        elif field_type.casefold() in {"button", "checkbox", "btn"}:
            field_type = "Btn"
        field_object: internal_PdfDictionary = {
            PdfName.of("Type"): PdfName.of("Annot"),
            PdfName.of("Subtype"): PdfName.of("Widget"),
            PdfName.of("FT"): PdfName.of(field_type),
            PdfName.of("T"): PdfString(form_field.name.encode("utf-8")),
            PdfName.of("Rect"): list(form_field.bbox or (0, 0, 0, 0)),
        }
        is_button = field_type.casefold() in {"btn", "button"}
        if not is_button:
            field_object[PdfName.of("DA")] = PdfString(
                f"/{font_resource.resource_name} 10 Tf 0 g".encode("ascii")
            )
        if form_field.value_text:
            field_object[PdfName.of("V")] = PdfString(form_field.value_text.encode("utf-8"))
        flags = (1 if form_field.read_only else 0) | (2 if form_field.required else 0)
        flags |= 4 if form_field.no_export else 0
        if flags:
            field_object[PdfName.of("Ff")] = flags
        if form_field.options:
            field_object[PdfName.of("Opt")] = [
                PdfString(option.encode("utf-8")) for option in form_field.options
            ]
        x0, y0, x1, y1 = form_field.bbox or (0, 0, 0, 0)
        appearance_text = (
            b"BT /"
            + font_resource.resource_name.encode("ascii")
            + b" 10 Tf 0 g 2 2 Td "
            + serialize_pdf_string(form_field.value_text.encode("utf-8"))
            + b" Tj ET"
        )
        appearance = graph.add(
            PdfStream(
                {
                    PdfName.of("Type"): PdfName.of("XObject"),
                    PdfName.of("Subtype"): PdfName.of("Form"),
                    PdfName.of("BBox"): [0, 0, max(0, x1 - x0), max(0, y1 - y0)],
                    PdfName.of("Resources"): {
                        PdfName.of("Font"): {
                            PdfName.of(font_resource.resource_name): font_resource.reference
                        }
                    },
                },
                appearance_text,
            )
        )
        if is_button:
            off_appearance = graph.add(
                PdfStream(
                    {
                        PdfName.of("Type"): PdfName.of("XObject"),
                        PdfName.of("Subtype"): PdfName.of("Form"),
                        PdfName.of("BBox"): [0, 0, max(0, x1 - x0), max(0, y1 - y0)],
                    },
                    b"",
                )
            )
            state = PdfName.of("Yes") if form_field.value_text else PdfName.of("Off")
            field_object[PdfName.of("AS")] = state
            field_object[PdfName.of("AP")] = {
                PdfName.of("N"): {
                    PdfName.of("Off"): off_appearance,
                    PdfName.of("Yes"): appearance,
                }
            }
        else:
            field_object[PdfName.of("AP")] = {PdfName.of("N"): appearance}
        field_reference = graph.add(field_object)
        annotation_references.append(field_reference)
        context.form_field_references.append(field_reference)
    for figure in page.figures:
        marker = {
            "kind": figure.kind,
            "bbox": figure.bbox,
            "metadata": dict(figure.metadata),
        }
        annotation_references.append(
            graph.add(
                {
                    PdfName.of("Type"): PdfName.of("Annot"),
                    PdfName.of("Subtype"): PdfName.of("CoreFigure"),
                    PdfName.of("Rect"): list(figure.bbox or (0, 0, 0, 0)),
                    PdfName.of("Contents"): PdfString(
                        json.dumps(marker, default=str).encode("utf-8")
                    ),
                }
            )
        )
    return annotation_references


def internal_add_pages(
    context: internal_PdfBuildContext,
    *,
    tagged_structure: bool,
) -> None:
    for page_index, (page, tagged_lines) in enumerate(
        zip(context.document.pages, context.page_tagged_lines, strict=True)
    ):
        content = content_stream_for_page(
            page,
            context.font_resource,
            tagged=tagged_structure,
            tagged_lines=tagged_lines,
        )
        content_reference = context.graph.add(PdfStream({}, content))
        page_object: internal_PdfDictionary = {
            PdfName.of("Type"): PdfName.of("Page"),
            PdfName.of("Parent"): context.pages_reference,
            PdfName.of("MediaBox"): [0, 0, page.width or 612.0, page.height or 792.0],
            PdfName.of("Resources"): {
                PdfName.of("Font"): {
                    PdfName.of(context.font_resource.resource_name): context.font_resource.reference
                },
            },
            PdfName.of("Contents"): content_reference,
        }
        if tagged_structure:
            page_object[PdfName.of("StructParents")] = page_index
        annotation_references = internal_add_page_interactives(context, page)
        if annotation_references:
            page_object[PdfName.of("Annots")] = annotation_references
        if page.rotation:
            page_object[PdfName.of("Rotate")] = page.rotation
        if page.cropbox is not None:
            page_object[PdfName.of("CropBox")] = list(page.cropbox)
        page_reference = context.graph.add(page_object)
        context.page_references.append(page_reference)
        context.page_objects.append((page_reference, page_object))


def internal_add_signature(
    context: internal_PdfBuildContext,
    signature: PdfSignaturePlan | None,
) -> PdfReference | None:
    if signature is None:
        return None
    signature_dictionary = context.graph.add(
        {
            PdfName.of("Type"): PdfName.of("Sig"),
            PdfName.of("Filter"): PdfName.of("Adobe.PPKLite"),
            PdfName.of("SubFilter"): PdfName.of("adbe.pkcs7.detached"),
            PdfName.of("ByteRange"): internal_PdfByteRangePlaceholder(),
            PdfName.of("Contents"): internal_PdfSignatureContentsPlaceholder(
                signature.contents_length
            ),
        }
    )
    signature_field = context.graph.add(
        {
            PdfName.of("Type"): PdfName.of("Annot"),
            PdfName.of("Subtype"): PdfName.of("Widget"),
            PdfName.of("FT"): PdfName.of("Sig"),
            PdfName.of("Rect"): [0, 0, 0, 0],
            PdfName.of("T"): "Signature1",
            PdfName.of("V"): signature_dictionary,
            PdfName.of("F"): 4,
        }
    )
    first_page_reference, first_page = context.page_objects[0]
    existing_annotations = first_page.get(PdfName.of("Annots"))
    if isinstance(existing_annotations, (list, tuple)):
        first_page_annotations = [*existing_annotations, signature_field]
    elif existing_annotations is None:
        first_page_annotations = [signature_field]
    else:
        first_page_annotations = [existing_annotations, signature_field]
    context.graph.replace(
        first_page_reference,
        {**first_page, PdfName.of("Annots"): first_page_annotations},
    )
    return signature_field


def internal_finish_page_tree(context: internal_PdfBuildContext) -> None:
    context.graph.replace(
        context.pages_reference,
        {
            PdfName.of("Type"): PdfName.of("Pages"),
            PdfName.of("Kids"): context.page_references,
            PdfName.of("Count"): len(context.page_references),
        },
    )


def internal_create_catalog(context: internal_PdfBuildContext) -> internal_PdfDictionary:
    catalog: internal_PdfDictionary = {
        PdfName.of("Type"): PdfName.of("Catalog"),
        PdfName.of("Pages"): context.pages_reference,
    }
    language = context.document.metadata.get("Lang")
    if language:
        catalog[PdfName.of("Lang")] = PdfString(str(language).encode("utf-8"))
    if context.document.metadata.get("Title"):
        catalog[PdfName.of("ViewerPreferences")] = {
            PdfName.of("DisplayDocTitle"): True,
        }
    return catalog


def internal_add_outlines(
    context: internal_PdfBuildContext,
    catalog: internal_PdfDictionary,
    outlines: Sequence[Sequence[object]] | None,
) -> None:
    generated_outlines = tuple(
        (block.level or 1, block.text.strip(), page_number)
        for page_number, page in enumerate(context.document.pages, 1)
        for block in page.blocks
        if block.kind is BlockKind.HEADING and block.text.strip()
    )
    outline_source = outlines if outlines is not None else generated_outlines
    if not outline_source or not context.page_references:
        return
    outline_items: list[PdfReference] = []
    outline_levels: list[int] = []
    for row in outline_source:
        if len(row) < 3 or not isinstance(row[1], str) or not isinstance(row[2], int):
            continue
        page_index = row[2] - 1
        if page_index < 0 or page_index >= len(context.page_references):
            continue
        outline_items.append(
            context.graph.add(
                {
                    PdfName.of("Title"): PdfString(row[1].encode("utf-8")),
                    PdfName.of("Dest"): [
                        context.page_references[page_index],
                        PdfName.of("Fit"),
                    ],
                }
            )
        )
        outline_levels.append(int(row[0]) if isinstance(row[0], int) else 1)
    if not outline_items:
        return
    outline_root = context.graph.add(
        {
            PdfName.of("Type"): PdfName.of("Outlines"),
            PdfName.of("Count"): len(outline_items),
            PdfName.of("First"): outline_items[0],
            PdfName.of("Last"): outline_items[-1],
        }
    )
    stack: list[tuple[int, PdfReference]] = []
    last_by_parent: dict[PdfReference, PdfReference] = {}
    top_level_items: list[PdfReference] = []
    for index, item_reference in enumerate(outline_items):
        item = cast(
            internal_PdfDictionary,
            context.graph.objects[item_reference.object_number],
        )
        level = max(1, outline_levels[index])
        while stack and stack[-1][0] >= level:
            stack.pop()
        parent = stack[-1][1] if stack else outline_root
        item[PdfName.of("Parent")] = parent
        previous = last_by_parent.get(parent)
        if previous is not None:
            item[PdfName.of("Prev")] = previous
            previous_item = cast(
                internal_PdfDictionary,
                context.graph.objects[previous.object_number],
            )
            previous_item[PdfName.of("Next")] = item_reference
            context.graph.replace(previous, previous_item)
        else:
            parent_item = cast(
                internal_PdfDictionary,
                context.graph.objects[parent.object_number],
            )
            parent_item[PdfName.of("First")] = item_reference
            context.graph.replace(parent, parent_item)
        last_by_parent[parent] = item_reference
        if parent == outline_root:
            top_level_items.append(item_reference)
        stack.append((level, item_reference))
        context.graph.replace(item_reference, item)
    root_item = cast(
        internal_PdfDictionary,
        context.graph.objects[outline_root.object_number],
    )
    root_item[PdfName.of("First")] = top_level_items[0]
    root_item[PdfName.of("Last")] = top_level_items[-1]
    context.graph.replace(outline_root, root_item)
    catalog[PdfName.of("Outlines")] = outline_root


def internal_add_acroform(
    context: internal_PdfBuildContext,
    catalog: internal_PdfDictionary,
    signature_field: PdfReference | None,
) -> None:
    if signature_field is not None:
        catalog[PdfName.of("AcroForm")] = {
            PdfName.of("SigFlags"): 3,
            PdfName.of("NeedAppearances"): False,
            PdfName.of("Fields"): [*context.form_field_references, signature_field],
        }
    elif context.form_field_references:
        catalog[PdfName.of("AcroForm")] = {
            PdfName.of("NeedAppearances"): False,
            PdfName.of("Fields"): context.form_field_references,
        }


def internal_add_tagged_structure(
    context: internal_PdfBuildContext,
    catalog: internal_PdfDictionary,
    *,
    enabled: bool,
) -> None:
    if not enabled or not context.page_references:
        return
    graph = context.graph
    structure_root = graph.add(
        {PdfName.of("Type"): PdfName.of("StructTreeRoot"), PdfName.of("K"): []}
    )
    structure_kids: list[PdfReference] = []
    parent_tree_entries: list[object] = []
    for page_number, page_reference in enumerate(context.page_references, 1):
        page_group = graph.add(
            {
                PdfName.of("Type"): PdfName.of("StructElem"),
                PdfName.of("S"): PdfName.of("Div"),
                PdfName.of("P"): structure_root,
                PdfName.of("Pg"): page_reference,
                PdfName.of("K"): [],
                PdfName.of("T"): PdfString(f"Page {page_number}".encode("utf-8")),
            }
        )
        structure_kids.append(page_group)
        page_group_kids: list[PdfReference] = []
        section_kids: dict[PdfReference, list[PdfReference]] = {}
        section_parents: dict[PdfReference, PdfReference] = {}
        section_ordinals: dict[PdfReference, int] = {}
        section_stack: list[tuple[int, PdfReference]] = []
        active_section: PdfReference | None = None
        page_entries: list[PdfReference] = []
        for mcid, (role, _) in enumerate(context.page_tagged_lines[page_number - 1]):
            if role.startswith("H"):
                level = int(role[1:])
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                section_parent = section_stack[-1][1] if section_stack else page_group
                active_section = graph.add(
                    {
                        PdfName.of("Type"): PdfName.of("StructElem"),
                        PdfName.of("S"): PdfName.of("Sect"),
                        PdfName.of("P"): section_parent,
                        PdfName.of("Pg"): page_reference,
                        PdfName.of("K"): [],
                        PdfName.of("T"): PdfString(
                            f"Section {len(section_kids) + 1}".encode("utf-8")
                        ),
                    }
                )
                section_kids[active_section] = []
                section_parents[active_section] = section_parent
                section_ordinals[active_section] = len(section_ordinals) + 1
                if section_parent == page_group:
                    page_group_kids.append(active_section)
                else:
                    section_kids[section_parent].append(active_section)
                section_stack.append((level, active_section))
            active_section = section_stack[-1][1] if section_stack else None
            element = graph.add(
                {
                    PdfName.of("Type"): PdfName.of("StructElem"),
                    PdfName.of("S"): PdfName.of(role),
                    PdfName.of("P"): active_section or page_group,
                    PdfName.of("Pg"): page_reference,
                    PdfName.of("K"): {
                        PdfName.of("Type"): PdfName.of("MCR"),
                        PdfName.of("MCID"): mcid,
                    },
                    PdfName.of("T"): PdfString(
                        f"Page {page_number} line {mcid + 1}".encode("utf-8")
                    ),
                }
            )
            if active_section:
                section_kids[active_section].append(element)
            else:
                page_group_kids.append(element)
            page_entries.append(element)
        for figure in context.document.pages[page_number - 1].figures:
            metadata = figure.metadata
            decorative = bool(metadata.get("decorative", False))
            figure_element: internal_PdfDictionary = {
                PdfName.of("Type"): PdfName.of("StructElem"),
                PdfName.of("S"): PdfName.of("Artifact" if decorative else "CoreFigure"),
                PdfName.of("P"): active_section or page_group,
                PdfName.of("Pg"): page_reference,
                PdfName.of("K"): [],
            }
            if not decorative:
                alternate = next(
                    (
                        str(metadata[key])
                        for key in ("alt", "alternate_text", "description")
                        if metadata.get(key)
                    ),
                    None,
                )
                if alternate:
                    figure_element[PdfName.of("Alt")] = PdfString(alternate.encode("utf-8"))
            figure_reference = graph.add(figure_element)
            if active_section:
                section_kids[active_section].append(figure_reference)
            else:
                page_group_kids.append(figure_reference)
        for section, children in section_kids.items():
            graph.replace(
                section,
                {
                    PdfName.of("Type"): PdfName.of("StructElem"),
                    PdfName.of("S"): PdfName.of("Sect"),
                    PdfName.of("P"): section_parents.get(section, page_group),
                    PdfName.of("Pg"): page_reference,
                    PdfName.of("K"): children,
                    PdfName.of("T"): PdfString(
                        f"Section {section_ordinals[section]}".encode("utf-8")
                    ),
                },
            )
        graph.replace(
            page_group,
            {
                PdfName.of("Type"): PdfName.of("StructElem"),
                PdfName.of("S"): PdfName.of("Div"),
                PdfName.of("P"): structure_root,
                PdfName.of("Pg"): page_reference,
                PdfName.of("K"): page_group_kids,
                PdfName.of("T"): PdfString(f"Page {page_number}".encode("utf-8")),
            },
        )
        parent_tree_entries.append(page_entries)
    language = context.document.metadata.get("Lang")
    graph.replace(
        structure_root,
        {
            PdfName.of("Type"): PdfName.of("StructTreeRoot"),
            PdfName.of("K"): structure_kids,
            PdfName.of("RoleMap"): {PdfName.of("CoreFigure"): PdfName.of("Figure")},
            **({PdfName.of("Lang"): PdfString(str(language).encode("utf-8"))} if language else {}),
            PdfName.of("ParentTree"): graph.add(
                {
                    PdfName.of("Nums"): [
                        item
                        for index, entries in enumerate(parent_tree_entries)
                        for item in (index, entries)
                    ]
                }
            ),
        },
    )
    catalog[PdfName.of("StructTreeRoot")] = structure_root


def internal_add_attachments(
    context: internal_PdfBuildContext,
    catalog: internal_PdfDictionary,
    attachments: Mapping[str, bytes] | None,
) -> None:
    if not attachments:
        return
    names: list[object] = []
    for filename, data in attachments.items():
        stream = context.graph.add(PdfStream({}, bytes(data)))
        filespec = context.graph.add(
            {
                PdfName.of("Type"): PdfName.of("Filespec"),
                PdfName.of("F"): PdfString(filename.encode("utf-8")),
                PdfName.of("EF"): {PdfName.of("F"): stream},
            }
        )
        names.extend((PdfString(filename.encode("utf-8")), filespec))
    catalog[PdfName.of("Names")] = {PdfName.of("EmbeddedFiles"): {PdfName.of("Names"): names}}


def internal_build_trailer(
    context: internal_PdfBuildContext,
    catalog: internal_PdfDictionary,
) -> dict[object, object]:
    trailer_info: PdfReference | None = None
    if context.document.metadata:
        info = {
            PdfName.of(str(key)): PdfString(str(value).encode("utf-8"))
            for key, value in context.document.metadata.items()
            if not str(key).startswith("_")
        }
        if info:
            trailer_info = context.graph.add(info)
    catalog_reference = context.graph.add(catalog)
    trailer: dict[object, object] = {PdfName.of("Root"): catalog_reference}
    if trailer_info is not None:
        trailer[PdfName.of("Info")] = trailer_info
    return trailer


def internal_canonical_file_id(document: Document) -> bytes:
    file_id_hasher = sha256()
    digest_record = document_to_json_dict(document)
    digest_record.pop("diagnostics", None)
    for page_record in cast(list[dict[str, object]], digest_record.get("pages", [])):
        page_record.pop("diagnostics", None)
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    for chunk in encoder.iterencode(digest_record):
        file_id_hasher.update(chunk.encode("utf-8"))
    return file_id_hasher.digest()[:16]


def internal_serialize_pdf(
    context: internal_PdfBuildContext,
    trailer: Mapping[object, object],
    *,
    encryption: StandardPdfEncryption | None,
    signature: PdfSignaturePlan | None,
    version: str,
) -> bytes:
    if encryption is None:
        output = context.graph.to_pdf(trailer=trailer, version=version)
        return apply_signature_plan(output, signature) if signature is not None else output
    return serialize_encrypted_pdf_file(
        context.graph.objects,
        trailer=trailer,
        encryption=encryption,
        file_id=internal_canonical_file_id(context.document),
        version=version,
    )


def serialize_document_to_pdf(
    document: Document,
    *,
    font_name: str = "Helvetica",
    font_provider: PdfFontProvider | None = None,
    encryption: StandardPdfEncryption | None = None,
    signature: PdfSignaturePlan | None = None,
    version: str = "1.7",
    attachments: Mapping[str, bytes] | None = None,
    outlines: Sequence[Sequence[object]] | None = None,
    tagged_structure: bool = True,
) -> bytes:
    """Serialize pages and their extracted text into a new PDF file."""
    if signature is not None and encryption is not None:
        raise ValueError("PDF encryption and signature containers cannot be combined")
    if signature is not None and not document.pages:
        raise ValueError("a signed PDF requires at least one page")
    context = internal_create_build_context(document, font_name, font_provider)
    internal_add_pages(context, tagged_structure=tagged_structure)
    signature_field = internal_add_signature(context, signature)
    internal_finish_page_tree(context)
    catalog = internal_create_catalog(context)
    internal_add_outlines(context, catalog, outlines)
    internal_add_acroform(context, catalog, signature_field)
    internal_add_tagged_structure(context, catalog, enabled=tagged_structure)
    internal_add_attachments(context, catalog, attachments)
    trailer = internal_build_trailer(context, catalog)
    return internal_serialize_pdf(
        context,
        trailer,
        encryption=encryption,
        signature=signature,
        version=version,
    )


def content_stream_for_page(
    page: Page,
    font: PdfFontResource,
    lines: Iterable[TextLine] | None = None,
    *,
    tagged: bool = False,
    tagged_lines: Iterable[tuple[str, TextLine]] | None = None,
) -> bytes:
    commands: list[bytes] = []
    role_lines = tuple(
        tagged_lines or (("P", line) for line in (lines or internal_page_lines(page)))
    )
    for mcid, (role, line) in enumerate(role_lines):
        tagged_line = tagged
        text = line.text.replace("\n", " ")
        encoded = font.encode_text(text)
        x, y = internal_line_position(page, line)
        font_size = internal_line_font_size(line)
        commands.extend(
            (
                (f"/{role} <</MCID {mcid}>> BDC\n".encode("ascii") if tagged_line else b""),
                b"BT\n",
                f"/{font.resource_name} {internal_number(font_size)} Tf\n".encode("ascii"),
                f"1 0 0 1 {internal_number(x)} {internal_number(y)} Tm\n".encode("ascii"),
                serialize_pdf_string(encoded) + b" Tj\nET\n",
                (b"EMC\n" if tagged_line else b""),
            )
        )
    for figure in page.figures:
        metadata = figure.metadata
        color = metadata.get("color")
        if isinstance(color, (tuple, list)) and len(color) >= 3:
            commands.append(
                (
                    "{} {} {} RG\n".format(*[internal_number(float(value)) for value in color[:3]])
                ).encode("ascii")
            )
        width = metadata.get("width")
        if isinstance(width, (int, float)):
            commands.append(f"{internal_number(float(width))} w\n".encode("ascii"))
        if figure.kind == "line":
            p1 = metadata.get("p1", (figure.bbox or (0, 0, 0, 0))[:2])
            p2 = metadata.get("p2", (figure.bbox or (0, 0, 0, 0))[2:])
            if isinstance(p1, (tuple, list)) and isinstance(p2, (tuple, list)):
                commands.append(
                    (
                        f"{internal_number(float(p1[0]))} {internal_number(float(p1[1]))} m\n"
                        f"{internal_number(float(p2[0]))} {internal_number(float(p2[1]))} l\nS\n"
                    ).encode("ascii")
                )
        elif figure.kind == "rect" and figure.bbox is not None:
            x0, y0, x1, y1 = figure.bbox
            commands.append(
                f"{internal_number(x0)} {internal_number(y0)} "
                f"{internal_number(x1 - x0)} {internal_number(y1 - y0)} re\nS\n".encode("ascii")
            )
    return b"".join(commands)


def internal_page_lines(page: Page) -> Iterable[TextLine]:
    for _, line in internal_tagged_lines(page):
        yield line


def internal_tagged_lines(page: Page) -> Iterable[tuple[str, TextLine]]:
    for block in page.blocks:
        role = (
            f"H{max(1, min(6, block.level or 1))}"
            if block.kind is BlockKind.HEADING
            else "LI"
            if block.kind is BlockKind.LIST
            else "Figure"
            if block.kind is BlockKind.FIGURE
            else "P"
        )
        yield from ((role, line) for line in block.lines)
    for table in page.tables:
        for row_index, row in enumerate(table.rows):
            role = "TH" if row_index == 0 else "TD"
            yield role, TextLine(" | ".join(cell.text for cell in row))


def internal_line_position(page: Page, line: TextLine) -> tuple[float, float]:
    if line.bbox is not None:
        return line.bbox[0], line.bbox[1]
    return 36.0, max(36.0, (page.height or 792.0) - 36.0)


def internal_line_font_size(line: TextLine) -> float:
    if line.bbox is None:
        return 12.0
    return max(1.0, line.bbox[3] - line.bbox[1])


def internal_number(value: float) -> str:
    return format(value, ".4g")


__all__ = ("content_stream_for_page", "serialize_document_to_pdf")
