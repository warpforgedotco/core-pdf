from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, cast

import pytest

from core_pdf import PdfDocument
from core_pdf.api.compat.llamaindex import (
    MetadataMode,
    TextNode,
    get_nodes_from_documents,
    load_data,
)
from core_pdf.api.compat.pdfminer import extract_text as extract_pdfminer_text
from core_pdf.api.compat.pdfplumber import open as open_pdfplumber
from core_pdf.api.compat.pikepdf import Pdf
from core_pdf.api.compat.pymupdf import Annot, Matrix
from core_pdf.api.compat.pymupdf import open as fitz_open
from core_pdf.api.compat.pypdf import Destination, PdfMerger, PdfReader, PdfWriter, Rectangle
from core_pdf.api.compat.unstructured import ElementMetadata, PageBreak, partition_pdf
from core_pdf.api.compat.xray import inspect as inspect_xray
from core_pdf.impl.exceptions import PdfUnsupportedError

FIXTURE = Path("tests/fixtures/pdfminer.six/samples/simple1.pdf")
ANNOTATION_FIXTURE = Path("tests/fixtures/pdfminer.six/samples/contrib/issue-1082-annotations.pdf")
FORM_FIXTURE = Path("tests/fixtures/pdfminer.six/samples/acroform/AcroForm_TEST.pdf")


def test_pypdf_facade_reads_and_writes_high_level_pages() -> None:
    reader = PdfReader(FIXTURE, strict=False)
    assert reader.pages[0].extract_text(extraction_mode="layout")
    output = BytesIO()
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    writer.write(output)
    assert len(PdfReader(output.getvalue()).pages) == 1
    reader.close()


def test_pypdf_writer_pages_are_page_objects() -> None:
    reader = PdfReader(FIXTURE)
    writer = PdfWriter()
    writer.add_page(reader.pages[0])

    assert writer.pages[0].extract_text()
    assert writer.pages[0].mediabox.width > 0
    reader.close()


def test_pypdf_reader_exposes_outline_destinations() -> None:
    reader = PdfReader(FIXTURE)
    assert reader.outline == []
    destination = Destination("first", 0)
    assert reader.get_destination_page_number(destination) == 0
    reader.close()


def test_pypdf_reader_resolves_page_numbers() -> None:
    reader = PdfReader(FIXTURE)
    assert reader.get_page_number(reader.pages[0]) == 0
    assert reader.get_page_number(reader.get_page(0)) == 0
    reader.close()


def test_pypdf_writer_creates_outline_items(tmp_path: Path) -> None:
    output = tmp_path / "outlined.pdf"
    writer = PdfWriter()
    writer.add_blank_page(200, 200)
    destination = writer.add_outline_item("First page", 0)
    writer.add_bookmark("Nested", 0, parent=destination)
    writer.write(output)

    with PdfReader(output) as reader:
        assert [(item.title, item.page) for item in reader.outline] == [
            ("First page", 0),
            ("Nested", 0),
        ]


def test_pypdf_page_transforms_persist_in_written_output() -> None:
    reader = PdfReader(FIXTURE)
    page = reader.pages[0].rotate(90).scale_to(300, 400).set_cropbox((10, 20, 250, 350))
    writer = PdfWriter()
    writer.add_page(page)
    output = BytesIO()
    writer.write(output)
    written = PdfReader(output.getvalue())
    assert written.pages[0].rotation == 90
    assert written.pages[0].mediabox == (0, 0, 300, 400)
    assert written.pages[0].cropbox == (10, 20, 250, 350)
    reader.close()


def test_pypdf_page_boxes_expose_geometry_and_transfer_rotation() -> None:
    page = PdfWriter().add_blank_page(width=200, height=100)

    assert isinstance(page.mediabox, Rectangle)
    assert page.mediabox.width == 200
    assert page.mediabox.height == 100
    assert page.mediabox.upper_right == (200.0, 100.0)

    page.rotate(90).transfer_rotation_to_content()
    assert page.rotation == 0
    assert page.mediabox.width == 100
    assert page.mediabox.height == 200


def test_pypdf_page_merge_combines_structured_content() -> None:
    reader = PdfReader(FIXTURE)
    page = reader.pages[0]
    original = page.extract_text()
    page.merge_page(reader.pages[0])
    assert page.extract_text().count(original) == 2
    reader.close()


def test_pypdf_writer_persists_metadata_and_attachments(tmp_path: Path) -> None:
    output = tmp_path / "attached.pdf"
    reader = PdfReader(FIXTURE)
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    writer.add_metadata({"Title": "Attached"})
    writer.add_attachment("note.txt", b"local attachment")
    writer.write(output)
    with PdfReader(output) as reopened:
        assert reopened.metadata["Title"] == "Attached"
        files = reopened._document.source_pdf.embedded_files()
        assert files
        assert files[0].filename == "note.txt"
        assert files[0].data == b"local attachment"
    reader.close()


def test_pypdf_writer_add_uri_persists_link(tmp_path: Path) -> None:
    output = tmp_path / "uri-link.pdf"
    writer = PdfWriter()
    writer.add_blank_page(200, 200)
    writer.add_uri(0, "https://example.test", (10, 20, 80, 100))
    writer.write(output)
    with PdfReader(output) as reader:
        assert reader.pages[0].links[0].url == "https://example.test"


def test_pypdf_writer_add_annotation_persists_note(tmp_path: Path) -> None:
    output = tmp_path / "note.pdf"
    writer = PdfWriter()
    writer.add_blank_page(200, 200)
    writer.add_annotation(
        0,
        {"/Subtype": "/Text", "/Rect": (10, 20, 30, 40), "/Contents": "note"},
    )
    writer.write(output)
    with PdfReader(output) as reader:
        assert reader.pages[0].annotations[0].contents == "note"


def test_pypdf_merger_reopens_merged_output() -> None:
    output = BytesIO()
    merger = PdfMerger()
    merger.append(FIXTURE)
    merger.append(FIXTURE)
    merger.write(output)
    assert len(PdfReader(output.getvalue()).pages) == 2
    merger.close()


def test_pypdf_merger_preserves_outlines(tmp_path: Path) -> None:
    source = tmp_path / "source-outlines.pdf"
    source_writer = PdfWriter()
    source_writer.add_blank_page(100, 100)
    source_writer.add_outline_item("Source page", 0)
    source_writer.write(source)

    merger = PdfMerger()
    merger.append(source)
    output = BytesIO()
    merger.write(output)

    with PdfReader(output.getvalue()) as reader:
        assert [(item.title, item.page) for item in reader.outline] == [("Source page", 0)]
    merger.close()


def test_pypdf_merger_offsets_outlines_when_inserting(tmp_path: Path) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    for path, title in ((first, "First"), (second, "Second")):
        writer = PdfWriter()
        writer.add_blank_page(100, 100)
        writer.add_outline_item(title, 0)
        writer.write(path)

    merger = PdfMerger()
    merger.append(first)
    merger.merge(0, second)
    output = BytesIO()
    merger.write(output)

    with PdfReader(output.getvalue()) as reader:
        assert [(item.title, item.page) for item in reader.outline] == [
            ("First", 1),
            ("Second", 0),
        ]
    merger.close()


def test_pypdf_merger_preserves_metadata_and_attachments(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    writer = PdfWriter()
    writer.add_blank_page(100, 100)
    writer.add_metadata({"Title": "Merged"})
    writer.add_attachment("merged.txt", b"merged data")
    writer.write(source)
    merger = PdfMerger()
    merger.append(source)
    output = BytesIO()
    merger.write(output)
    reopened = PdfReader(output.getvalue())
    assert reopened.metadata["Title"] == "Merged"
    assert reopened._document.source_pdf.embedded_files()[0].data == b"merged data"
    merger.close()


def test_pypdf_merger_convenience_metadata_and_attachment_methods() -> None:
    merger = PdfMerger()
    merger.append(FIXTURE)
    merger.add_metadata({"Title": "Convenience merged"})
    merger.add_attachment("extra.txt", b"extra data")
    output = BytesIO()
    merger.write(output)

    with PdfReader(output.getvalue()) as reader:
        assert reader.metadata["Title"] == "Convenience merged"
        attachments = reader._document.source_pdf.embedded_files()
        assert [
            (item.filename, item.data) for item in attachments if item.filename == "extra.txt"
        ] == [("extra.txt", b"extra data")]
    merger.close()


def test_pypdf_writer_supports_blank_pages_and_reader_clone() -> None:
    reader = PdfReader(FIXTURE)
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    writer.add_blank_page(100, 200)
    output = BytesIO()
    writer.write(output)
    reopened = PdfReader(output.getvalue())
    assert len(reopened.pages) == len(reader.pages) + 1
    assert reopened.pages[-1].mediabox == (0, 0, 100, 200)
    reader.close()


def test_pypdf_writer_encrypts_output_and_exposes_reader_helpers() -> None:
    writer = PdfWriter()
    writer.add_blank_page(100, 100)
    writer.encrypt("secret")
    output = BytesIO()
    writer.write(output)
    assert b"/Encrypt" in output.getvalue()
    reader = PdfReader(output.getvalue())
    assert reader.is_encrypted
    assert not reader.decrypt("wrong")
    assert reader.decrypt("secret")
    assert reader.num_pages == len(reader.pages)
    assert reader.get_page(0).mediabox == (0, 0, 100, 100)
    reader.close()


def test_pikepdf_open_forwards_password() -> None:
    writer = PdfWriter()
    writer.add_blank_page(100, 100)
    writer.encrypt("secret")
    output = BytesIO()
    writer.write(output)
    with pytest.raises(PdfUnsupportedError, match="password|Password"):
        Pdf.open(output.getvalue())
    with Pdf.open(output.getvalue(), password="secret") as pdf:
        assert len(pdf.pages) == 1


def test_pypdf_writer_preserves_annotations_and_links() -> None:
    reader = PdfReader(ANNOTATION_FIXTURE)
    original_annotations = reader.pages[0].annotations
    assert original_annotations
    assert reader.pages[0].links == reader._document.pages[0].links
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    output = BytesIO()
    writer.write(output)
    reopened = PdfReader(output.getvalue())
    assert reopened.pages[0].annotations
    assert reopened.pages[0].annotations[0].subtype == original_annotations[0].subtype
    reader.close()


def test_pypdf_writer_preserves_form_fields_and_values() -> None:
    reader = PdfReader(FORM_FIXTURE)
    assert "Combo Box0" in reader.get_fields()
    assert reader.pages[0].form_fields
    field = next(field for field in reader._document.form_fields if field.name == "Combo Box0")
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    writer.update_page_form_field_values(writer.pages[0], {field.name: "compat value"})
    output = BytesIO()
    writer.write(output)
    reopened = PdfReader(output.getvalue())
    values = {item.name: item.value_text for item in reopened._document.form_fields}
    assert values["Combo Box0"] == "compat value"
    reader.close()


def test_pypdf_reader_form_convenience_helpers() -> None:
    reader = PdfReader(FORM_FIXTURE)
    assert isinstance(reader.get_form_text_fields(), dict)
    assert reader.get_form_xfa() is None
    reader.close()


def test_pymupdf_and_pikepdf_facades_expose_text() -> None:
    document = fitz_open(FIXTURE)
    assert document.load_page(0).get_text(flags=0)
    assert document.page_count == len(document.pages)
    assert "<div>" in str(document.load_page(0).get_text("html"))
    assert "xmlns" in str(document.load_page(0).get_text("xhtml"))
    assert '"blocks"' in str(document.load_page(0).get_text("json"))
    assert '"blocks"' in str(document.load_page(0).get_text("rawjson"))
    assert document.load_page(0).search_for("Hello")
    assert document.load_page(0).search_for("Hello")[0][0] > 0
    words = cast(list[tuple[object, ...]], document.load_page(0).get_text("words"))
    assert words
    assert len(words[0]) == 8
    assert words[0][4] == "Hello"
    blocks = cast(list[tuple[object, ...]], document.load_page(0).get_text("blocks"))
    assert blocks
    assert len(blocks[0]) == 7
    block_dict = cast(dict[str, object], document.load_page(0).get_text("dict"))
    assert block_dict["blocks"]
    document.save(BytesIO(), garbage=4, deflate=True)
    document.close()


def test_document_facades_conform_to_the_canonical_page_view() -> None:
    with PdfDocument.open(FIXTURE) as source:
        canonical = source.structured_document.pages[0]

    with PdfReader(FIXTURE) as reader:
        pypdf_page = reader.pages[0]
        assert pypdf_page.extract_text().strip() == canonical.text.strip()
        assert pypdf_page.mediabox.width == canonical.width
        assert pypdf_page.mediabox.height == canonical.height

    with fitz_open(FIXTURE) as document:
        pymupdf_page = document.load_page(0)
        assert cast(str, pymupdf_page.get_text()).strip() == canonical.text.strip()
        assert pymupdf_page.mediabox.width == canonical.width
        assert pymupdf_page.mediabox.height == canonical.height

    with Pdf.open(FIXTURE) as document:
        pikepdf_page = document.pages[0]
        assert pikepdf_page.extract_text().strip() == canonical.text.strip()
        assert pikepdf_page.mediabox.width == canonical.width
        assert pikepdf_page.mediabox.height == canonical.height


def test_extraction_facades_preserve_canonical_page_content() -> None:
    with PdfDocument.open(FIXTURE) as source:
        canonical = source.structured_document.pages[0]
        expected = "Hello World"
        assert expected in canonical.text

    assert expected in " ".join(extract_pdfminer_text(FIXTURE).split())

    with open_pdfplumber(FIXTURE) as pdf:
        assert expected in " ".join(pdf.pages[0].extract_text().split())

    elements = partition_pdf(FIXTURE)
    assert elements
    assert expected in " ".join(element.text for element in elements)

    documents = load_data(FIXTURE)
    assert documents
    assert expected in " ".join(document.text for document in documents)

    findings = inspect_xray(FIXTURE)
    assert all(isinstance(page_number, int) for page_number in findings)
    assert all(
        isinstance(finding, dict)
        for findings_on_page in findings.values()
        for finding in findings_on_page
    )


def test_pymupdf_new_page_supports_local_editing_and_save(tmp_path: Path) -> None:
    document = fitz_open(FIXTURE)
    page = document.new_page(pno=0, width=200, height=150)
    page.insert_text((20, 30), "new page")
    output = tmp_path / "new-page.pdf"
    document.save(output)

    with fitz_open(output) as reopened:
        assert reopened.page_count == 2
        assert cast(str, reopened[0].get_text()).strip() == "new page"
        assert reopened[0].mediabox == (0, 0, 200, 150)
    document.close()


def test_pymupdf_pixmap_supports_matrix_dpi_and_png(tmp_path: Path) -> None:
    with fitz_open(FIXTURE) as document:
        pixmap = cast(Any, document).load_page(0).get_pixmap(matrix=Matrix(0.5, 0.5), alpha=False)
        assert pixmap.width > 0
        assert pixmap.height > 0
        assert pixmap.n == 3
        output = tmp_path / "page.png"
        pixmap.save(output)
        assert output.read_bytes().startswith(b"\x89PNG")


def test_pymupdf_textpage_reuses_high_level_extraction() -> None:
    with fitz_open(FIXTURE) as document:
        textpage = cast(Any, document).load_page(0).get_textpage()
        assert "Hello" in textpage.extractText()
        assert textpage.extractWORDS()
        assert textpage.extractBLOCKS()
        assert textpage.extractDICT()["blocks"]
        assert "<div>" in textpage.extractHTML()


def test_pymupdf_rawdict_exposes_character_records() -> None:
    with fitz_open(FIXTURE) as document:
        raw = cast(Any, document).load_page(0).get_text("rawdict")
        span = raw["blocks"][0]["lines"][0]["spans"][0]
        assert span["chars"]
        assert span["chars"][0]["c"]


def test_pymupdf_document_page_helpers_delegate_locally() -> None:
    with fitz_open(FIXTURE) as document:
        facade = cast(Any, document)
        assert isinstance(facade.get_page_images(0), list)
        assert facade.get_page_pixmap(0, matrix=Matrix(0.25, 0.25)).width > 0
        assert isinstance(facade.load_page(0).get_image_info(), list)


def test_pymupdf_textbox_and_text_length_helpers() -> None:
    with fitz_open(FIXTURE) as document:
        page = cast(Any, document).load_page(0)
        assert page.get_textbox((0, 0, page.mediabox.width, page.mediabox.height))
        assert page.get_text_length("Hello", fontsize=10) > 0


def test_pymupdf_insert_text_persists_through_save(tmp_path: Path) -> None:
    output = tmp_path / "inserted.pdf"
    with fitz_open(FIXTURE) as document:
        page = cast(Any, document).load_page(0)
        assert page.insert_text((40, 40), "Inserted locally") > 0
        cast(Any, document).save(output)
    with fitz_open(output) as reopened:
        assert "Inserted locally" in cast(Any, reopened).load_page(0).get_text()


def test_pymupdf_draw_primitives_persist_through_save(tmp_path: Path) -> None:
    output = tmp_path / "drawn.pdf"
    with fitz_open(FIXTURE) as document:
        page = cast(Any, document).load_page(0)
        page.draw_rect((20, 20, 80, 80), color=(1, 0, 0), width=2)
        page.draw_line((20, 20), (80, 80), color=(0, 0, 1))
        assert {item["type"] for item in page.get_drawings()} >= {"rect", "line"}
        cast(Any, document).save(output)
    with fitz_open(output) as reopened:
        drawings = cast(Any, reopened).load_page(0).get_drawings()
        assert {item["type"] for item in drawings} >= {"rect", "line"}


def test_pymupdf_annotation_creation_persists(tmp_path: Path) -> None:
    output = tmp_path / "created-annots.pdf"
    with fitz_open(FIXTURE) as document:
        page = cast(Any, document).load_page(0)
        page.add_text_annot((20, 20), "created note")
        page.add_highlight_annot((30, 30, 60, 45))
        page.add_underline_annot((30, 50, 60, 65))
        page.add_rect_annot((70, 70, 100, 100))
        cast(Any, document).save(output)
    with fitz_open(output) as reopened:
        subtypes = {annot.type[0] for annot in cast(Any, reopened).load_page(0).annots()}
        assert subtypes >= {"Text", "Highlight", "Underline", "Square"}


def test_pymupdf_widgets_update_persistently(tmp_path: Path) -> None:
    output = tmp_path / "widget.pdf"
    with fitz_open(FORM_FIXTURE) as document:
        page = cast(Any, document).load_page(0)
        widgets = page.widgets()
        assert widgets
        widgets[0].field_value = "widget value"
        assert widgets[0].update()
        cast(Any, document).save(output)
    with fitz_open(output) as reopened:
        assert cast(Any, reopened).load_page(0).widgets()[0].field_value == "widget value"


def test_pymupdf_embedded_file_helpers_persist(tmp_path: Path) -> None:
    output = tmp_path / "embedded.pdf"
    with fitz_open(FIXTURE) as document:
        facade = cast(Any, document)
        facade.embfile_add("note.txt", b"local payload")
        assert facade.embfile_get("note.txt") == b"local payload"
        facade.save(output)
    with fitz_open(output) as reopened:
        facade = cast(Any, reopened)
        assert "note.txt" in facade.embfile_names()
        assert facade.embfile_get("note.txt") == b"local payload"


def test_pymupdf_toc_round_trip_uses_high_level_rows(tmp_path: Path) -> None:
    output = tmp_path / "toc.pdf"
    with fitz_open(FIXTURE) as document:
        facade = cast(Any, document)
        facade.set_toc([[1, "Introduction", 1], [2, "Details", 1]])
        assert facade.get_toc() == [[1, "Introduction", 1], [2, "Details", 1]]
        assert facade.get_toc(simple=False)[0][3]["kind"] == "goto"
        facade.save(output)
    with fitz_open(output) as reopened:
        assert cast(Any, reopened).get_toc() == [[1, "Introduction", 1], [2, "Details", 1]]
    document.close()
    pdf = Pdf.open(FIXTURE)
    assert pdf.pages[0].extract_text()
    pdf.close()


def test_pymupdf_redaction_mutation_saves_sanitized_output(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    writer = PdfWriter()
    writer.add_blank_page(200, 200)
    writer.write(source)
    document = fitz_open(source)
    document.load_page(0).redact((0, 0, 100, 100))
    document.apply_redactions()
    output = tmp_path / "redacted.pdf"
    document.save(output)
    assert len(PdfReader(output).pages) == 1
    document.close()


def test_pymupdf_document_lifecycle_operations() -> None:
    document = fitz_open(FIXTURE)
    assert len(document) == document.page_count
    document.select([0])
    assert document.page_count == 1
    document.insert_pdf(fitz_open(FIXTURE), from_page=0, to_page=0)
    assert document.page_count == 2
    document.delete_page(1)
    assert document.page_count == 1
    document.set_metadata({"Title": "Updated"})
    assert document.metadata["Title"] == "Updated"
    document.close()


def test_pymupdf_annotation_info_updates_persist(tmp_path: Path) -> None:
    document = fitz_open(ANNOTATION_FIXTURE)
    annotation = cast(Annot, document.load_page(0).annots()[0])
    annotation.set_info({"content": "updated locally"})
    annotation.set_rect((10, 20, 110, 120))
    output = tmp_path / "updated-annotations.pdf"
    document.save(output)
    reopened = fitz_open(output)
    reopened_annotation = cast(Annot, reopened.load_page(0).annots()[0])
    assert reopened_annotation.info["content"] == "updated locally"
    assert reopened_annotation.rect == (10, 20, 110, 120)
    document.close()
    reopened.close()


def test_pymupdf_link_mutations_persist(tmp_path: Path) -> None:
    document = fitz_open(FIXTURE)
    page = document.load_page(0)
    link = {"from": (10.0, 20.0, 110.0, 40.0), "uri": "https://example.test/one"}
    page.insert_link(link)
    updated = {"from": link["from"], "uri": "https://example.test/two"}
    document.load_page(0).update_link(updated)
    output = tmp_path / "updated-links.pdf"
    document.save(output)
    reopened = fitz_open(output)
    assert reopened.load_page(0).get_links()[-1]["uri"] == "https://example.test/two"
    reopened.load_page(0).delete_link(updated)
    assert not any(
        item["uri"] == "https://example.test/two" for item in reopened.load_page(0).get_links()
    )
    document.close()
    reopened.close()


def test_pikepdf_save_preserves_metadata_and_attachments(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    writer = PdfWriter()
    writer.add_blank_page(100, 100)
    writer.add_metadata({"Title": "Preserve"})
    writer.add_attachment("data.bin", b"payload")
    writer.write(source)
    output = tmp_path / "saved.pdf"
    pdf = Pdf.open(source)
    pdf.attachments["new.txt"] = b"new payload"
    del pdf.attachments["data.bin"]
    pdf.save(output)
    reopened = PdfReader(output)
    assert reopened.metadata["Title"] == "Preserve"
    files = reopened._document.source_pdf.embedded_files()
    assert [(item.filename, item.data) for item in files] == [("new.txt", b"new payload")]
    pdf.close()


def test_pikepdf_new_and_context_lifecycle() -> None:
    with Pdf.new() as pdf:
        assert len(pdf.pages) == 0
        assert pdf.attachments == {}


def test_pikepdf_docinfo_and_filename_are_high_level_accessors() -> None:
    pdf = Pdf.open(FIXTURE)
    assert pdf.filename is not None
    assert pdf.is_linearized is False
    pdf.docinfo["/Title"] = "Facade"
    assert pdf.metadata["Title"] == "Facade"
    pdf.close()


def test_pikepdf_pages_can_be_mutated_and_saved(tmp_path: Path) -> None:
    pdf = Pdf.new()
    blank = PdfWriter()
    page = blank.add_blank_page(120, 140)
    pdf.pages.append(page)
    pdf.pages.insert(0, blank.add_blank_page(80, 90))
    del pdf.pages[1]
    output = tmp_path / "pages.pdf"
    pdf.save(output)
    reopened = Pdf.open(output)
    assert len(reopened.pages) == 1
    assert reopened.pages[0].mediabox == (0, 0, 80, 90)
    pdf.close()
    reopened.close()


def test_pikepdf_save_preserves_outlines(tmp_path: Path) -> None:
    source = tmp_path / "outlined-source.pdf"
    writer = PdfWriter()
    writer.add_blank_page(100, 100)
    writer.add_outline_item("Bookmark", 0)
    writer.write(source)

    pdf = Pdf.open(source)
    output = tmp_path / "outlined-copy.pdf"
    pdf.save(output)
    with PdfReader(output) as reader:
        assert [(item.title, item.page) for item in reader.outline] == [("Bookmark", 0)]
    pdf.close()


def test_unstructured_and_llamaindex_adapters_preserve_provenance() -> None:
    elements = partition_pdf(FIXTURE)
    documents = load_data(FIXTURE)
    assert elements
    assert all("page_number" in element.metadata for element in elements)
    assert elements[0].id == elements[0].element_id
    assert elements[0].page_number == elements[0].metadata["page_number"]
    assert documents
    assert documents[0].metadata["page_numbers"]
    nodes = get_nodes_from_documents(FIXTURE)
    assert nodes[0].node_id == "node-0"
    assert nodes[0].source_node == "page-1"
    assert nodes[0].ref_doc_id == "page-1"
    assert nodes[0].get_content() == nodes[0].text
    assert "page_numbers" in nodes[0].get_metadata_str()
    assert nodes[0].to_dict()["node_id"] == "node-0"
    assert any(isinstance(node, type(nodes[0])) for node in get_nodes_from_documents(documents))


def test_ingestion_facades_accept_metadata_controls() -> None:
    from core_pdf.api.compat.llamaindex import load_data as llama_load_data

    documents = llama_load_data(FIXTURE, extra_info={"source": "fixture"})
    assert documents[0].metadata["source"] == "fixture"
    assert partition_pdf(FIXTURE, include_metadata=False)[0].metadata == {}


def test_unstructured_partition_supports_page_breaks() -> None:
    elements = partition_pdf(FIXTURE, include_page_breaks=True)
    assert not any(isinstance(element, PageBreak) for element in elements)


def test_unstructured_metadata_supports_mapping_and_attributes() -> None:
    metadata = ElementMetadata(
        page_number=2,
        coordinates=(1.0, 2.0, 3.0, 4.0),
        text_as_html="<p>text</p>",
    )
    element = partition_pdf(FIXTURE)[0]
    element.metadata.update(metadata)
    assert element.metadata.page_number == 2
    assert element.coordinates == (1.0, 2.0, 3.0, 4.0)
    assert element.text_as_html == "<p>text</p>"
    assert element.metadata.to_dict()["page_number"] == 2


def test_llamaindex_schema_supports_metadata_modes_and_text_nodes() -> None:
    from core_pdf.api.compat.llamaindex import Document

    document = Document(
        "body",
        {"source": "fixture", "secret": "hidden"},
        id_="doc-1",
        excluded_llm_metadata_keys=frozenset({"secret"}),
    )
    assert document.doc_id == "doc-1"
    assert document.get_content(MetadataMode.NONE) == "body"
    assert "source: fixture" in document.get_content(MetadataMode.LLM)
    assert "secret" not in document.get_content(MetadataMode.LLM)
    node = TextNode("node", id_="node-1", metadata={"page": 1})
    assert node.to_dict()["node_id"] == "node-1"
