from __future__ import annotations

from pathlib import Path

import pytest
from core_pdf.api.compat.state import Chunk, Element, StructuredState, chunk_elements

from core_pdf import PdfDocument
from core_pdf.impl.engine.structured import Annotation, Block, BlockKind, Document, Page, TextLine

LocalDocument = StructuredState


def test_chunk_elements_preserves_reading_order_and_provenance() -> None:
    items = (
        Element("p1-e0", "paragraph", "alpha", 1),
        Element("p1-e1", "table", "beta", 1),
        Element("p2-e0", "paragraph", "gamma", 2),
    )
    chunks = chunk_elements(items, max_characters=10)
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert [chunk.text for chunk in chunks] == ["alpha", "beta", "gamma"]
    assert chunks[0].element_ids == ("p1-e0",)
    assert chunks[-1].page_numbers == (2,)


def test_chunk_elements_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError, match="positive"):
        chunk_elements((), max_characters=0)


def test_split_and_merge_serialize_structured_pages() -> None:
    first = LocalDocument.__new__(LocalDocument)
    first.structured = Document(pages=(Page(page_number=1, width=72, height=72),))
    second = LocalDocument.__new__(LocalDocument)
    second.structured = Document(pages=(Page(page_number=1, width=72, height=72),))
    assert first.split(((1, 1),))[0].startswith(b"%PDF-")
    assert LocalDocument.merge((first, second)).startswith(b"%PDF-")


def test_local_document_exposes_form_and_annotation_views() -> None:
    document = LocalDocument.__new__(LocalDocument)
    document.structured = Document(pages=(Page(page_number=1),))
    assert document.forms == ()
    assert document.annotations == ()
    assert document.comments == ()
    assert document.highlights == ()
    assert document.redactions == ()
    assert document.links == ()
    assert document.form_fields == ()


def test_local_document_editing_operations_return_new_documents() -> None:
    document = LocalDocument.__new__(LocalDocument)
    document.pdf = None
    document.structured = Document(pages=(Page(page_number=1, width=72, height=72),))
    stamped = document.stamp("DRAFT", bbox=(1, 1, 40, 20))
    inserted = stamped.insert_page(Page(page_number=1, width=72, height=72), position=1)
    updated = inserted.update_metadata({"title": "Edited"})
    deleted = updated.delete_page(1)
    assert stamped.pages[0].blocks[-1].text == "DRAFT"
    assert len(inserted.pages) == 2
    assert updated.metadata["title"] == "Edited"
    assert len(deleted.pages) == 1


def test_fill_form_rejects_unknown_fields() -> None:
    document = LocalDocument.__new__(LocalDocument)
    document.pdf = None
    document.structured = Document(pages=(Page(page_number=1),))
    with pytest.raises(KeyError):
        document.fill_form("missing", "value")


def test_engine_edit_is_reserved_for_source_backed_state() -> None:
    document = LocalDocument.__new__(LocalDocument)
    document.pdf = None
    document.structured = Document(pages=(Page(page_number=1),))

    with pytest.raises(ValueError, match="engine editor"):
        document.engine_edit()


def test_capability_page_is_backed_by_the_canonical_engine() -> None:
    fixture = Path("tests/fixtures/pdfminer.six/samples/simple1.pdf")
    with PdfDocument.open(fixture) as source:
        document = StructuredState(source, source.structured_document)
        page = document.capability_page(1)
        assert page.info.number == 1
        assert page.info.width == 612


def test_normalized_elements_include_non_text_content() -> None:
    document = LocalDocument.__new__(LocalDocument)
    document.pdf = None
    document.structured = Document(pages=(Page(page_number=1),))
    assert document.elements == ()
    assert document.chunks() == ()
    assert document.to_elements() == ()
    assert document.to_chunks() == ()


def test_normalized_records_serialize_stably() -> None:
    item = Element("p1-e0", "paragraph", "hello", 1)
    assert item.to_dict()["element_id"] == "p1-e0"
    assert chunk_elements((item,))[0].to_dict()["text"] == "hello"


def test_local_document_reads_acroform_widgets_from_real_fixture() -> None:
    source = Path("tests/fixtures/pdfminer.six/samples/acroform/AcroForm_TEST.pdf")
    with LocalDocument.open(source) as document:
        assert len(document.form_fields) == 7
        assert {field.field_type for field in document.form_fields} >= {"Btn", "Ch", "Tx"}
        assert all(field.bbox is not None for field in document.form_fields)


def test_local_document_reads_annotation_records_from_real_fixture() -> None:
    source = Path("tests/fixtures/pdfminer.six/samples/acroform/AcroForm_TEST.pdf")
    with LocalDocument.open(source) as document:
        assert document.annotations
        assert all(annotation.bbox is not None for annotation in document.annotations)


def test_save_form_value_persists_through_incremental_update(tmp_path: Path) -> None:
    source = Path("tests/fixtures/pdfminer.six/samples/acroform/AcroForm_TEST.pdf")
    target = tmp_path / "filled.pdf"
    with LocalDocument.open(source) as document:
        updated = document.save_form_value("Combo Box0", "persisted", target)
    assert updated.startswith(source.read_bytes())
    with LocalDocument.open(target) as document:
        field = next(field for field in document.form_fields if field.name == "Combo Box0")
        assert field.value_text == "persisted"


def test_save_annotation_contents_persists_through_incremental_update(tmp_path: Path) -> None:
    source = Path("tests/fixtures/pdfminer.six/samples/simple5.pdf")
    target = tmp_path / "annotated.pdf"
    with LocalDocument.open(source) as document:
        assert document.highlights
        updated = document.save_annotation(0, target, contents="updated highlight")
    assert updated.startswith(source.read_bytes())
    with LocalDocument.open(target) as document:
        assert document.annotations[0].contents == "updated highlight"


def test_save_link_destination_persists_through_incremental_update(tmp_path: Path) -> None:
    source = Path("tests/fixtures/pdfminer.six/samples/simple5.pdf")
    target = tmp_path / "linked.pdf"
    with LocalDocument.open(source) as document:
        assert document.links
        updated = document.save_link(0, target, destination="new-destination")
    assert updated.startswith(source.read_bytes())
    with LocalDocument.open(target) as document:
        assert document.links[0].url == "new-destination"


def test_save_comment_contents_persists_with_popup_fixture(tmp_path: Path) -> None:
    source = Path("tests/fixtures/pdfminer.six/samples/contrib/issue-1082-annotations.pdf")
    target = tmp_path / "commented.pdf"
    with LocalDocument.open(source) as document:
        comments = document.comments
        assert comments
        assert comments[0].subtype == "Text"
        annotation_index = next(
            index
            for index, annotation in enumerate(document.annotations)
            if annotation.subtype == "Text"
        )
        updated = document.save_annotation(annotation_index, target, contents="updated comment")
    assert updated.startswith(source.read_bytes())
    with LocalDocument.open(target) as document:
        assert any(annotation.contents == "updated comment" for annotation in document.comments)
        assert any(annotation.subtype == "Popup" for annotation in document.annotations)


def test_stamp_overlay_split_and_merge_outputs_reopen(tmp_path: Path) -> None:
    source = Path("tests/fixtures/pdfminer.six/samples/simple1.pdf")
    with LocalDocument.open(source) as document:
        stamped = document.stamp("STAMP", bbox=(10, 10, 60, 25))
        split_paths = []
        for index, data in enumerate(stamped.split(((1, 1),))):
            path = tmp_path / f"split-{index}.pdf"
            path.write_bytes(data)
            split_paths.append(path)
        with LocalDocument.open(split_paths[0]) as reopened:
            assert "STAMP" in reopened.structured.text
        merged = LocalDocument.merge((stamped, stamped))
    merged_path = tmp_path / "merged.pdf"
    merged_path.write_bytes(merged)
    with LocalDocument.open(merged_path) as reopened:
        assert len(reopened.pages) == 2
        assert reopened.structured.text.count("STAMP") == 2


def test_stamp_output_has_rendered_pixels(tmp_path: Path) -> None:
    source = Path("tests/fixtures/pdfminer.six/samples/simple1.pdf")
    with LocalDocument.open(source) as document:
        output = tmp_path / "stamped.pdf"
        document.stamp("STAMP", bbox=(10, 10, 60, 25)).write(output)
    with LocalDocument.open(output) as document:
        rendered = document.source_pdf.pages[0].render()
        assert rendered.raster_size() == (612, 792)
        assert any(item.kind == "text" for item in rendered.display_list.items)


def test_write_redacted_removes_covered_text_from_emitted_pdf(tmp_path: Path) -> None:
    document = LocalDocument.__new__(LocalDocument)
    document.pdf = None
    document.structured = Document(
        pages=(
            Page(
                page_number=1,
                width=200,
                height=200,
                blocks=(
                    Block(
                        0,
                        BlockKind.PARAGRAPH,
                        (TextLine("SECRET", bbox=(10, 10, 80, 30)),),
                        bbox=(10, 10, 80, 30),
                    ),
                ),
                annotations=(Annotation("Redact", (0, 0, 100, 50)),),
            ),
        )
    )
    output = tmp_path / "redacted.pdf"
    document.write_redacted(output)
    with LocalDocument.open(output) as reopened:
        assert "SECRET" not in reopened.structured.text


def test_redaction_audit_preserves_unrelated_content_and_geometry(tmp_path: Path) -> None:
    document = LocalDocument.__new__(LocalDocument)
    document.pdf = None
    document.structured = Document(
        pages=(
            Page(
                page_number=1,
                width=200,
                height=200,
                blocks=(
                    Block(
                        0,
                        BlockKind.PARAGRAPH,
                        (TextLine("SECRET", bbox=(10, 10, 80, 30)),),
                        bbox=(10, 10, 80, 30),
                    ),
                    Block(
                        1,
                        BlockKind.PARAGRAPH,
                        (TextLine("PUBLIC", bbox=(10, 100, 80, 120)),),
                        bbox=(10, 100, 80, 120),
                    ),
                ),
                annotations=(Annotation("Redact", (0, 0, 100, 50), "reason"),),
            ),
        )
    )
    assert document.redactions[0].bbox == (0, 0, 100, 50)
    output = tmp_path / "audited-redacted.pdf"
    document.write_redacted(output)
    with LocalDocument.open(output) as reopened:
        assert "SECRET" not in reopened.structured.text
        assert "PUBLIC" in reopened.structured.text
