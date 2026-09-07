from pathlib import Path

import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

from .test_pymupdf_geometry import internal_geometry_expected
from .test_pymupdf_text import internal_PDFS, real_pymupdf

pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("path", internal_PDFS, ids=lambda path: path.name)
def test_native_save_reopen(path: Path) -> None:
    with real_pymupdf.open(path) as reference, compat_pymupdf.open(path) as actual:
        original = reference[0].get_text("rawdict")
        for document in [reference, actual]:
            data = document.tobytes(no_new_id=True)
            with real_pymupdf.open(stream=data) as reopened:
                assert reopened[0].get_text("rawdict") == internal_geometry_expected(original)
                assert reopened.metadata == reference.metadata


def test_native_blank_pages_and_objects() -> None:
    with real_pymupdf.open() as reference, compat_pymupdf.open() as actual:
        for document in [reference, actual]:
            document.new_page(width=200, height=300)
            document.new_page(pno=0, width=400, height=500)
        assert actual.xref_length() == reference.xref_length()
        for xref in range(1, reference.xref_length()):
            assert actual.xref_object(xref) == reference.xref_object(xref)
        with real_pymupdf.open(stream=actual.tobytes(no_new_id=True)) as reopened:
            assert [tuple(p.rect) for p in reopened] == [tuple(p.rect) for p in reference]


@pytest.mark.parametrize("compress", [False, True])
def test_native_stream_edit_and_snapshot_lifetime(compress: bool) -> None:
    with (
        real_pymupdf.open(internal_PDFS[0]) as reference,
        compat_pymupdf.open(internal_PDFS[0]) as actual,
    ):
        results = []
        for document in [reference, actual]:
            page = document[0]
            snapshot = page.get_textpage()
            before = snapshot.extractText()
            xref = document.get_new_xref()
            document.update_object(xref, "<<>>")
            document.update_stream(
                xref, b"BT /helv 11 Tf 50 50 Td (Updated text) Tj ET\n", compress=compress
            )
            document.xref_set_key(page.xref, "Contents", f"{xref} 0 R")
            results.append(page.get_text())
            assert snapshot.extractText() == before
            data = document.tobytes(no_new_id=True)
            with real_pymupdf.open(stream=data) as reopened:
                assert reopened[0].get_text() == results[-1]
        assert results[0] == results[1]


@pytest.mark.parametrize("method", ["update_object", "xref_set_key"])
def test_native_metadata_object_updates(method: str) -> None:
    results = []
    for module in [real_pymupdf, compat_pymupdf]:
        with module.open(internal_PDFS[0]) as document:
            xref = document.get_new_xref()
            document.update_object(xref, "<< /Title (Before) /Author (Writer) >>")
            document.xref_set_key(-1, "Info", f"{xref} 0 R")
            if method == "update_object":
                document.update_object(xref, "<< /Title (After) /Author (Writer) >>")
            else:
                document.xref_set_key(xref, "Title", "(After)")
            results.append(document.metadata)
            with real_pymupdf.open(stream=document.tobytes(no_new_id=True)) as reopened:
                results.append(reopened.metadata)
    assert results[:2] == results[2:]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("Nested/Value", "[1 2 /Name]"),
        ("Created/Child", "(Text)"),
        ("Scalar/Child", "true"),
        ("Nested/Value", "null"),
    ],
)
def test_native_dictionary_mutation_and_persistence(key: str, value: str) -> None:
    results = []
    for module in [real_pymupdf, compat_pymupdf]:
        with module.open(internal_PDFS[0]) as document:
            xref = document.get_new_xref()
            document.update_object(xref, "<< /Nested << /Value 1 >> /Scalar 2 >>")
            document.xref_set_key(xref, key, value)
            results.append(document.xref_object(xref))
            with real_pymupdf.open(stream=document.tobytes(no_new_id=True)) as reopened:
                assert reopened.xref_object(xref) == results[-1]
    assert results[0] == results[1]


@pytest.mark.parametrize("compress", [False, True])
def test_compressed_stream_replacement(compress: bool) -> None:
    results = []
    for module in [real_pymupdf, compat_pymupdf]:
        with module.open(internal_PDFS[0]) as document:
            xref = document.get_new_xref()
            document.update_object(xref, "<< /Type /XObject /Subtype /Form /BBox [0 0 10 10] >>")
            for payload in [b"Native stream\n" * 200, b"Short"]:
                document.update_stream(xref, payload, compress=compress)
                results.append((document.xref_object(xref), document.xref_stream_raw(xref)))
                with real_pymupdf.open(stream=document.tobytes(no_new_id=True)) as reopened:
                    assert reopened.xref_stream(xref) == payload
    assert results[:2] == results[2:]


def test_updating_stream_dictionary_retains_bytes() -> None:
    results = []
    for module in [real_pymupdf, compat_pymupdf]:
        with module.open(internal_PDFS[0]) as document:
            xref = document.get_new_xref()
            document.update_object(xref, "<<>>")
            document.update_stream(xref, b"Preserved stream")
            document.update_object(xref, "<< /Custom true >>")
            results.append(
                (
                    document.xref_object(xref),
                    document.xref_is_stream(xref),
                    document.xref_stream(xref),
                )
            )
    assert results[0] == results[1]


def test_save_id_lifecycle() -> None:
    for module in [real_pymupdf, compat_pymupdf]:
        with module.open(internal_PDFS[0]) as document:
            data = document.tobytes()
            first_id = document.xref_get_key(-1, "ID")
            assert first_id[0] == "array"
            with real_pymupdf.open(stream=data) as reopened:
                assert reopened.xref_get_key(-1, "ID") == first_id
            document.tobytes(no_new_id=True)
            assert document.xref_get_key(-1, "ID") == first_id
            document.tobytes()
            assert document.xref_get_key(-1, "ID") != first_id


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("update_object", (0, "<<>>")),
        ("update_object", (-1, "<<>>")),
        ("xref_set_key", (0, "A", "1")),
        ("update_stream", (0, b"Data")),
    ],
)
def test_invalid_mutation_references(method: str, args: tuple[object, ...]) -> None:
    results = []
    for module in [real_pymupdf, compat_pymupdf]:
        with module.open(internal_PDFS[0]) as document:
            try:
                getattr(document, method)(*args)
            except Exception as error:
                results.append((type(error).__name__, str(error)))
            else:
                results.append(None)
    assert results[0] == results[1]


def test_file_and_stream_save(tmp_path: Path) -> None:
    from io import BytesIO

    for module in [real_pymupdf, compat_pymupdf]:
        source = tmp_path / (module.__name__ + ".pdf")
        source.write_bytes(internal_PDFS[0].read_bytes())
        with module.open(source) as document:
            with pytest.raises(ValueError, match="save to original must be incremental"):
                document.save(source)
            buffer = BytesIO()
            document.save(buffer, no_new_id=True)
            output = source.with_suffix(".out.pdf")
            document.save(output, no_new_id=True)
        for data in [buffer.getvalue(), output.read_bytes()]:
            with real_pymupdf.open(stream=data) as reopened:
                with real_pymupdf.open(source) as original:
                    assert reopened[0].get_text() == original[0].get_text()


def test_blank_page_insertion_preserves_inherited_page_settings() -> None:
    with real_pymupdf.open() as source:
        page = source.new_page(width=240, height=360)
        root = int(source.xref_get_key(page.xref, "Parent")[1].split()[0])
        intermediate = source.get_new_xref()
        source.update_object(
            intermediate,
            f"<< /Type /Pages /Parent {root} 0 R /Kids [{page.xref} 0 R] "
            "/Count 1 /MediaBox [0 0 240 360] /Rotate 90 /Resources <<>> >>",
        )
        source.update_object(page.xref, f"<< /Type /Page /Parent {intermediate} 0 R >>")
        source.xref_set_key(root, "Kids", f"[{intermediate} 0 R]")
        data = source.tobytes(no_new_id=True)
    results = []
    for engine in (real_pymupdf, compat_pymupdf):
        with engine.open(stream=data) as document:
            document.new_page(width=100, height=200)
            with real_pymupdf.open(stream=document.tobytes(no_new_id=True)) as reopened:
                results.append([(tuple(p.rect), p.rotation) for p in reopened])
    assert results[0] == results[1]
