from io import BytesIO
from typing import Any

import pytest

from core_pdf.api.compat import pypdf as compat_pypdf

from .support import FIXTURES_ROOT

real_pypdf = pytest.importorskip("pypdf")
pytestmark = pytest.mark.compat_differential


def internal_box_snapshot(reader: Any) -> list[object]:
    pages = reader.pages
    result: list[object] = [len(pages)]
    for page in pages:
        result.append(page.extract_text())
        for name in ("mediabox", "cropbox"):
            try:
                result.append((name, tuple(getattr(page, name))))
            except (TypeError, ValueError) as error:
                result.append((name, type(error).__name__))
    return result


@pytest.mark.parametrize("media", ["missing", "null", "valid", "short", "long"])
@pytest.mark.parametrize("crop", ["missing", "null", "valid"])
@pytest.mark.parametrize("inherited", [False, True])
@pytest.mark.parametrize("indirect", [False, True])
def test_page_box_resolution_matches_reference(
    media: str, crop: str, inherited: bool, indirect: bool
) -> None:
    generic = real_pypdf.generic
    writer = real_pypdf.PdfWriter()
    page = writer.add_blank_page(width=200, height=300)
    del page["/MediaBox"]
    owner = writer.root_object["/Pages"] if inherited else page
    for name, kind in (("/MediaBox", media), ("/CropBox", crop)):
        if kind == "missing":
            continue
        if kind == "null":
            value = generic.NullObject()
        else:
            coordinates = [10, 20, 190, 280]
            if kind == "short":
                coordinates.pop()
            elif kind == "long":
                coordinates.append(999)
            value = generic.ArrayObject([generic.NumberObject(item) for item in coordinates])
        owner[generic.NameObject(name)] = writer._add_object(value) if indirect else value
    stream = BytesIO()
    writer.write(stream)
    source = stream.getvalue()

    with (
        real_pypdf.PdfReader(BytesIO(source), strict=False) as expected,
        compat_pypdf.PdfReader(BytesIO(source), strict=False) as actual,
    ):
        assert internal_box_snapshot(actual) == internal_box_snapshot(expected)


def test_missing_page_bounds_are_rejected_only_on_access() -> None:
    source = FIXTURES_ROOT / "pikepdf/tests/resources/cyclic-toc.pdf"
    with (
        real_pypdf.PdfReader(source, strict=False) as expected,
        compat_pypdf.PdfReader(source, strict=False) as actual,
    ):
        snapshot = internal_box_snapshot(expected)
        assert ("mediabox", "TypeError") in snapshot
        assert internal_box_snapshot(actual) == snapshot


def test_null_page_box_shadows_ancestor_box() -> None:
    generic = real_pypdf.generic
    name = generic.NameObject
    writer = real_pypdf.PdfWriter()
    page = writer.add_blank_page(width=200, height=300)
    del page["/MediaBox"]
    root = writer.root_object["/Pages"]
    root[name("/MediaBox")] = generic.ArrayObject(
        [generic.NumberObject(item) for item in (10, 20, 190, 280)]
    )
    parent = generic.DictionaryObject(
        {
            name("/Type"): name("/Pages"),
            name("/Kids"): generic.ArrayObject([page.indirect_reference]),
            name("/Count"): generic.NumberObject(1),
            name("/Parent"): root.indirect_reference,
            name("/MediaBox"): generic.NullObject(),
        }
    )
    parent_ref = writer._add_object(parent)
    root[name("/Kids")] = generic.ArrayObject([parent_ref])
    page[name("/Parent")] = parent_ref
    stream = BytesIO()
    writer.write(stream)
    with (
        real_pypdf.PdfReader(BytesIO(stream.getvalue())) as expected,
        compat_pypdf.PdfReader(BytesIO(stream.getvalue())) as actual,
    ):
        assert internal_box_snapshot(actual) == internal_box_snapshot(expected)


@pytest.mark.parametrize("parent_kind", ["missing", "wrong"])
@pytest.mark.parametrize("parent_location", ["page", "ancestor"])
@pytest.mark.parametrize("null_shadow", [False, True])
@pytest.mark.parametrize("indirect", [False, True])
def test_box_inheritance_uses_page_tree_ancestry(
    parent_kind: str, parent_location: str, null_shadow: bool, indirect: bool
) -> None:
    generic = real_pypdf.generic
    name = generic.NameObject
    writer = real_pypdf.PdfWriter()
    page = writer.add_blank_page(width=200, height=300)
    del page["/MediaBox"]
    root = writer.root_object["/Pages"]
    for key, coordinates in (
        ("/MediaBox", (10, 20, 190, 280)),
        ("/CropBox", (20, 30, 180, 270)),
    ):
        box = generic.ArrayObject([generic.NumberObject(item) for item in coordinates])
        root[name(key)] = writer._add_object(box) if indirect else box
    ancestor = generic.DictionaryObject(
        {
            name("/Type"): name("/Pages"),
            name("/Kids"): generic.ArrayObject([page.indirect_reference]),
            name("/Count"): generic.NumberObject(1),
            name("/Parent"): root.indirect_reference,
        }
    )
    if null_shadow:
        null = generic.NullObject()
        ancestor[name("/MediaBox")] = writer._add_object(null) if indirect else null
    ancestor_ref = writer._add_object(ancestor)
    root[name("/Kids")] = generic.ArrayObject([ancestor_ref])
    page[name("/Parent")] = ancestor_ref
    broken_node = page if parent_location == "page" else ancestor
    if parent_kind == "missing":
        del broken_node["/Parent"]
    else:
        unrelated = generic.DictionaryObject(
            {
                name("/Type"): name("/Pages"),
                name("/Kids"): generic.ArrayObject(),
                name("/Count"): generic.NumberObject(0),
                name("/MediaBox"): generic.ArrayObject(
                    [generic.NumberObject(item) for item in (100, 200, 900, 1000)]
                ),
            }
        )
        broken_node[name("/Parent")] = writer._add_object(unrelated)
    stream = BytesIO()
    writer.write(stream)
    with (
        real_pypdf.PdfReader(BytesIO(stream.getvalue())) as expected,
        compat_pypdf.PdfReader(BytesIO(stream.getvalue())) as actual,
    ):
        expected_snapshot = internal_box_snapshot(expected)
        assert ("mediabox", "TypeError" if null_shadow else (10, 20, 190, 280)) in (
            expected_snapshot
        )
        assert ("cropbox", (20, 30, 180, 270)) in expected_snapshot
        assert internal_box_snapshot(actual) == expected_snapshot
