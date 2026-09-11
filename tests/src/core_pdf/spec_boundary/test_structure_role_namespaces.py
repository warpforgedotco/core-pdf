# SPDX-License-Identifier: AGPL-3.0-only
"""Native structure projection preserves resolved namespace identities and reader recovery."""

import pytest

from core_pdf import PdfDocument
from core_pdf.impl._impl.document.structure import StructureElement
from core_pdf_spec.s_14_structure.roles import PDF_1_7_NAMESPACE, PDF_2_0_NAMESPACE


def internal_pdf(
    element: bytes,
    *,
    version: str = "2.0",
    mapping: bytes = b"",
    namespaces: tuple[bytes, ...] = (),
) -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R /StructTreeRoot 3 0 R >>",
        b"<< /Type /Pages /Kids [] /Count 0 >>",
        b"<< /Type /StructTreeRoot /K 4 0 R /RoleMap << "
        + mapping
        + b" >> /Namespaces ["
        + b" ".join(f"{n + 5} 0 R".encode() for n in range(len(namespaces)))
        + b"] >>",
        b"<< /Type /StructElem /P 3 0 R " + element + b" >>",
        *namespaces,
    ]
    data = f"%PDF-{version}\n".encode()
    offsets = [0]
    for n, body in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(data)
    data += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode()
    data += b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets[1:])
    data += f"trailer\n<< /Root 1 0 R /Size {len(offsets)} >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return data


@pytest.mark.parametrize("version", ["1.4", "1.7", "2.0", "9.0"])
def test_document_role_namespaces_follow_declared_objects_even_in_older_headers(
    version: str,
) -> None:
    data = internal_pdf(
        b"/S /Heading /NS 5 0 R",
        version=version,
        namespaces=(
            b"<< /Type /Namespace /NS (urn:one) /RoleMapNS << /Heading [/Heading 6 0 R] >> >>",
            b"<< /Type /Namespace /NS (urn:two) /RoleMapNS << /Heading [/Title 7 0 R] >> >>",
            b"<< /Type /Namespace /NS (http://iso.org/pdf2/ssn) >>",
        ),
    )
    with PdfDocument(data) as document:
        tree = document.structure
        assert tree is not None
        element = tree.find("Title")
        assert element is not None
        assert element.type == "Heading"
        assert element.role_namespace == PDF_2_0_NAMESPACE
        assert element.role_resolution is not None
        assert element.role_resolution.status == "standard"
        assert element.role_error is None
        assert len(element.role_resolution.path) == 3


def test_absent_namespace_uses_transitive_global_map_and_pdf17_default() -> None:
    with PdfDocument(internal_pdf(b"/S /Custom", mapping=b"/Custom /Middle /Middle /P")) as doc:
        tree = doc.structure
        assert tree is not None
        element = tree.find("P")
        assert element is not None
        assert element.role_namespace == PDF_1_7_NAMESPACE


@pytest.mark.parametrize("value", [b"null", b"5 0 R", b"99 0 R"])
def test_optional_namespace_null_or_undefined_reference_defaults_to_pdf17(value: bytes) -> None:
    data = internal_pdf(b"/S /Custom /NS " + value, mapping=b"/Custom /P", namespaces=(b"null",))
    with PdfDocument(data) as doc:
        tree = doc.structure
        assert tree is not None
        element = tree.find("P")
        assert element is not None
        assert element.role_namespace == PDF_1_7_NAMESPACE
        assert element.role_error is None


def test_child_does_not_inherit_its_parents_namespace() -> None:
    data = internal_pdf(
        b"/S /Parent /NS 5 0 R /K << /Type /StructElem /S /Custom /P 4 0 R >>",
        mapping=b"/Custom /P",
        namespaces=(b"<< /NS (urn:custom) /RoleMapNS << /Parent /Sect /Custom /H1 >> >>",),
    )
    with PdfDocument(data) as doc:
        tree = doc.structure
        assert tree is not None
        child = tree.find("P")
        assert child is not None
        assert child.role_resolution is not None
        assert child.role_resolution.path[0].namespace is None
        assert child.role_namespace == PDF_1_7_NAMESPACE


@pytest.mark.parametrize(("version", "role"), [("1.4", "P"), ("1.5", "H1"), ("2.0", "H1")])
def test_document_standard_remapping_changes_at_pdf15(version: str, role: str) -> None:
    with PdfDocument(
        internal_pdf(b"/S /P", version=version, mapping=b"/P /Custom /Custom /H1")
    ) as doc:
        tree = doc.structure
        assert tree is not None
        assert tree.find(role) is not None


@pytest.mark.parametrize("explicit", [False, True])
def test_explicit_default_namespace_does_not_apply_the_global_map(explicit: bool) -> None:
    data = internal_pdf(
        b"/S /P" + (b" /NS 5 0 R" if explicit else b""),
        mapping=b"/P /H1",
        namespaces=(b"<< /NS (http://iso.org/pdf/ssn) >>",),
    )
    with PdfDocument(data) as doc:
        tree = doc.structure
        assert tree is not None
        element = tree.find("P" if explicit else "H1")
        assert element is not None
        assert element.role_namespace == PDF_1_7_NAMESPACE


@pytest.mark.parametrize("mapping", [b"/Custom 12", b"/Custom [/P 5 0 R]"])
def test_malformed_global_map_retains_original_type_and_recovery_diagnostic(mapping: bytes) -> None:
    with PdfDocument(internal_pdf(b"/S /Custom", mapping=mapping)) as doc:
        tree = doc.structure
        assert tree is not None
        element = tree.find("Custom")
        assert element is not None
        assert element.role_resolution is None
        assert element.role_namespace is None
        assert element.role_error is not None


def test_circular_map_is_exposed_without_hanging_tree_queries() -> None:
    with PdfDocument(internal_pdf(b"/S /Custom", mapping=b"/Custom /Other /Other /Custom")) as doc:
        tree = doc.structure
        assert tree is not None
        element = tree.find("Custom")
        assert element is not None
        assert element.role_resolution is not None
        assert element.role_resolution.status == "cycle"
        assert element.role_error is None


@pytest.mark.parametrize(
    "declaration",
    [
        b"<< /NS null >>",
        b"<< /NS /Invalid >>",
        b"<< /NS (urn:a) /RoleMapNS << /Custom [/P 99 0 R] >> >>",
        b"<< /NS (urn:a) /RoleMapNS << /Custom [/P << /NS (http://iso.org/pdf/ssn) >>] >> >>",
    ],
)
def test_reader_retains_original_role_when_namespace_declarations_are_malformed(
    declaration: bytes,
) -> None:
    data = internal_pdf(b"/S /Custom /NS 5 0 R", namespaces=(declaration,))
    with PdfDocument(data) as doc:
        tree = doc.structure
        assert tree is not None
        element = tree.find("Custom")
        assert element is not None
        assert element.role_error is not None
        assert element.role_resolution is None
        assert element.role_namespace is None


def test_reader_keeps_legacy_text_role_names_and_underdeclared_unicode_namespaces() -> None:
    data = internal_pdf(
        b"/S /Custom /NS 5 0 R",
        version="1.7",
        namespaces=(b"<< /NS <efbbbf75726e3a637573746f6d> /RoleMapNS << /Custom (P) >> >>",),
    )
    with PdfDocument(data) as doc:
        tree = doc.structure
        assert tree is not None
        element = tree.find("P")
        assert isinstance(element, StructureElement)
        assert element.role_resolution is not None
        assert element.role_resolution.path[0].namespace == "urn:custom"
        assert element.role_namespace == PDF_1_7_NAMESPACE
