from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

real_pymupdf = pytest.importorskip("pymupdf")
real_pypdf = pytest.importorskip("pypdf")
pytestmark = pytest.mark.compat_differential


def internal_encrypted_source(algorithm: str, user: str, owner: str) -> bytes:
    with real_pymupdf.open() as document:
        document.new_page().insert_text((50, 50), "Protected text")
        source = document.tobytes()
    writer = real_pypdf.PdfWriter(clone_from=BytesIO(source))
    writer.add_metadata({"/Title": "Protected title"})
    writer.encrypt(user, owner_password=owner, algorithm=algorithm)
    stream = BytesIO()
    writer.write(stream)
    return stream.getvalue()


def internal_document_state(document: Any) -> tuple[object, ...]:
    state = (
        document.is_encrypted,
        document.page_count,
        document.metadata,
    )
    if document.is_encrypted:
        for operation in (lambda: document[0], document.get_toc, document.tobytes):
            with pytest.raises(ValueError):
                operation()
        return state
    return (*state, document[0].get_text())


@pytest.mark.parametrize("algorithm", ["RC4-40", "RC4-128", "AES-128", "AES-256-R5", "AES-256"])
@pytest.mark.parametrize(
    "credentials", [("user", "owner"), ("", "owner"), ("same", "same"), ("user", "")]
)
def test_authentication_roles_and_repeated_attempts(
    algorithm: str, credentials: tuple[str, str]
) -> None:
    source = internal_encrypted_source(algorithm, *credentials)
    with (
        real_pymupdf.open(stream=source) as expected,
        compat_pymupdf.open(stream=source) as actual,
    ):
        assert actual.needs_pass == expected.needs_pass
        assert internal_document_state(actual) == internal_document_state(expected)
        loaded_pages: list[tuple[Any, Any]] = []
        valid_password = None if expected.is_encrypted else ""
        for password in ("", "wrong", "user", "owner", "same", "wrong"):
            previous_text = None if expected.is_encrypted else expected[0].get_text()
            status = actual.authenticate(password)
            assert status == expected.authenticate(password)
            if status:
                valid_password = password
            assert actual.is_encrypted == expected.is_encrypted
            if expected.is_encrypted:
                assert internal_document_state(actual) == internal_document_state(expected)
                continue
            if status == 0:
                assert actual[0].get_text() == previous_text
            # Restore valid authentication before inspecting content: failed attempts
            # leave MuPDF's prior unlocked flag but replace its internal cipher key.
            assert valid_password is not None
            for document in (actual, expected):
                document.authenticate(valid_password)
            assert internal_document_state(actual) == internal_document_state(expected)
            if not expected.is_encrypted:
                loaded_pages.append((actual[0], expected[0]))
            for actual_page, expected_page in loaded_pages:
                assert actual_page.get_text() == expected_page.get_text()
        # MuPDF also replaces its cipher key when reading needs_pass after login,
        # so compare that status after the content and page-lifetime checks.
        assert actual.needs_pass == expected.needs_pass
    assert actual.is_closed == expected.is_closed
    for document in (actual, expected):
        with pytest.raises(ValueError):
            document.authenticate("user")


@pytest.mark.parametrize("authenticate", [False, True])
def test_locked_file_lifetime(tmp_path: Path, authenticate: bool) -> None:
    path = tmp_path / "encrypted.pdf"
    path.write_bytes(internal_encrypted_source("AES-256", "user", "owner"))
    with real_pymupdf.open(path) as expected, compat_pymupdf.open(path) as actual:
        assert internal_document_state(actual) == internal_document_state(expected)
        if authenticate:
            assert actual.authenticate("owner") == expected.authenticate("owner")
            assert internal_document_state(actual) == internal_document_state(expected)
    assert actual.is_closed == expected.is_closed


def test_authentication_on_unencrypted_document() -> None:
    with real_pymupdf.open() as expected, compat_pymupdf.open() as actual:
        for password in ("", "wrong"):
            assert actual.authenticate(password) == expected.authenticate(password)


@pytest.mark.parametrize("algorithm", ["AES-256-R5", "AES-256"])
@pytest.mark.parametrize("owner_field", ["O", "OE"])
@pytest.mark.parametrize("user", ["", "user"])
def test_malformed_owner_credential_rejects_document(
    algorithm: str, owner_field: str, user: str
) -> None:
    with real_pymupdf.open() as document:
        document.new_page().insert_text((50, 50), "User-authenticated text")
        source = document.tobytes()
    writer = real_pypdf.PdfWriter(clone_from=BytesIO(source))
    writer.encrypt(user, owner_password="owner", algorithm=algorithm)
    writer._encrypt_entry[real_pypdf.generic.NameObject("/" + owner_field)] = (
        real_pypdf.generic.ByteStringObject(b"short")
    )
    buffer = BytesIO()
    writer.write(buffer)
    source = buffer.getvalue()
    for module in (real_pymupdf, compat_pymupdf):
        with pytest.raises(module.FileDataError):
            module.open(stream=source)
