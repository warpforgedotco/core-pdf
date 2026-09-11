# SPDX-License-Identifier: AGPL-3.0-only
"""Historical name grammar must not create, erase or bypass authentication."""

from pathlib import Path
from typing import cast

import pytest

from core_pdf import PdfDocument
from core_pdf.impl._impl.document import document as engine_document
from core_pdf.impl._impl.document import standards as reader_standards
from core_pdf_spec.exceptions import PdfDecryptionError, PdfUnsupportedError
from core_pdf_spec.s_07_syntax.xref import XRefScanner
from core_pdf_spec.standards import PdfVersion
from core_pdf_spec.types import PdfReference

FIXTURES = Path(__file__).resolve().parents[4] / "tests/fixtures/security_interop"


def internal_pdf(
    trailer: bytes, *, version: bytes = b"1.1", catalog: bytes = b"", recovered: bool = False
) -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R " + catalog + b" >>",
        b"<< /Type /Pages /Kids [] /Count 0 >>",
    ]
    data = b"%PDF-" + version + b"\n"
    offsets = []
    for n, body in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    start = len(data)
    if not recovered:
        data += b"xref\n0 3\n0000000000 65535 f \n"
        data += b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets)
    data += b"trailer << /Size 3 /Root 1 0 R " + trailer + b" >>\n"
    return data + (b"" if recovered else f"startxref\n{start}\n".encode()) + b"%%EOF\n"


def internal_encrypted(
    *,
    trailer_extra: bytes = b"",
    dictionary_extra: bytes = b"",
    catalog_extra: bytes = b"",
    version: bytes = b"1.1",
    escaped_security: bool = False,
    recovered: bool = False,
) -> bytes:
    # Rewrite a qpdf-produced R2 fixture without changing its encrypted bytes,
    # object identities, or IDs. Rebuild offsets so regular controls stay valid.
    original = (FIXTURES / "rc4-40-r2.pdf").read_bytes()
    start = XRefScanner.find_startxref(original)
    assert start is not None
    entries, _ = XRefScanner.load_section_chain(original, start, set())
    locations = sorted((entry.offset, key) for key, entry in entries.items() if entry.in_use)
    output = b"%PDF-" + version + b"\n"
    offsets: dict[int, int] = {}
    for i, (offset, key) in enumerate(locations):
        end = locations[i + 1][0] if i + 1 < len(locations) else start
        raw = original[offset:end]
        if key >> 16 == 1:
            raw = raw.replace(b" >>", b" " + catalog_extra + b" >>", 1)
        if b"/Filter /Standard" in raw:
            raw = raw.replace(b" >>", b" " + dictionary_extra + b" >>", 1)
            if escaped_security:
                raw = raw.replace(b"/Filter", b"/Fil#74er").replace(b"/V ", b"/#56 ")
                raw = raw.replace(b"/R ", b"/#52 ")
        offsets[key >> 16] = len(output)
        output += raw
    xref = len(output)
    if not recovered:
        output += f"xref\n0 {len(offsets) + 1}\n0000000000 65535 f \n".encode()
        output += b"".join(f"{offsets[n]:010} 00000 n \n".encode() for n in sorted(offsets))
    trailer = original[original.index(b"trailer", start) : original.index(b"startxref", start)]
    end = trailer.rfind(b">>")
    trailer = trailer[:end] + trailer_extra + b" " + trailer[end:]
    if escaped_security:
        trailer = trailer.replace(b"/Encrypt", b"/Encr#79pt")
    output += trailer
    return output + (b"" if recovered else f"startxref\n{xref}\n".encode()) + b"%%EOF\n"


@pytest.mark.parametrize("recovered", [False, True])
@pytest.mark.parametrize("key", [b"Encr#79pt", b"Auth#43ode"])
def test_legacy_private_security_names_are_not_authentication_declarations(
    key: bytes, recovered: bool
) -> None:
    data = internal_pdf(b"/" + key + b" 42", recovered=recovered)
    with PdfDocument(data) as document:
        assert document.decipher is None
        assert document.standards.effective_version == PdfVersion(1, 1)
        assert document.page_count() == 0
        assert document.xref_was_recovered == recovered


@pytest.mark.parametrize("recovered", [False, True])
@pytest.mark.parametrize("entry", [b"/Encrypt 42", b"/AuthCode 42", b"/AuthCode null"])
def test_real_malformed_security_declarations_survive_xref_recovery(
    entry: bytes, recovered: bool
) -> None:
    with pytest.raises(PdfUnsupportedError):
        PdfDocument(internal_pdf(entry, recovered=recovered))


@pytest.mark.parametrize("recovered", [False, True])
@pytest.mark.parametrize("key", [b"Encr#79pt", b"Auth#43ode"])
def test_modern_escaped_security_declarations_are_not_ignored(key: bytes, recovered: bool) -> None:
    with pytest.raises(PdfUnsupportedError):
        PdfDocument(internal_pdf(b"/" + key + b" 42", version=b"1.2", recovered=recovered))


@pytest.mark.parametrize("recovered", [False, True])
def test_private_null_alias_cannot_erase_real_password_authentication(recovered: bool) -> None:
    data = internal_encrypted(trailer_extra=b"/Encr#79pt null", recovered=recovered)
    with pytest.raises(PdfUnsupportedError, match="Incorrect password"):
        PdfDocument(data)
    with PdfDocument(data, password="user-40") as document:
        assert document.decipher is not None
        assert document.page_count() == 1
        info = document.metadata["info"]
        assert isinstance(info, dict)
        assert cast(dict[str, object], info)["Title"] == "Core PDF Security Fixture"


@pytest.mark.parametrize("recovered", [False, True])
def test_private_security_parameter_aliases_cannot_overwrite_literal_fields(
    recovered: bool,
) -> None:
    data = internal_encrypted(
        dictionary_extra=b"/#56 0 /#52 0 /Fil#74er /Bogus /#50 0", recovered=recovered
    )
    with pytest.raises(PdfUnsupportedError, match="Incorrect password"):
        PdfDocument(data)
    with PdfDocument(data, password="user-40") as document:
        assert document.decipher is not None
        info = document.metadata["info"]
        assert isinstance(info, dict)
        assert cast(dict[str, object], info)["Title"] == "Core PDF Security Fixture"
        params = document.resolver.resolve_dict(document.trailer_dict["Encrypt"])
        assert params is not None
        assert (params["V"], params["R"], params["P"]) == (1, 2, -4)


@pytest.mark.parametrize("version", [b"1.1", b"1.2"])
@pytest.mark.parametrize("recovered", [False, True])
def test_escaped_catalog_upgrade_selects_escaped_security_keys_before_authentication(
    version: bytes, recovered: bool
) -> None:
    data = internal_encrypted(
        version=version,
        catalog_extra=b"/V#65rsion /1.4",
        escaped_security=True,
        recovered=recovered,
    )
    with pytest.raises(PdfUnsupportedError, match="Incorrect password"):
        PdfDocument(data)
    with PdfDocument(data, password="user-40") as document:
        assert document.decipher is not None
        assert document.standards.effective_version == PdfVersion(1, 4)
        info = document.metadata["info"]
        assert isinstance(info, dict)
        assert cast(dict[str, object], info)["Title"] == "Core PDF Security Fixture"


@pytest.mark.parametrize("catalog", [b"/Version null", b"/V#65rsion null", b"/Version /2.0"])
def test_inconclusive_or_modern_probe_cannot_hide_literal_security_with_aliases(
    catalog: bytes,
) -> None:
    data = internal_encrypted(trailer_extra=b"/Encr#79pt null", catalog_extra=catalog)
    with pytest.raises(PdfUnsupportedError, match="Ambiguous security"):
        PdfDocument(data)


def test_inconclusive_probe_rejects_security_parameter_alias_collisions() -> None:
    data = internal_encrypted(catalog_extra=b"/Version null", dictionary_extra=b"/#56 0")
    with pytest.raises(PdfUnsupportedError, match="Ambiguous security"):
        PdfDocument(data)


def test_escaped_encrypt_reference_still_checks_nested_alias_collisions() -> None:
    data = internal_encrypted(catalog_extra=b"/Version /1.4", dictionary_extra=b"/#56 0").replace(
        b"/Encrypt 8 0 R", b"/Encr#79pt 8 0 R"
    )
    with pytest.raises(PdfUnsupportedError, match="Ambiguous security"):
        PdfDocument(data)


@pytest.mark.parametrize("recovered", [False, True])
def test_private_authcode_alias_cannot_hide_a_real_invalid_authcode(recovered: bool) -> None:
    data = internal_pdf(b"/AuthCode 42 /Auth#43ode null", recovered=recovered)
    with pytest.raises(PdfUnsupportedError, match="AuthCode"):
        PdfDocument(data)


def test_invalid_escaped_version_keeps_conservative_security_bootstrap_policy() -> None:
    # Without a usable upgrade declaration, this ambiguous document retains
    # modern security parsing rather than guessing that its security key is private.
    data = internal_pdf(b"/Encr#79pt 42", catalog=b"/V#65rsion null")
    with pytest.raises(PdfUnsupportedError, match="Encrypt"):
        PdfDocument(data)


def test_historical_escaped_version_upgrade_selects_security_before_authentication() -> None:
    data = internal_encrypted(catalog_extra=b"/V#65rsion /1.4", escaped_security=True)
    previous = XRefScanner.find_startxref(data)
    assert previous is not None
    root = len(data)
    data += b"1 0 obj\n<< /Type /Catalog /Pages 4 0 R >>\nendobj\n"
    xref = len(data)
    data += f"xref\n1 1\n{root:010} 00000 n \n".encode()
    data += f"trailer << /Size 9 /Root 1 0 R /Encrypt 8 0 R /Prev {previous} >>\n".encode()
    data += f"startxref\n{xref}\n%%EOF\n".encode()
    with pytest.raises(PdfUnsupportedError, match="Incorrect password"):
        PdfDocument(data)
    with PdfDocument(data, password="user-40") as document:
        assert document.decipher is not None
        assert document.standards.effective_version == PdfVersion(1, 4)
        assert document.standards.catalog_version is None
        info = document.metadata["info"]
        assert isinstance(info, dict)
        assert cast(dict[str, object], info)["Title"] == "Core PDF Security Fixture"


def test_version_probe_resolves_only_version_names_without_following_catalog_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = reader_standards.internal_VersionProbeResolver.resolve
    resolved: list[int] = []

    def guarded(self: reader_standards.internal_VersionProbeResolver, value: object) -> object:
        if isinstance(value, PdfReference):
            resolved.append(value.object_number)
            assert value.object_number == 1
        assert self.decipher is None
        return original(self, value)

    monkeypatch.setattr(reader_standards.internal_VersionProbeResolver, "resolve", guarded)
    with PdfDocument(internal_pdf(b"", catalog=b"/Metadata 42 0 R /Extensions 43 0 R")) as doc:
        assert doc.page_count() == 0
    assert resolved == [1]


@pytest.mark.parametrize("file", ["aes-256-r6-cbc-mac.pdf", "aes-256-r7-gcm-mac.pdf"])
@pytest.mark.parametrize("alias", [False, True])
def test_tampered_mac_old_header_is_rejected_before_full_catalog_discovery(
    file: str, alias: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = (FIXTURES / "pdf_mac" / file).read_bytes()
    data = data.replace(b"%PDF-2.0", b"%PDF-1.1", 1)
    if alias:
        end = data.rfind(b">>", 0, data.rfind(b"startxref"))
        data = data[:end] + b" /Auth#43ode null " + data[end:]

    def forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("full catalog discovery preceded MAC authentication")

    monkeypatch.setattr(engine_document, "discover_document_standards", forbidden)
    with pytest.raises(PdfDecryptionError, match="Invalid PDF MAC"):
        PdfDocument(data, password="user-mac-cbc" if "cbc" in file else "user-mac-gcm")
