"""Check PDF MAC (ISO 32004) verification on real AES-256 files.

No file under tests/upstream carries a PDF MAC and no library the facades follow writes one, so
this script adds one. It appends an incremental update to a real revision 6 file that adds
/KDFSalt to the encryption dictionary, the ISO 32004 extension declaration to the catalog, and a
standalone /AuthCode to the trailer. Its token is CMS AuthenticatedData built here with
asn1crypto, following ISO 32004 and RFC 5652, apart from core's verifier.

A file with a correct MAC must open with its pages and text unchanged, and a save that keeps its
encryption must be refused. Every altered file must be refused: changed bytes, a wrong key or
MAC, a ByteRange that misses bytes, a missing declaration or salt, a MAC attached to a signature,
and unsupported or unprotected algorithms.

uv run --all-packages python scripts/check_pdf_mac.py
"""

import collections
import hashlib
import hmac
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

from asn1crypto import algos, cms
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.keywrap import aes_key_wrap

from core_pdf import EditableDocument, PdfDocument
from core_pdf.impl.exceptions import PdfError, PdfUnsupportedError
from core_pdf.impl.types import PdfName, PdfString
from core_pdf_crypto.pdf_mac import PdfMacIntegrityInfo
from core_pdf_spec.s_07_syntax.serialize import serialize_object

ENCRYPTION = "tests/upstream/pypdf/resources/encryption"
SOURCES = (
    (f"{ENCRYPTION}/r6-both-passwords.pdf", "foo"),
    (f"{ENCRYPTION}/r6-both-passwords.pdf", "bar"),
    (f"{ENCRYPTION}/r6-user-password.pdf", "asdfzxcv"),
)
INTEGRITY_INFO_OID = "1.0.32004.1.0"
WRAP_KDF_OID = "1.0.32004.1.1"
DIGESTS = {"sha256": hashlib.sha256, "sha384": hashlib.sha384, "sha512": hashlib.sha512}
WIDTH = 10  # digits of each ByteRange entry, zero-padded so the trailer's length is fixed


class Variant(NamedTuple):
    """How to build one file. The defaults build a correct PDF MAC."""

    digest: str = "sha256"
    digest_oid: str | None = None  # names an algorithm other than the one used
    location: str = "Standalone"
    declare_extension: bool = True
    salt_in_dictionary: bool = True
    auth_code: bool = True
    wrap_key_wrong: bool = False  # wrap the MAC key with a key not derived from the file key
    mac_key_wrong: bool = False  # compute the MAC with a key other than the wrapped one
    protection: str | None = None  # None, "match", or "mismatch"
    short_range: bool = False  # bytes after the second range that it does not cover
    tamper: Callable[[bytearray], None] | None = None  # change covered bytes after the MAC


def token(
    variant: Variant, file_key: bytes, salt: bytes, mac_key: bytes, data_digest: bytes
) -> bytes:
    digest_name = variant.digest_oid or variant.digest
    digest_algorithm = algos.DigestAlgorithm({"algorithm": digest_name})
    mac_algorithm = algos.HmacAlgorithm({"algorithm": "sha256"})
    integrity = PdfMacIntegrityInfo({"version": 0, "data_digest": data_digest}).dump()

    kek_input = os.urandom(32) if variant.wrap_key_wrong else file_key
    kek = HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=b"PDFMAC").derive(kek_input)
    recipient = cms.RecipientInfo(
        name="pwri",
        value=cms.PasswordRecipientInfo(
            {
                "version": "v0",
                "key_derivation_algorithm": {"algorithm": WRAP_KDF_OID},
                "key_encryption_algorithm": {"algorithm": "aes256_wrap"},
                "encrypted_key": aes_key_wrap(kek, mac_key),
            }
        ),
    )

    attributes = [
        {"type": "content_type", "values": [INTEGRITY_INFO_OID]},
        {
            "type": "message_digest",
            "values": [DIGESTS[variant.digest](integrity).digest()],
        },
    ]
    if variant.protection is not None:
        protected_mac = "sha256" if variant.protection == "match" else "sha512"
        attributes.append(
            {
                "type": "cms_algorithm_protection",
                "values": [
                    {
                        "digest_algorithm": digest_algorithm,
                        "mac_algorithm": algos.HmacAlgorithm({"algorithm": protected_mac}),
                    }
                ],
            }
        )
    auth_attrs = cms.CMSAttributes(attributes)

    content = cms.ContentInfo(
        {
            "content_type": "authenticated_data",
            "content": cms.AuthenticatedData(
                {
                    "version": "v0",
                    "recipient_infos": [recipient],
                    "mac_algorithm": mac_algorithm,
                    "digest_algorithm": digest_algorithm,
                    "encap_content_info": {
                        "content_type": INTEGRITY_INFO_OID,
                        "content": integrity,
                    },
                    "auth_attrs": auth_attrs,
                    "mac": b"\0" * 32,
                }
            ),
        }
    )
    # RFC 5652 9.2: the MAC covers the DER of authAttrs with its own SET OF tag.
    parsed = cms.ContentInfo.load(content.dump())
    signed = parsed["content"]["auth_attrs"].untag().dump(force=True)
    mac_input_key = os.urandom(32) if variant.mac_key_wrong else mac_key
    parsed["content"]["mac"] = hmac.new(mac_input_key, signed, hashlib.sha256).digest()
    return parsed.dump(force=True)


def with_mac(path: str, password: str, variant: Variant) -> bytes:
    """The file at path with an incremental update that adds a PDF MAC built as variant says."""
    original = Path(path).read_bytes()
    with PdfDocument.open(path, password=password) as source:
        handler: Any = source.security_handler
        file_key = handler.file_key
        editor = EditableDocument(source)
        trailer = dict(editor.trailer)
        root: Any = trailer["Root"]
        encrypt_reference: Any = editor.encrypt_entry
        catalog: dict[Any, Any] = dict(source.resolver.dict_at(trailer, "Root") or {})
        encrypt: dict[Any, Any] = dict(source.resolver.dict_at(trailer, "Encrypt") or {})

    salt = os.urandom(32)
    if variant.salt_in_dictionary:
        encrypt["KDFSalt"] = PdfString(salt, is_literal=False)
    if variant.declare_extension:

        def encrypted(text: bytes) -> PdfString:
            data = handler.encrypt(root.object_number, root.generation_number, text)
            return PdfString(data, is_literal=False)

        catalog["Extensions"] = {
            "ISO_": [
                {
                    "Type": PdfName.of("DeveloperExtensions"),
                    "BaseVersion": PdfName.of("2.0"),
                    "ExtensionLevel": 32004,
                    "ExtensionRevision": encrypted(b":2024"),
                    "URL": encrypted(b"https://www.iso.org/standard/45877.html"),
                }
            ]
        }

    update = bytearray(original if original.endswith(b"\n") else original + b"\n")
    offsets = {}
    for reference, value in ((root, catalog), (encrypt_reference, encrypt)):
        offsets[reference.object_number] = (len(update), reference.generation_number)
        update += f"{reference.object_number} {reference.generation_number} obj\n".encode()
        update += serialize_object(value) + b"\nendobj\n"
    xref_offset = len(update)
    update += b"xref\n0 1\n0000000000 65535 f \n"
    for number, (offset, generation) in sorted(offsets.items()):
        update += f"{number} 1\n{offset:010d} {generation:05d} n \n".encode()

    mac_key = os.urandom(32)
    placeholder = token(
        variant, file_key, salt, mac_key, b"\0" * hashlib.new(variant.digest).digest_size
    )
    zero = b"0" * WIDTH
    trailer_entries = {
        key: value for key, value in trailer.items() if key not in {"Prev", "AuthCode"}
    }
    trailer_entries["Prev"] = original_startxref(original)
    update += b"trailer\n<<" + serialize_object(trailer_entries)[2:-2]
    if variant.auth_code:
        update += (
            b" /AuthCode << /MACLocation /"
            + variant.location.encode()
            + b" /ByteRange [0 "
            + zero
            + b" "
            + zero
            + b" "
            + zero
            + b"] /MAC "
        )
        mac_start = len(update)
        update += b"<" + b"0" * (2 * len(placeholder)) + b">"
        mac_end = len(update)
        update += b" >>"
    update += f" >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    if not variant.auth_code:
        return bytes(update)

    covered_end = len(update)
    if variant.short_range:
        update += b"% not covered\n"
    byte_range = b"0 %0*d %0*d %0*d" % (
        WIDTH,
        mac_start,
        WIDTH,
        mac_end,
        WIDTH,
        covered_end - mac_end,
    )
    range_at = update.index(b"/ByteRange [0 " + zero, 0, mac_start) + len(b"/ByteRange [")
    update[range_at : range_at + len(byte_range)] = byte_range

    data_digest = DIGESTS[variant.digest](
        bytes(update[:mac_start]) + bytes(update[mac_end:covered_end])
    ).digest()
    real = token(variant, file_key, salt, mac_key, data_digest)
    assert len(real) == len(placeholder)
    update[mac_start:mac_end] = b"<" + real.hex().upper().encode() + b">"
    if variant.tamper is not None:
        variant.tamper(update)
    return bytes(update)


def original_startxref(data: bytes) -> int:
    at = data.rindex(b"startxref")
    return int(data[at + len(b"startxref") :].split()[0])


def flip_old_trailer_id(data: bytearray) -> None:
    """Change one digit of the first trailer's /ID, which the newest trailer replaces."""
    at = data.index(b"/ID [<") + len(b"/ID [<")
    data[at] = ord("0") if data[at] != ord("0") else ord("1")


def page_texts(document: Any) -> list[str]:
    return ["\n".join(block.text for block in page.extract().blocks) for page in document.pages]


def outcome(data: bytes, password: str) -> tuple[str, str]:
    """'opened', or the error opening and reading the file raised, with its and its cause's text."""
    try:
        with PdfDocument.open(data, password=password) as document:
            page_texts(document)
    except PdfError as error:
        return type(error).__name__, f"{error} <- {error.__cause__}"
    return "opened", ""


# Each altered file, and words the error or its cause must contain, so a file refused for
# some other reason (a parse error, say) does not count.
REFUSED = {
    "tampered": (Variant(tamper=flip_old_trailer_id), "document digest does not match"),
    "wrong_wrapping_key": (Variant(wrap_key_wrong=True), "key unwrap failed"),
    "wrong_mac_key": (Variant(mac_key_wrong=True), "verification failed"),
    "range_misses_bytes": (Variant(short_range=True), "does not cover the entire file"),
    "no_extension": (Variant(declare_extension=False), "extension declaration"),
    "no_salt": (Variant(salt_in_dictionary=False), "Invalid PDF MAC"),
    "salt_without_auth_code": (Variant(auth_code=False), "Invalid PDF MAC"),
    "unprotected_algorithm": (Variant(protection="mismatch"), "algorithm is not protected"),
}
UNSUPPORTED = {
    "attached_to_signature": (Variant(location="AttachedToSig"), "Attached-to-signature"),
    "unknown_location": (Variant(location="Elsewhere"), "Unsupported PDF MAC location"),
    "sha1_digest": (Variant(digest_oid="sha1"), "digest algorithm: 1.3.14.3.2.26"),
}
ACCEPTED = {
    "sha256": Variant(),
    "sha384": Variant(digest="sha384"),
    "sha512": Variant(digest="sha512"),
    "algorithm_protection": Variant(protection="match"),
}


def check_source(path: str, password: str, counts: collections.Counter[str]) -> None:
    with PdfDocument.open(path, password=password) as source:
        expected = page_texts(source)
    for name, variant in ACCEPTED.items():
        data = with_mac(path, password, variant)
        try:
            with PdfDocument.open(data, password=password) as document:
                same = page_texts(document) == expected
                kept = refused(EditableDocument(document).save)
                kept = kept and refused(EditableDocument(document).save_incremental)
                decrypted = EditableDocument(document).save(encryption="decrypt")
        except PdfError as error:
            counts["valid_refused"] += 1
            print(f"{name}: {path}: refused a correct MAC: {type(error).__name__}: {error}")
            continue
        counts["valid_opened" if same else "valid_text_changed"] += 1
        counts["keep_encryption_refused" if kept else "keep_encryption_saved"] += 1
        plain = not decrypted.encrypted and b"/AuthCode" not in decrypted
        counts["decrypted_save" if plain else "decrypted_save_wrong"] += 1
        wrong, _text = outcome(data, "definitely wrong")
        good = wrong == "PdfPasswordError"
        counts["wrong_password_refused" if good else "wrong_password_mishandled"] += 1
        if not same:
            print(f"{name}: {path}: text changed")
    for group, cases, error in (
        ("altered", REFUSED, "PdfDecryptionError"),
        ("unsupported", UNSUPPORTED, "PdfUnsupportedError"),
    ):
        for name, (variant, reason) in cases.items():
            kind, text = outcome(with_mac(path, password, variant), password)
            good = kind == error and reason in text
            counts[f"{group}_refused" if good else f"{group}_mishandled"] += 1
            if not good:
                print(f"{name}: {path}: {kind}: {text}")


def refused(action: Callable[[], object]) -> bool:
    try:
        action()
    except PdfUnsupportedError:
        return True
    return False


def main() -> int:
    counts: collections.Counter[str] = collections.Counter()
    for path, password in SOURCES:
        check_source(path, password, counts)
    print(dict(counts))
    allowed = {
        "valid_opened",
        "keep_encryption_refused",
        "decrypted_save",
        "wrong_password_refused",
        "altered_refused",
        "unsupported_refused",
    }
    expected = len(SOURCES) * len(ACCEPTED)
    complete = counts["valid_opened"] == expected
    complete = complete and counts["altered_refused"] == len(SOURCES) * len(REFUSED)
    complete = complete and counts["unsupported_refused"] == len(SOURCES) * len(UNSUPPORTED)
    return 0 if complete and not set(counts) - allowed else 1


if __name__ == "__main__":
    sys.exit(main())
