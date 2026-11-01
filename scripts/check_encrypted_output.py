"""Check saving encrypted PDFs: the policy in docs/object-model.md, on real encrypted files.

Each file is opened, edited and saved every way the policy allows. The output must reopen with
the same pages and text, and must refuse a wrong password. qpdf, when installed, is a second
reader. Tampered files must be refused, and key material must stay out of reprs.

uv run --all-packages python scripts/check_encrypted_output.py
"""

import collections
import glob
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from core_pdf import EditableDocument, NewEncryption, PdfDocument
from core_pdf.impl.exceptions import PdfError, PdfPasswordError, PdfUnsupportedError
from core_pdf.impl.types import PdfName, PdfString
from core_pdf_spec.s_07_security.output import ALGORITHMS
from core_pdf_spec.s_07_syntax.stream import PdfStream

ENCRYPTION = "tests/upstream/pypdf/resources/encryption"
# (pattern, password that opens the file)
SOURCES = (
    (f"{ENCRYPTION}/r?-empty-password.pdf", ""),
    (f"{ENCRYPTION}/r?-user-password.pdf", "asdfzxcv"),
    (f"{ENCRYPTION}/r?-aes-user-password.pdf", "asdfzxcv"),
    (f"{ENCRYPTION}/r?-owner-password.pdf", "asdfzxcv"),
    (f"{ENCRYPTION}/r6-both-passwords.pdf", "foo"),
    (f"{ENCRYPTION}/r6-both-passwords.pdf", "bar"),
    (f"{ENCRYPTION}/r4-aes-v2-no-key-length.pdf", ""),
    ("tests/upstream/pypdf/resources/encrypted-file.pdf", "test"),
    ("tests/upstream/pypdf/resources/crazyones-encrypted-256.pdf", "password"),
    ("tests/upstream/pypdf/resources/libreoffice-writer-password.pdf", "openpassword"),
    ("tests/upstream/pypdf/resources/encrypted_doc_no_id.pdf", ""),
    ("tests/upstream/pdfminer.six/samples/encryption/aes-128*.pdf", "foo"),
    ("tests/upstream/pdfminer.six/samples/encryption/aes-256.pdf", "foo"),
    ("tests/upstream/pdfminer.six/samples/encryption/aes-256-m.pdf", "foo"),
    ("tests/upstream/pdfminer.six/samples/encryption/aes-256-r6.pdf", "usersecret"),
    ("tests/upstream/pdfminer.six/samples/encryption/aes-256-r6.pdf", "ownersecret"),
    ("tests/upstream/pdfminer.six/samples/encryption/rc4-*.pdf", "foo"),
    ("tests/upstream/pdfminer.six/samples/encryption/base.pdf", ""),
)
WRONG_PASSWORD = "definitely wrong"
NEW_USER = "new-user"
NEW_OWNER = "new-owner"


def page_texts(document: Any) -> list[str]:
    return ["\n".join(block.text for block in page.extract().blocks) for page in document.pages]


def edited(path: str, password: str) -> tuple[Any, EditableDocument]:
    source = PdfDocument.open(path, password=password)
    editor = EditableDocument(source)
    editor.trailer["Info"] = editor.add_object({"Title": PdfString(b"Edited title")})
    return source, editor


def qpdf_opens(data: bytes, password: str) -> bool | None:
    if shutil.which("qpdf") is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".pdf") as output:
        output.write(data)
        output.flush()
        result = subprocess.run(
            ["qpdf", f"--password={password}", "--check", output.name],
            capture_output=True,
            check=False,
        )
    # 3 means warnings, which the input may have had already.
    return result.returncode in {0, 3}


def check_output(
    label: str,
    data: Any,
    password: str,
    expected: list[str],
    counts: collections.Counter[str],
    *,
    encrypted: bool,
    second_reader: bool = True,
) -> None:
    if data.encrypted != encrypted:
        counts["wrong_statement"] += 1
        print(f"save misreports its encryption: {label}")
    with PdfDocument.open(bytes(data), password=password) as result:
        if (result.security_handler is not None) != encrypted:
            counts["encryption_changed"] += 1
            print(f"output encryption differs: {label}")
        elif page_texts(result) != expected:
            counts["text_mismatch"] += 1
            print(f"text differs: {label}")
        elif result.get_metadata()["info"].get("Title") != "Edited title":
            counts["edit_lost"] += 1
            print(f"edit lost: {label}")
        else:
            counts["identical"] += 1
    if encrypted and b"Edited title" in bytes(data):
        counts["plaintext_leak"] += 1
        print(f"edited string written in the clear: {label}")
    if encrypted and password:
        try:
            PdfDocument.open(bytes(data), password=WRONG_PASSWORD).close()
        except PdfPasswordError:
            counts["wrong_password_refused"] += 1
        else:
            counts["wrong_password_accepted"] += 1
            print(f"wrong password accepted: {label}")
    verdict = qpdf_opens(bytes(data), password) if second_reader else None
    if verdict is not None:
        counts["qpdf_opened" if verdict else "qpdf_refused"] += 1
        if not verdict:
            print(f"qpdf refused: {label}")


def check_file(path: str, password: str, counts: collections.Counter[str]) -> None:
    source, editor = edited(path, password)
    with source:
        expected = page_texts(source)
        counts["files_with_text" if any(expected) else "files_without_text"] += 1
        encrypted = source.security_handler is not None
        label = f"{path} ({password!r})"
        # qpdf refuses r4-aes-v2-no-key-length.pdf itself, and saving keeps that dictionary.
        readable = qpdf_opens(bytes(source.raw_data), password) is not False
        check_output(
            f"{label} keep",
            editor.save(),
            password,
            expected,
            counts,
            encrypted=encrypted,
            second_reader=readable,
        )
        incremental = editor.save_incremental()
        if not incremental.startswith(bytes(source.raw_data)) or not incremental.incremental:
            counts["prefix_changed"] += 1
            print(f"original bytes changed: {label}")
        check_output(
            f"{label} incremental",
            incremental,
            password,
            expected,
            counts,
            encrypted=encrypted,
            second_reader=readable,
        )
        check_output(
            f"{label} keep xref stream",
            editor.save(xref_stream=True),
            password,
            expected,
            counts,
            encrypted=encrypted,
            second_reader=readable,
        )
        check_output(
            f"{label} incremental xref stream",
            editor.save_incremental(xref_stream=True),
            password,
            expected,
            counts,
            encrypted=encrypted,
            second_reader=readable,
        )
        check_output(
            f"{label} decrypt",
            editor.save(encryption="decrypt"),
            "",
            expected,
            counts,
            encrypted=False,
        )
        for algorithm in ALGORITHMS:
            request = NewEncryption(NEW_USER, NEW_OWNER, algorithm)
            data = editor.save(encryption=request, xref_stream=algorithm == "AES-128")
            for new_password in (NEW_USER, NEW_OWNER):
                check_output(
                    f"{label} {algorithm}", data, new_password, expected, counts, encrypted=True
                )
            if data.encryption != algorithm:
                counts["wrong_statement"] += 1
                print(f"save misnames {algorithm} as {data.encryption}: {label}")


def expect_refused(label: str, opener: Any, counts: collections.Counter[str]) -> None:
    try:
        opener()
    except PdfError:
        counts["tampered_refused"] += 1
    else:
        counts["tampered_accepted"] += 1
        print(f"tampered file accepted: {label}")


def check_tampering(counts: collections.Counter[str]) -> None:
    base = f"{ENCRYPTION}/unencrypted.pdf"
    with PdfDocument.open(base) as source:
        for algorithm in ALGORITHMS:
            editor = EditableDocument(source)
            data = bytearray(editor.save(encryption=NewEncryption("user", "owner", algorithm)))
            start = data.index(b"/U <") + 4
            flipped = bytes(data[start:]).translate(bytes.maketrans(b"0123456789", b"1234567890"))
            bad_user = bytes(data[:start]) + flipped[:8] + bytes(data[start + 8 :])
            expect_refused(
                f"{algorithm} /U",
                lambda password=bad_user: PdfDocument.open(password, password="user").close(),
                counts,
            )
            if b"/Perms <" in data:
                start = data.index(b"/Perms <") + 8
                bad_perms = bytes(data[:start]) + b"00" + bytes(data[start + 2 :])
                expect_refused(
                    f"{algorithm} /Perms",
                    lambda password=bad_perms: PdfDocument.open(password, password="user").close(),
                    counts,
                )


def check_refusals(counts: collections.Counter[str]) -> None:
    locked = f"{ENCRYPTION}/r6-both-passwords.pdf"
    for password in ("", WRONG_PASSWORD):
        try:
            EditableDocument(PdfDocument.open(locked, password=password))
        except PdfUnsupportedError:
            counts["unauthenticated_refused"] += 1
        else:
            counts["unauthenticated_accepted"] += 1
            print(f"opened without the password: {password!r}")
    with PdfDocument.open(locked, password="foo") as source:
        editor = EditableDocument(source)
        handler: Any = source.security_handler
        key = handler.file_key
        request = NewEncryption("a secret", "another secret")
        reprs = repr(handler) + repr(source.decipher) + repr(request)
        leaked = key.hex() in reprs or repr(key) in reprs or "secret" in reprs
        counts["repr_leak" if leaked else "repr_clean"] += 1
        if leaked:
            print("key material in a repr")
        del editor


def refused(action: Any) -> bool:
    try:
        action()
    except PdfUnsupportedError:
        return True
    return False


def check_exemptions(counts: collections.Counter[str]) -> None:
    """The parts of a file that encryption leaves alone, and what a save must refuse."""
    with PdfDocument.open(f"{ENCRYPTION}/r6-both-passwords.pdf", password="foo") as source:
        editor = EditableDocument(source)
        signature = bytes.fromhex("3082beef")
        reference = editor.add_object(
            {
                "Type": PdfName.of("Sig"),
                "ByteRange": [0, 1, 2, 3],
                "Contents": PdfString(signature, is_literal=False),
            }
        )
        catalog = editor.edit_object(editor.trailer["Root"])
        catalog["Signature"] = reference
        data = editor.save_incremental()
        with PdfDocument.open(bytes(data), password="foo") as result:
            value: Any = result.resolver.resolve(reference)
            kept = value["Contents"].data == signature and b"<3082BEEF>" in data
        counts["signature_kept" if kept else "signature_changed"] += 1

        encrypt: Any = editor.encrypt_entry
        editor.set_object(encrypt, {})
        blocked = refused(editor.save) and refused(editor.save_incremental)
        del editor.overlay[encrypt.object_number]
        counts["encrypt_edit_refused" if blocked else "encrypt_edit_accepted"] += 1

        editor.trailer["AuthCode"] = {}
        blocked = refused(editor.save) and refused(editor.save_incremental)
        decrypted = editor.save(encryption="decrypt")
        allowed = not decrypted.encrypted and b"/AuthCode" not in decrypted
        counts["mac_refused" if blocked and allowed else "mac_mishandled"] += 1

    with PdfDocument.open(f"{ENCRYPTION}/unencrypted.pdf") as source:
        editor = EditableDocument(source)
        metadata = PdfStream({"Type": PdfName.of("Metadata")}, b"<plain metadata/>")
        reference = editor.add_object(metadata)
        editor.edit_object(editor.trailer["Root"])["Metadata"] = reference
        request = NewEncryption("user", "owner", "AES-128", encrypt_metadata=False)
        data = editor.save(encryption=request)
        with PdfDocument.open(bytes(data), password="user") as result:
            stream: Any = result.resolver.resolve(reference)
            kept = b"<plain metadata/>" in data and bytes(stream.raw_data) == b"<plain metadata/>"
        counts["metadata_kept" if kept else "metadata_changed"] += 1
        plain = EditableDocument(source).save()
        counts["plain_stays_plain" if not plain.encrypted else "plain_encrypted"] += 1


def check_xref_stream_files(counts: collections.Counter[str]) -> None:
    """An update to a file that uses xref streams has to be one, encrypted or not."""
    with PdfDocument.open(f"{ENCRYPTION}/r6-both-passwords.pdf", password="foo") as source:
        saved = EditableDocument(source).save(xref_stream=True)
    with PdfDocument.open(bytes(saved), password="foo") as source:
        editor = EditableDocument(source)
        editor.trailer["Info"] = editor.add_object({"Title": PdfString(b"Edited title")})
        data = editor.save_incremental()
        forced = refused(lambda: editor.save_incremental(xref_stream=False))
        kept = data.xref_stream and data.encrypted and data.startswith(saved) and forced
        counts["xref_stream_continued" if kept else "xref_stream_broken"] += 1
    check_output("stream update", data, "foo", page_texts_of(saved), counts, encrypted=True)


def page_texts_of(data: bytes) -> list[str]:
    with PdfDocument.open(bytes(data), password="foo") as document:
        return page_texts(document)


def main() -> int:
    counts: collections.Counter[str] = collections.Counter()
    for pattern, password in SOURCES:
        paths = sorted(glob.glob(pattern))  # noqa: PTH207
        if not paths:
            print(f"no files for {pattern}")
            counts["missing_fixture"] += 1
        for path in paths:
            try:
                check_file(path, password, counts)
            except PdfError as error:
                counts["errors"] += 1
                print(f"error: {Path(path).name}: {type(error).__name__}: {str(error)[:70]}")
    check_tampering(counts)
    check_refusals(counts)
    check_exemptions(counts)
    check_xref_stream_files(counts)
    print(dict(counts))
    allowed = {
        "files_with_text",
        "files_without_text",
        "identical",
        "wrong_password_refused",
        "qpdf_opened",
        "tampered_refused",
        "unauthenticated_refused",
        "repr_clean",
        "signature_kept",
        "encrypt_edit_refused",
        "mac_refused",
        "metadata_kept",
        "plain_stays_plain",
        "xref_stream_continued",
    }
    return 1 if set(counts) - allowed else 0


if __name__ == "__main__":
    sys.exit(main())
