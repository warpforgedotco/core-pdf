# SPDX-License-Identifier: AGPL-3.0-only

"""Copy-on-write editing over a parsed document, saved as a full rewrite."""

from __future__ import annotations

import re
import secrets
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

from core_pdf.impl.base.exceptions import PdfUnsupportedError
from core_pdf.impl.base.types import PdfReference, PdfString
from core_pdf.impl.document.object_writer import serialize_indirect_object, serialize_object
from core_pdf.impl.recovery.lexer import StrictPdfLexer
from core_pdf.impl.recovery.xref import XRefScanner
from core_pdf.impl.spec.exceptions import PdfParseError
from core_pdf.impl.spec.s_07_security.output import (
    NewEncryption,
    create_security_handler,
    describe_encryption,
    encrypt_object,
    encryption_dictionary,
)
from core_pdf.impl.spec.s_07_security.standard import StandardSecurityHandler
from core_pdf.impl.spec.s_07_syntax.serialize import ObjectLayout, serialize_xref_stream
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.spec.s_07_syntax_primitives.tokens import LexicalRules
lazy from core_pdf_cythonized import collect_references

if TYPE_CHECKING:
    from core_pdf.impl.document.pdf_document import PdfDocument

BINARY_MARKER = b"%\xe2\xe3\xcf\xd3\n"


def file_header(version_line: bytes) -> bytes:
    """The start of a file: its %PDF- version line, then a comment of high bytes (7.5.2)."""
    return version_line + b"\n" + BINARY_MARKER


TRAILER_KEYS = ("Root", "Info", "ID")
# 7.6: AES needs PDF 1.6 and AES-256 needs 1.7 with an extension or 2.0.
ENCRYPTED_HEADER_VERSION = b"1.7"
# 7.5.8: xref streams arrived in PDF 1.5.
XREF_STREAM_HEADER_VERSION = b"1.5"
# A stale startxref points at neither kind of section, and counts as a table.
XREF_STREAM_START = re.compile(rb"[\x00\t\n\x0c\r ]*\d+[\x00\t\n\x0c\r ]+\d+[\x00\t\n\x0c\r ]+obj")


class SavedPdf(bytes):
    """The bytes of a save, which also state how they were made.

    `incremental` is False for a full rewrite, which drops old revisions, and True for an
    append, which keeps them. `encryption` names the output's algorithm, or is None when the
    output is not encrypted, as in `encrypted`. `xref_stream` is True when the section this
    save wrote is an xref stream rather than a classic table. `removed` lists the objects the
    source reached and the saved document no longer does, which an incremental update leaves
    readable in the earlier revision. A full rewrite leaves them out, so it lists none.
    """

    incremental: bool
    encryption: str | None
    xref_stream: bool
    removed: tuple[int, ...]

    def __new__(
        cls,
        data: bytes,
        *,
        incremental: bool,
        encryption: str | None,
        xref_stream: bool,
        removed: tuple[int, ...] = (),
    ) -> Self:
        saved = super().__new__(cls, data)
        saved.incremental = incremental
        saved.encryption = encryption
        saved.xref_stream = xref_stream
        saved.removed = removed
        return saved

    @property
    def encrypted(self) -> bool:
        return self.encryption is not None


def mutable_copy(value: Any) -> Any:
    if isinstance(value, PdfStream):
        return value.replace(dictionary=mutable_copy(dict(value.dictionary)))
    if isinstance(value, dict):
        return {key: mutable_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [mutable_copy(item) for item in value]
    return value


def scan_difference(
    first: LexicalRules, second: LexicalRules, known: dict[int, re.Pattern[bytes] | bool]
) -> re.Pattern[bytes] | bool:
    """Where the object scanner may read differently under two lexical rules.

    True when it reads alike everywhere, False when it may read differently anywhere, or a
    pattern of the bytes its two tables classify differently: bytes it reads alike unless it
    meets one of those.
    """
    if id(first) in known:
        return known[id(first)]
    difference: re.Pattern[bytes] | bool
    if first is second:
        difference = True
    elif (
        first.name_escapes != second.name_escapes
        or first.split_whitespace_compatible != second.split_whitespace_compatible
    ):
        difference = False
    else:
        differing = bytes(
            index
            for index in range(256)
            if first.whitespace_table[index] != second.whitespace_table[index]
            or first.separator_table[index] != second.separator_table[index]
        )
        difference = re.compile(b"[" + re.escape(differing) + b"]") if differing else True
    known[id(first)] = difference
    return difference


def references_in(value: Any) -> list[PdfReference]:
    """Every reference in value, depth first; the order decides the order of a save."""
    return collect_references(value, PdfReference, PdfStream)


class EditableDocument:
    def __init__(self, document: PdfDocument) -> None:
        self.security = document.security_handler
        encrypt_entry = document.trailer_dict.get("Encrypt")
        if encrypt_entry is not None and self.security is None:
            raise PdfUnsupportedError("editing an encrypted document needs its password")
        self.document = document
        self.trailer: dict[Any, Any] = mutable_copy(dict(document.trailer_dict))
        self.overlay: dict[int, Any] = {}
        # Neither of these is encrypted, so neither may go through the object table.
        self.encrypt_entry = mutable_copy(encrypt_entry) if self.security is not None else None
        if self.security is not None and isinstance(self.trailer.get("ID"), PdfReference):
            self.trailer["ID"] = mutable_copy(document.resolver.resolve(self.trailer["ID"]))
        # What an incremental update would repeat; with nothing else changed, it adds nothing.
        self.source_trailer = {
            key: mutable_copy(self.trailer[key]) for key in TRAILER_KEYS if key in self.trailer
        }
        # /Size is often wrong, so a new number must also clear every object in use.
        size = document.resolver.resolve(self.trailer.get("Size"))
        declared = size if isinstance(size, int) and size > 0 else 1
        # The walk below reads every reachable object; a save can then reuse those readings.
        record_strict_reads = getattr(document.resolver, "record_strict_reads", None)
        if record_strict_reads is not None:
            record_strict_reads()
        numbers = set(self.reachable_references())
        self.source_reachable = frozenset(numbers)
        if isinstance(self.encrypt_entry, PdfReference):
            numbers.add(self.encrypt_entry.object_number)
        self.next_object_number = max(declared, max(numbers, default=0) + 1)

    def get_object(self, reference: PdfReference) -> Any:
        if reference.object_number in self.overlay:
            return self.overlay[reference.object_number]
        return self.document.resolver.resolve(reference)

    def edit_object(self, reference: PdfReference) -> Any:
        """Return a copy of the object that can be changed in place and will be saved."""
        number = reference.object_number
        if number not in self.overlay:
            self.overlay[number] = mutable_copy(self.document.resolver.resolve(reference))
        return self.overlay[number]

    def set_object(self, reference: PdfReference, value: Any) -> None:
        self.overlay[reference.object_number] = value

    def add_object(self, value: Any) -> PdfReference:
        reference = PdfReference(self.next_object_number, 0)
        self.next_object_number += 1
        self.overlay[reference.object_number] = value
        return reference

    def reachable_references(self) -> dict[int, PdfReference]:
        reachable: dict[int, PdfReference] = {}
        pending = references_in({key: self.trailer.get(key) for key in TRAILER_KEYS})
        while pending:
            reference = pending.pop()
            if reference.object_number in reachable:
                continue
            reachable[reference.object_number] = reference
            pending.extend(references_in(self.get_object(reference)))
        return reachable

    def save(
        self,
        target: str | Path | None = None,
        *,
        encryption: Literal["keep", "decrypt"] | NewEncryption = "keep",
        xref_stream: bool = False,
    ) -> SavedPdf:
        """Write every reachable object once, dropping old revisions and unused objects.

        An encrypted input stays encrypted with the same key and passwords. Pass "decrypt" for
        a plain file, or a NewEncryption for new passwords and a new key. See the policy in
        docs/object-model.md. `xref_stream` writes an xref stream in place of the classic table.
        """
        reachable = self.reachable_references()
        objects = {reference: self.get_object(reference) for reference in reachable.values()}
        trailer = {key: self.trailer[key] for key in TRAILER_KEYS if key in self.trailer}
        security = None
        exempt: set[int] = set()
        if isinstance(encryption, NewEncryption):
            document_id = first_identifier(trailer)
            if document_id is None:
                document_id = secrets.token_bytes(16)
                identifier = PdfString(document_id, is_literal=False)
                trailer["ID"] = [identifier, identifier]
            security = create_security_handler(encryption, document_id)
            reference = PdfReference(self.next_object_number, 0)
            objects[reference] = encryption_dictionary(security.config)
            exempt.add(reference.object_number)
            trailer["Encrypt"] = reference
        elif encryption == "keep" and self.security is not None:
            security = self.sealing_handler()
            trailer["Encrypt"] = self.encrypt_entry
            if isinstance(self.encrypt_entry, PdfReference):
                objects[self.encrypt_entry] = self.document.resolver.resolve(self.encrypt_entry)
                exempt.add(self.encrypt_entry.object_number)
        elif encryption not in {"keep", "decrypt"}:
            raise ValueError("encryption must be 'keep', 'decrypt' or a NewEncryption")
        keeps_bytes = encryption == "keep" or self.security is None and encryption == "decrypt"
        verbatim = self.verbatim_objects(objects) if keeps_bytes else {}
        minimum = ENCRYPTED_HEADER_VERSION if security is not None else b"1.0"
        if xref_stream:
            minimum = max(minimum, XREF_STREAM_HEADER_VERSION)
        header = output_header(self.document.raw_data[:8], minimum)
        output = bytearray(file_header(header))
        output += serialize_body(
            len(output),
            objects,
            trailer,
            xref_stream=xref_stream,
            security=security,
            exempt=exempt,
            verbatim=verbatim,
        )
        return write_output(
            bytes(output), target, incremental=False, security=security, xref_stream=xref_stream
        )

    def save_incremental(
        self,
        target: str | Path | None = None,
        *,
        xref_stream: bool | None = None,
        layout: ObjectLayout = ObjectLayout.COMPACT,
        keep_encryption: bool = True,
    ) -> SavedPdf:
        """Append changed objects after the original bytes, which stay identical.

        Earlier revisions remain readable in the file, so this is not a way to remove data:
        objects the document no longer reaches are listed in `SavedPdf.removed`. An
        encrypted input keeps its encryption: new objects use the same key, and this raises if
        that cannot be done.

        By default the update uses the kind of section the file ends with. `xref_stream=True`
        asks for a stream either way. `xref_stream=False` raises on a file that uses xref
        streams, because a file indexed by them has to be continued by one (7.5.8.4).

        `layout=ObjectLayout.SPACED` writes the update as pypdf does, from the objects to the
        xref stream, so the pypdf facade's output matches pypdf's byte for byte.

        `keep_encryption=False` writes an encrypted input's update as pypdf does: its objects in
        the clear and no /Encrypt in the new trailer. Readers then treat the file as plain, so
        the earlier objects no longer decrypt. Only the pypdf facade asks for this.
        """
        original = bytes(self.document.raw_data)
        previous = XRefScanner.find_startxref(original)
        if previous is None:
            raise PdfUnsupportedError("an incremental update needs a readable startxref")
        continues_stream = XREF_STREAM_START.match(original, previous) is not None
        if xref_stream is False and continues_stream:
            raise PdfUnsupportedError("a file that uses xref streams must be updated by one")
        use_stream = continues_stream if xref_stream is None else xref_stream
        security = None
        if self.security is not None and keep_encryption:
            security = self.sealing_handler()
        output = bytearray(original)
        # pypdf's layout appends right after the last byte, as pypdf does.
        if not output.endswith(b"\n") and layout is not ObjectLayout.SPACED:
            output += b"\n"
        offsets: dict[int, tuple[int, int]] = {}
        reachable = self.reachable_references()
        # The old sections of a damaged file do not read, so a stream replaces them. Recovery
        # would otherwise prefer a literal trailer to it, and lose the update.
        # pypdf's layout continues the old sections as pypdf does, and recovery reads that too.
        standalone = (
            use_stream and self.document.xref_was_recovered and layout is not ObjectLayout.SPACED
        )
        changed = (
            {**self.mislabelled_streams(reachable), **self.overlay} if standalone else self.overlay
        )
        trailer: dict[Any, Any] = {
            key: self.trailer[key] for key in TRAILER_KEYS if key in self.trailer
        }
        if not changed and not standalone and trailer == self.source_trailer:
            # Nothing to append, so the file stays as it is, encryption included, as in pypdf.
            # An xref table with no entries would not read in other readers.
            return write_output(
                original,
                target,
                incremental=True,
                security=self.security,
                xref_stream=continues_stream,
                removed=tuple(sorted(self.source_reachable - reachable.keys())),
            )
        for number in sorted(changed):
            existing = reachable.get(number)
            generation = 0 if existing is None else existing.generation_number
            offsets[number] = (len(output), generation)
            reference = PdfReference(number, generation)
            value = changed[number]
            sealed = (
                value
                if security is None
                else encrypt_object(security, reference, value, self.get_object)
            )
            output += serialize_indirect_object(reference, sealed, layout)
        if not standalone:
            trailer["Prev"] = previous
        if security is not None:
            trailer["Encrypt"] = self.encrypt_entry
        section_offset = len(output)
        size = max(self.next_object_number, max(offsets, default=0) + 1)
        if standalone:
            kept, compressed = self.existing_entries()
            size = max(size, *(number + 1 for number in (*kept, *compressed)))
            output += serialize_xref_stream(
                size,
                {**kept, **offsets},
                trailer,
                section_offset,
                compressed={key: value for key, value in compressed.items() if key not in offsets},
                complete=True,
            ).rstrip()
        elif use_stream:
            # The stream takes the next free number, past every object in the file.
            output += serialize_xref_stream(
                size, offsets, trailer, section_offset, layout=layout
            ).rstrip()
        else:
            trailer["Size"] = size
            output += table_section(offsets) + b"trailer\n" + serialize_object(trailer)
        output += b"\nstartxref\n%d\n%%%%EOF\n" % section_offset
        return write_output(
            bytes(output),
            target,
            incremental=True,
            security=security,
            xref_stream=use_stream,
            removed=tuple(sorted(self.source_reachable - reachable.keys())),
        )

    def verbatim_objects(self, objects: dict[PdfReference, Any]) -> dict[int, bytes]:
        """Find the file's own bytes for each object nobody changed, adding to `objects` what
        those bytes refer to.

        An object inside an object stream has no bytes of its own in the file, so it isn't
        included. Nor is a stream with a wrong /Length, which only recovery can read.
        """
        offsets, _ = self.existing_entries()
        resolver = self.document.resolver
        # The strict lexer decides which objects keep their own bytes; its compiled
        # reading of arrays and dictionaries returns what its own code would.
        lexer = StrictPdfLexer(self.document.raw_data, reference_resolver=resolver.resolve)
        # An object the resolver's scanner already read as the strict scanner would, under the
        # same lexical tables, has the strict reading and end the lexer would find again.
        strict_reads = getattr(resolver, "strict_reads", None) or {}
        differences: dict[int, re.Pattern[bytes] | bool] = {}
        found: dict[int, bytes] = {}
        pending = list(objects)
        while pending:
            reference = pending.pop()
            number = reference.object_number
            located = offsets.get(number)
            if number in self.overlay or located is None:
                continue
            if located[1] != reference.generation_number:
                continue
            read = strict_reads.get(located[0])
            reusable = False
            if read is not None and read[0] == number and read[1] == located[1]:
                difference = scan_difference(read[4], lexer.lexical_rules, differences)
                # The scanner reads from the header to the byte after endobj.
                reusable = difference is True or (
                    isinstance(difference, re.Pattern)
                    and difference.search(self.document.raw_data, located[0], read[2] + 1) is None
                )
            if read is not None and reusable:
                parsed, end = read[3], read[2]
            else:
                lexer.rewind(located[0])
                try:
                    parsed = lexer.parse_indirect_object(expected_reference=reference)
                except PdfParseError:
                    continue
                end = lexer.pos
            if isinstance(parsed, PdfStream):
                length = resolver.resolve(parsed.dictionary.get("Length"))
                if length != len(parsed.raw_data):
                    continue
            found[number] = bytes(self.document.raw_data[located[0] : end])
            # The resolver inlines some stream entries, so the kept bytes can refer to more
            # than the walk from the trailer saw.
            for target in references_in(parsed):
                if target not in objects:
                    objects[target] = self.get_object(target)
                    pending.append(target)
        return found

    def mislabelled_streams(self, reachable: dict[int, PdfReference]) -> dict[int, Any]:
        """Find streams whose /Length is wrong, which only recovery can read as they stand."""
        found = {}
        for number, reference in reachable.items():
            value = self.get_object(reference)
            if isinstance(value, PdfStream):
                length = self.document.resolver.resolve(value.dictionary.get("Length"))
                if length != len(value.raw_data):
                    found[number] = value
        return found

    def existing_entries(self) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
        """Split the document's xref into objects by offset and objects in object streams."""
        offsets: dict[int, tuple[int, int]] = {}
        compressed: dict[int, tuple[int, int]] = {}
        # Sorted so that a number's newest generation wins, as one number has one entry here.
        for key, entry in sorted(self.document.xref.items()):
            if not entry.in_use:
                continue
            number = key >> 16
            if entry.object_stream is None:
                offsets[number] = (entry.offset, entry.generation)
                compressed.pop(number, None)
            else:
                compressed[number] = (entry.object_stream, entry.index_in_stream or 0)
                offsets.pop(number, None)
        return offsets, compressed

    def sealing_handler(self) -> StandardSecurityHandler:
        """Return the input's handler, or raise if its encryption cannot be carried over."""
        security = self.security
        assert security is not None
        if "AuthCode" in self.trailer or security.config.pdf_mac_required:
            raise PdfUnsupportedError(
                "a PDF MAC cannot be rebuilt; save with encryption='decrypt' or new encryption"
            )
        entry = self.encrypt_entry
        if isinstance(entry, PdfReference) and entry.object_number in self.overlay:
            raise PdfUnsupportedError("the encryption dictionary cannot be edited")
        return security


def first_identifier(trailer: dict[Any, Any]) -> bytes | None:
    identifiers = trailer.get("ID")
    first = identifiers[0] if isinstance(identifiers, list) and identifiers else None
    return first.data if isinstance(first, PdfString) and first.data else None


def output_header(start: bytes | memoryview, minimum: bytes) -> bytes:
    header = bytes(start) if bytes(start).startswith(b"%PDF-") else b"%PDF-1.7"
    return header if header[5:] >= minimum else b"%PDF-" + minimum


def serialize_body(
    start: int,
    objects: dict[PdfReference, Any],
    trailer: dict[Any, Any],
    layout: ObjectLayout = ObjectLayout.COMPACT,
    *,
    xref_stream: bool = False,
    security: StandardSecurityHandler | None = None,
    exempt: set[int] | frozenset[int] = frozenset(),
    verbatim: dict[int, bytes] | None = None,
) -> bytes:
    """Write objects and an xref section with the trailer, starting at file offset `start`.

    The section is a classic table, or an xref stream if `xref_stream` is set. With `security`,
    every object is encrypted except those numbered in `exempt`. An object numbered in
    `verbatim` is written as those bytes, which are already in their final form.
    """
    output = bytearray()
    offsets: dict[int, tuple[int, int]] = {}
    for reference in sorted(objects, key=lambda item: item.object_number):
        offsets[reference.object_number] = (start + len(output), reference.generation_number)
        kept = None if verbatim is None else verbatim.get(reference.object_number)
        if kept is not None:
            output += kept + b"\n"
            continue
        value = objects[reference]
        if security is not None and reference.object_number not in exempt:
            value = encrypt_object(security, reference, value, objects.get)
        output += serialize_indirect_object(reference, value, layout)
    size = max(offsets, default=0) + 1
    xref_offset = start + len(output)
    if xref_stream:
        output += serialize_xref_stream(size, offsets, trailer, xref_offset, complete=True).rstrip()
        output += b"\nstartxref\n%d\n%%%%EOF\n" % xref_offset
        return bytes(output)
    output += b"xref\n0 %d\n0000000000 65535 f \n" % size
    for number in range(1, size):
        offset, generation = offsets.get(number, (0, 65535))
        marker = b"n" if number in offsets else b"f"
        output += b"%010d %05d %s \n" % (offset, generation, marker)
    output += b"trailer\n" + serialize_object({"Size": size, **trailer}, layout)
    output += b"\nstartxref\n%d\n%%%%EOF\n" % xref_offset
    return bytes(output)


def write_output(
    data: bytes,
    target: str | Path | None,
    *,
    incremental: bool,
    security: StandardSecurityHandler | None,
    xref_stream: bool,
    removed: tuple[int, ...] = (),
) -> SavedPdf:
    if target is not None:
        Path(target).write_bytes(data)
    encryption = None if security is None else describe_encryption(security.config)
    return SavedPdf(
        data,
        incremental=incremental,
        encryption=encryption,
        xref_stream=xref_stream,
        removed=removed,
    )


def table_section(offsets: dict[int, tuple[int, int]]) -> bytes:
    section = bytearray(b"xref\n")
    numbers = sorted(offsets)
    start = 0
    while start < len(numbers):
        end = start
        while end + 1 < len(numbers) and numbers[end + 1] == numbers[end] + 1:
            end += 1
        section += b"%d %d\n" % (numbers[start], end - start + 1)
        for number in numbers[start : end + 1]:
            offset, generation = offsets[number]
            section += b"%010d %05d n \n" % (offset, generation)
        start = end + 1
    return bytes(section)


__all__ = (
    "BINARY_MARKER",
    "EditableDocument",
    "NewEncryption",
    "SavedPdf",
    "file_header",
    "serialize_body",
)
