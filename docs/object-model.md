# Editable object model and writer

This is the design for lossless editing of PDFs in core-pdf (roadmap item 5). It is also
what pypdf's `PdfWriter` and `generic` types, and any future pikepdf facade, are built on.

## Why this was built

`pypdf`'s upstream suite stopped here: `PdfWriter` and the object types were missing, so
most of its tests could not run. The milestones below are finished, and the suite now runs
in CI.

## Requirements

1. **Lossless:** an object nobody changed is written back byte for byte, so unknown keys and
   producer quirks survive. `EditableDocument.save` copies each object's bytes from the file,
   and `save_incremental` keeps the whole file, so signatures survive it. Three cases are not
   byte for byte: an object inside an object stream, which has no bytes of its own; a stream
   with a wrong `/Length`, which is rewritten so that it reads without recovery; and output
   with changed encryption. `PdfWriter` rewrites follow pypdf's numbering and layout, so they
   are equal by value, and `PdfWriter(incremental=True)` is the byte-for-byte route. A full
   save moves objects, so it invalidates signatures even when their bytes are kept.
2. **Explicit:** an edit is visible in the object it touches. There is no hidden state
   shared between a reader and a writer.
3. **Layered:** spec holds the syntax, core holds the document, compat holds the pypdf
   and pikepdf shapes. Compat never writes bytes itself.
4. **Secure by default:** saving never silently keeps data the caller believes was
   removed, so a save states whether it was incremental or a full rewrite.

## Layers

| Layer | Package | Adds |
| ----- | ------- | ---- |
| Serializer | `core-pdf-spec` | PDF object to bytes, the inverse of the lexer |
| Editable document | `core-pdf` | An object table over the resolver, with copy-on-write |
| Writers | `core-pdf` | Full rewrite and incremental update, xref tables and streams |
| Object types | `core-pdf-compat` | `NameObject`, `DictionaryObject` and friends, over core objects |
| `PdfWriter` | `core-pdf-compat` | pypdf's page, annotation and merge API |

### Editable document

`EditableDocument` wraps a `PdfDocument`. Reading an object goes through the resolver. The
first write to an object copies it into an overlay keyed by object number, and new objects
take numbers past the highest in the file. `save` walks from the trailer, so unreferenced
objects and old revisions are dropped. `save_incremental` appends changed objects after the
original bytes, so signatures survive, but earlier data stays in the file. Use `save` when
removing data.

### Cross-reference sections

Writers use a classic xref table unless told otherwise. `save(xref_stream=True)` writes an xref
stream instead, covering every number from 0. `save_incremental` follows the file: an update to
a file that ends in an xref stream must use one (ISO 32000-2 7.5.8.4), so it does, and
`xref_stream=False` is refused there. `xref_stream=True` asks for a stream on a file with a
classic table too. On a damaged file, whose old sections do not read, the update's stream lists
every object and has no /Prev, so the output is readable without recovery. Streams are stored without a filter, and their field widths fit the file.
Object streams are not written. `SavedPdf.xref_stream` states which kind a save produced.

### PdfWriter

The pypdf facade works over its own table of generic objects. A full rewrite writes what the
trailer reaches, so a removed page or image leaves the file. `PdfWriter(incremental=True)`
hands changed objects to `EditableDocument.save_incremental`, so it adds no second way of
writing bytes.

`write` returns what pypdf returns, so the writer states how a save was made in
`writer.last_save`, a `SaveReport` with the kind (`"full"` or `"incremental"`), the objects
written, the earlier versions left readable (`superseded`) and the objects removed but still
in the file (`removed`). An incremental save with removals also logs a warning.

`scripts/check_writer_lossless.py` compares unchanged objects in writer output with the
source. A full rewrite keeps values and stream bytes but renumbers objects and normalizes
syntax, so byte-for-byte output needs an incremental update.

### Encrypted input

Encrypted files can be edited under this policy. Its rule is that encryption never goes away
by accident: nobody loses protection without asking for it, and each save says what it did.

| Input | Call | Output |
| ----- | ---- | ------ |
| Not opened with its password | `EditableDocument(document)` | Refused |
| Encrypted | `save()` | Encrypted with the same key and passwords |
| Encrypted | `save(encryption="decrypt")` | Plain |
| Any | `save(encryption=NewEncryption(...))` | Encrypted with a new key and passwords |
| Encrypted | `save_incremental()` | Original bytes kept, new objects encrypted with the same key |
| Plain | `save()` or `save_incremental()` | Plain |

- **Opening needs the right password.** `PdfDocument.open` refuses a wrong one, and
  `EditableDocument` refuses a document that has no handler. A file whose user password is
  empty opens without being given one, because the empty password is its correct password.
- **Saving keeps what it found.** Decrypting a file is a request, spelled out at the call,
  never a default or a side effect of editing. So is a weaker algorithm:
  `NewEncryption` does what it is told, and the choice is visible in that call.
- **New encryption defaults to AES-256.** `NewEncryption` also takes the older `RC4-40`,
  `RC4-128`, `AES-128` and `AES-256-R5`. RC4 is broken and exists only for old readers. With no
  owner password, the user password serves as both, so it also grants owner access. An empty
  user password encrypts nothing in practice, because any reader can open the file.
- **An incremental save cannot change encryption.** The original bytes stay, so a decrypted
  or re-keyed update would be a false promise. `save_incremental` takes no such option.
- **A save states how it was made.** `save` and `save_incremental` return `SavedPdf`, which is
  `bytes` with `incremental`, `encrypted` and `encryption` (the algorithm's name, or `None`).
- **Some saves are refused.** A file with a PDF MAC (`/AuthCode`, or /P demanding one) keeps
  its encryption only if the MAC is rebuilt, and core-pdf does not make MACs, so `save()` and
  `save_incremental()` raise `PdfUnsupportedError`. `encryption="decrypt"` or `NewEncryption`
  still work, and the output has no MAC. Editing the encryption dictionary is refused too.
- **Passwords and keys are not kept in the open.** `PdfDocument` does not hold the password.
  The file key stays in the handler, as reading needs it, and is left out of `repr`, as are
  the passwords in `NewEncryption`.

The pypdf facade follows pypdf rather than this table, because that is its contract: a
`PdfWriter` starts empty, so `clone_from` an encrypted reader gives a plain file until
`encrypt` is called. `encrypt` keeps only the key. A reader stays locked on a file whose /Perms
is wrong and reports `are_permissions_valid` as False, where pypdf warns and reads on.

Limits worth knowing:

- Permission bits (/P) are carried over, not enforced. Anyone with either password has the
  file key, so they are a request to readers, not access control. A user-password holder can
  therefore save a decrypted copy, which is what every other PDF tool allows too.
- Keeping encryption keeps the key. Old copies of the file open with the old password, and
  `NewEncryption` is the way to retire one.
- Under RC4, an object changed in place is encrypted with the same keystream as its old
  version, so anyone with both can compare them. PDF defines the key per object number, and an
  incremental update must reuse it. Use AES, or a `NewEncryption` full rewrite, for such edits.
- A full rewrite decrypts and re-encrypts. Under AES each string and stream gets a new IV, so
  ciphertext changes although content does not. `save_incremental` keeps it byte for byte.
- CBC and RC4 do not detect tampering with content. A wrong /Perms and a damaged AES padding
  or GCM tag are refused; a flipped RC4 byte is not detectable by anyone.
- Public-key (`Adobe.PubSec`) encryption is not supported, for reading or writing.

## Milestones

Each milestone is one or more reviewable PRs. Counts come from the upstream harness.

All six are done. Numbers are from main after #238.

1. **Serializer. Done.** Round-trips every object in the fixture corpus through the lexer.
   `scripts/check_object_roundtrip.py`: 64,649 objects with no mismatches, and 6 unreadable.
2. **Editable document and full rewrite. Done.** Open, change, save, reopen, and compare.
   `scripts/check_document_rewrite.py`: 145 identical and no page mismatches. 30,728 of the
   30,744 objects with bytes of their own came through a rewrite unchanged, and none was lost.
3. **Incremental update. Done.** Untouched bytes stay identical, checked by prefix
   comparison. The same script with `--incremental`: 142 identical and no page mismatches.
4. **pypdf `generic` types. Done.** No public `pypdf.generic` name is a stand-in, and it is
   no longer partial in `tests/compat/out_of_scope.toml`. Its private submodules stay stubs.
5. **`PdfWriter`. Done.** Pages, metadata, annotations, forms and merging. No in-scope test
   skips for a missing writer feature. Incremental mode is built on `save_incremental`.
   `scripts/check_writer_lossless.py`: 60,901 objects equal by value through a rewrite, and
   all 167 incremental updates start with the source's bytes.
6. **Encrypted output and xref streams. Done.** The policy is under
   [Encrypted input](#encrypted-input). qpdf, Poppler, Ghostscript and real pypdf opened all
   15 `PdfWriter.encrypt` outputs. `--xref-stream` gives the same results on both scripts:
   64,644 objects with no mismatches, 145 identical, and 142 with `--incremental`.

### Upstream suites

| Suite | Passed | Skipped |
| ----- | ------ | ------- |
| pypdf | 753 of 961 | 208 |
| pdfplumber | 169 | 0 |
| pdfminer | 127 | 0 |
| unstructured | 50 | 0 |
| llama_index | 27 | 0 |
| x-ray | 14 | 1 |

Real pypdf passes 959 of the same 961 and skips 2. None fail on the facade. Of its 208
skips, 206 reach a private pypdf module and 2 are also skipped on the real library, as
`scripts/audit_upstream_skips.py pypdf` reports.

### Known gaps

- Tests marked `enable_socket` download their files, so the harness leaves them out.
- R7 (AES-GCM) keep-encryption has no real fixture, and PDF MAC refusal was tested with an
  injected `/AuthCode`, not a real MAC file.
- Ghostscript refused AES-256-R5 with a four-character password, including on real pypdf's
  own file. qpdf, Poppler and pypdf open those files, and a seven-character password works.

## Out of scope for now

Linearization, object stream compression on write, and a pikepdf facade. Those can follow
once `PdfWriter` is stable.
