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
   with changed encryption. A full save moves objects, so it invalidates signatures even when
   their bytes are kept. The pypdf facade's `PdfWriter` is held to a different promise: its
   output is pypdf's, byte for byte, including where pypdf changes data (see PdfWriter).
2. **Explicit:** an edit is visible in the object it touches. There is no hidden state
   shared between a reader and a writer.
3. **Layered:** spec holds the syntax, core holds the document, compat holds the pypdf
   and pikepdf shapes. Compat never writes bytes itself.
4. **Secure by default:** in core, saving never silently keeps data the caller believes was
   removed, so a save states whether it was incremental or a full rewrite. The pypdf facade
   writes what pypdf writes, which can keep removed objects, and says so in `last_save`.

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

The pypdf facade works over its own table of generic objects, and its job is to write exactly
what pypdf writes, faster. A full rewrite writes every object the writer holds, reached or not,
as pypdf does, so a page removed through the writer stays in the file. `PdfWriter(incremental=True)`
hands changed objects to `EditableDocument.save_incremental`, so it adds no second way of
writing bytes. Use `EditableDocument.save` to remove data.

`write` returns what pypdf returns, so the writer states how a save was made in
`writer.last_save`, a `SaveReport` with the kind (`"full"` or `"incremental"`), the objects
written, the earlier versions left readable (`superseded`) and the objects the document no
longer reaches that are still in the file (`removed`). A save with any of those logs a warning.

`scripts/check_save_statements.py` checks these statements on the upstream PDFs. It adds a
page with a marker in its content, an image, an annotation and an embedded font, removes the
page and saves. On all 168 files that open unencrypted, `EditableDocument.save` drops the
marker and says it is full. A `PdfWriter` full rewrite keeps it, as pypdf does, names the kept
objects in `last_save` and logs a warning. An incremental update keeps the marker and the
source's bytes, says it is
incremental and names the six objects in `SavedPdf.removed`; `PdfWriter` also names them in
`last_save` and logs a warning. An encrypted `PdfWriter` save hides the
marker and names its algorithm in `SavedPdf.encryption`. The check also fails if a writer
module holds PDF structure as bytes.

`scripts/check_pypdf_bytes.py` writes each upstream PDF through the real pypdf 7.0.0 and the
facade the same way and compares the bytes: an incremental update with a new title, a
`clone_from` and an `append`. All three are identical on all 173 files pypdf can write, and CI
fails on any difference.

`scripts/check_pypdf_reads.py` reads every object of those files through both libraries, as
written and encrypted by pypdf with RC4-128 and AES-256, and compares each object's written
bytes, each stream's decoded data and each error's class: 189,444 reads, all identical, and CI
fails on any difference.

That means following pypdf where it changes or loses data, and the facade does:

- Reals are rounded to pypdf's precision, and a name read from GBK or Latin-1 bytes is
  written as UTF-8.
- `/Info` is copied shallowly, so an indirect value keeps the source's number and points at
  nothing in the output.
- An object pypdf's parser refuses is read with its rules: a bare word such as `OBJR`, or a
  name running into a NUL byte. pypdf then keeps only the entries read before it.
- A page a link or field reaches is copied, even if it is not one of the pages added.
- A signature's `/Contents` is decrypted on reading, though PDF leaves it in the clear.
- An incremental update to an encrypted file is written in the clear, without /Encrypt.

`scripts/check_writer_lossless.py` reads both sides as pypdf does and compares unchanged
objects in writer output with the
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
| Encrypted | `save_incremental(keep_encryption=False)` | Update in the clear, no /Encrypt: pypdf's output |
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
- **An incremental save cannot re-key.** The original bytes stay, so a decrypted or re-keyed
  update would be a false promise. The one exception is `keep_encryption=False`, which only
  the pypdf facade passes, to write what pypdf writes: the update in the clear and no
  /Encrypt in its trailer, so readers treat the file as plain and the earlier objects no
  longer decrypt.
- **A save states how it was made.** `save` and `save_incremental` return `SavedPdf`, which is
  `bytes` with `incremental`, `encrypted`, `encryption` (the algorithm's name, or `None`) and
  `removed`, the objects the source reached and the save no longer does. A full rewrite leaves
  them out, so it lists none; an incremental update leaves them readable, so it names them.
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

All six are done. Numbers are from main, and CI reruns each of these checks on every pull request.

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
   `scripts/check_writer_lossless.py`: 59,971 objects equal by value through a rewrite, and
   all 168 incremental updates start with the source's bytes.
6. **Encrypted output and xref streams. Done.** The policy is under
   [Encrypted input](#encrypted-input). qpdf, Poppler, Ghostscript and real pypdf opened all
   15 `PdfWriter.encrypt` outputs. `--xref-stream` gives the same results on both scripts:
   64,644 objects with no mismatches, 145 identical, and 142 with `--incremental`.

### Upstream suites

| Suite | Passed | Skipped |
| ----- | ------ | ------- |
| pypdf | 1,829 of 2,294 | 465 |
| pdfplumber | 169 | 0 |
| pdfminer | 127 | 0 |
| unstructured | 83 | 0 |
| llama_index | 27 | 0 |
| x-ray | 14 | 1 |

Counts are from CI's Linux runners; on macOS two more pypdf tests skip. Real pypdf 7.0.0
passes 2,303 of 2,310 and skips 7. None fail on the facade. Of its 465 skips, 458 reach a
private pypdf module and 7 are also skipped on the real library, as
`scripts/audit_upstream_skips.py pypdf` reports. The count includes pypdf's tests that
download their files (marked `enable_socket`); the harness fetches those files first, and CI
caches them.

### Upstream versions

The facades support the latest release of each library they follow. The upstream suites and
the comparison scripts pin those versions, and a weekly workflow keeps an issue open while
any pin is behind its latest release on PyPI.

### Facade output against the real libraries

Beside the upstream suites, CI compares each facade with its real library on the sample
PDFs, file by file, and fails on any difference a script does not list with its reason:

| Script | Library | Result |
| ------ | ------- | ------ |
| `check_pypdf_bytes.py` | pypdf 7.0.0 | Written bytes identical on every file pypdf writes, in each mode |
| `check_pypdf_reads.py` | pypdf 7.0.0 | Every object read identical, plain, RC4 and AES |
| `check_pypdf_text.py` | pypdf 7.0.0 | `extract_text` identical on all 3,890 pages compared, plain and layout |
| `check_pdfminer_output.py` | pdfminer.six 20260107 | Text identical on 178 of 184 files, layout on 179 |
| `check_pdfplumber_output.py` | pdfplumber 0.11.10 | Identical apart from 6 listed files |
| `check_unstructured_output.py` | unstructured 0.27.36 | 215 of 216 files identical |
| `check_llamaindex_output.py` | llama-index-readers-file 0.7.0, pypdf 6.21.4 | Identical apart from 11 listed files, where pypdf 7.0.0's text differs |
| `check_xray_output.py` | x-ray 0.4.0 | 203 of 211 identical, 6 with the same error |

Each script lists its remaining differences with the reason; the upstream suites pin the same
versions.

### Known gaps

- R7 (AES-GCM) keep-encryption has no real fixture, and PDF MAC refusal was tested with an
  injected `/AuthCode`, not a real MAC file.
- Ghostscript refused AES-256-R5 with a four-character password, including on real pypdf's
  own file. qpdf, Poppler and pypdf open those files, and a seven-character password works.

## Out of scope for now

Linearization, object stream compression on write, and a pikepdf facade. Those can follow
once `PdfWriter` is stable.
