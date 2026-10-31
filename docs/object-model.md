# Editable object model and writer

Core-pdf reads PDFs but cannot change them. This is the design for lossless editing
(roadmap item 5). It is also what pypdf's `PdfWriter` and `generic` types, and any future
pikepdf facade, are built on.

## Why now

`pypdf`'s upstream suite stops here. Of 1,031 tests that pass on the real library, the
facade can attempt 305, and 726 are skipped because they reach `PdfWriter` or the object
types. The rest of the facades pass their in-scope suites.

## Requirements

1. **Lossless:** an object nobody changed is written back byte for byte, so signatures,
   unknown keys and producer quirks survive.
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
take numbers past the highest in the file. Saving walks from the trailer, so unreferenced
objects are dropped by a full rewrite and kept by an incremental update.

### Encrypted input

`EditableDocument` refuses an encrypted file. Milestone 6 adds a stated policy for
decrypted and encrypted output.

## Milestones

Each milestone is one or more reviewable PRs. Counts come from the upstream harness.

1. **Serializer.** Round-trips every object in the fixture corpus through the lexer.
2. **Editable document and full rewrite.** Open, change, save, reopen, and compare.
3. **Incremental update.** Untouched bytes stay identical, checked by prefix comparison.
4. **pypdf `generic` types.** Removes the largest set of stand-ins.
5. **`PdfWriter`.** Pages, metadata, then annotations, forms and merging.
6. **Encrypted output and xref streams.**

## Out of scope for now

Linearization, object stream compression on write, and a pikepdf facade. Those can follow
once `PdfWriter` is stable.
