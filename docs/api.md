# Public API and compatibility facades

The canonical public API is exported from `core_pdf`. `PdfDocument`, `PdfPage`, structured
records, runtime controls, and errors are lazy exports backed by their engine owners.
The same public document API handles every recognized PDF format version.

```python
from core_pdf import PdfDocument

with PdfDocument.open("document.pdf") as document:
    print(document.structured_document.text)
```

`PdfPage.extract()` and `PdfPage.structured_view` return a structured `Page`;
`PdfDocument.extract()` and `PdfDocument.structured_document` return a structured `Document`.

## Extraction adapters

`DocumentAdapter`, exported from `core_pdf`, describes an `apply(document: Document) -> Document`
method. Adapters run in the supplied order after extraction releases its document operation.
They transform the structured result, and do not need to inherit from the protocol:

```python
from dataclasses import replace

from core_pdf import Document, PdfDocument


class AddTag:
    def apply(self, document: Document) -> Document:
        return replace(document, metadata={**document.metadata, "tag": "review"})


with PdfDocument.open("document.pdf") as document:
    result = document.extract(adapters=(AddTag(),))
```

## OCR extraction

Starting with 0.0.6, `core_pdf.PdfDocument` and the compatibility facades extract only PDF-native
text. Existing hidden text layers remain PDF-native text; image-only or flattened vector text
requires the separate `core-pdf-ocr` package. Installing it never changes core's behavior.

```python
from core_pdf_ocr import PdfDocument

with PdfDocument("scanned.pdf") as document:
    result = document.extract(pages=[1, 2])
    print(result.to_markdown())
```

`core_pdf_ocr.PdfDocument` and `PdfPage` subclass the core public classes and preserve their
constructor, selection, adapters, rendering, and close/cancellation behavior. Their extraction
includes native, OCR, and hybrid routes and returns the same structured model types. Shared
records and errors remain exported from `core_pdf`. The companion pins its exact matching core
release; its internal stage imports are not a public extension API.

Both packages inherit the same extraction lifecycle. The companion overrides the extraction
hooks and page factory to select recognition without copying operation or adapter handling.

Use `core-pdf-ocr document.pdf --print` or `python -m core_pdf_ocr document.pdf --print` to
select recognition from the command line. The options match `core-pdf`.

## Structured JSON

`Document.to_json_dict()` and `Document.to_json()` emit schema 5.0. The document is a normalized
graph: `pages` reference ordered `nodes`, nodes reference one canonical block/table/figure payload,
and blocks reference canonical lines. Stable IDs use
`p{page_number}:{kind}:{zero_based_index}`. Schema 4's duplicated page payloads and document-wide
line/table reference snapshots are not emitted, and there is no schema-4 compatibility encoder.

Metadata must already be composed of JSON scalar, mapping, list, or tuple values. Unsupported
objects raise a path-specific `TypeError` instead of being silently converted to strings.

## Compatibility facades

`core_pdf.api.compat` provides local projections for common third-party PDF interfaces.
Each facade imports engine owners directly and can be imported independently; the package root
resolves convenience exports lazily. The Unstructured facade requires an optional NLP extra;
the other facades require no additional runtime dependencies.

```python
from core_pdf.api.compat.pdfminer import extract_text
from core_pdf.api.compat.pdfplumber import open as open_pdf
from core_pdf.api.compat.pypdf import PdfReader

print(extract_text("document.pdf"))
with open_pdf("document.pdf") as pdf:
    print(pdf.pages[0].extract_words())
```

Additional local facades are available for `pikepdf`, `unstructured`, `llamaindex`, and x-ray.
They reproduce useful high-level behavior, not every implementation detail or private API of the
upstream libraries.

### Unstructured NLP requirements

Install `core-pdf[unstructured]` with spaCy and the pinned `en_core_web_sm` 3.8.0 model:

```sh
pip install "core-pdf[unstructured]" \
  "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
```

Both dependencies are declared by name in the extra. The model is distributed as an official
GitHub wheel, so pip needs its URL in the install command. In the workspace, uv uses the source
configured in `pyproject.toml`: `uv sync --extra unstructured`.

Importing `core_pdf.api.compat.unstructured`, any of its element classes, or the parent package's
`partition_pdf` convenience export loads the English pipeline. Missing dependencies or model
loading failures raise an actionable `ImportError` with the original cause. Tokenization,
sentence boundaries, and POS tags always come from that model; there is no lexical fallback or
runtime model download. The pipeline is reused for subsequent calls. Existing PDF projection and
classification rules remain unchanged when the model is available.

Importing `core_pdf`, `core_pdf.api.compat`, or another facade does not load the NLP model.

### Compatibility validation

An experimental PyMuPDF facade exists under `core_pdf.api.compat._unsupported.pymupdf`; it is
not part of the supported compatibility surface.

Differential tests in `tests/src/core_pdf/api/compat/differential` compare the `pdfplumber`,
`pypdf`, `pikepdf`, `unstructured`, `llamaindex`, and x-ray facades against their reference
libraries. By default, each facade uses its own upstream fixture corpus, plus selected
cross-corpus redaction cases for x-ray:

```sh
git submodule update --init --recursive
uv run --locked --extra unstructured --group test --group vendor-test pytest -n auto
```

Run the exhaustive every-facade/every-fixture matrix explicitly when compatibility work
calls for it:

```sh
CORE_PDF_COMPAT_DIFFERENTIAL_FULL=1 \
  uv run --locked --extra unstructured --group test --group vendor-test pytest -n auto
```

The x-ray facade performs its redaction inspection directly from engine drawing, glyph, and
raster evidence:

```python
from core_pdf.api.compat import inspect_xray

findings = inspect_xray("document.pdf")
```


## Low-level specification library

`core-pdf-spec` is independently installable and versioned. Its import namespace is
`core_pdf_spec`, with modules grouped by PDF chapter (`s_07_syntax`, `s_08_graphics`,
`s_09_fonts`, and the other implemented chapters). It supplies strict algorithms, PDF
primitives, standard data, and semantic service protocols. It has no `PdfDocument` facade
or command-line entry point; applications compose those operations or use `core_pdf`.

Core currently supports `core-pdf-spec>=0.4.1,<0.5.0`. Supported module exports are listed in
`__all__`; names beginning with `internal_` remain private. Parser extension methods used by
core are documented alongside their strict implementations. Spec reports errors rather than
repairing malformed input, except where a referenced standard prescribes a fallback or default.
This covers implemented features and is not a complete PDF conformance validator.

Core's public exceptions remain available under their existing names and share spec's error
identities. Core retains reader recovery, fontTools backends, Unicode guesses, selected device
color conversion, capture, extraction, and rendering. OCR continues to depend on core's exact
version and does not change native behavior when installed.

## PDF versions and conformance claims

`PdfDocument.standards` returns an immutable `core_pdf_spec.standards.DocumentStandards`:
header, catalog, and effective versions; current extensions; unverified profile claims;
and declaration diagnostics. Its `.context` supplies version-sensitive low-level semantics.
Profile discovery retains XMP namespaces and leaves the existing metadata shape unchanged.

Structure elements retain their original tag in `type`; `role` follows transitive
root or namespace role mappings. `role_namespace` identifies the resulting namespace,
`role_resolution` retains the path and terminal status (including cycles), and
`role_error` records malformed mapping recovery. These projections do not certify
logical hierarchy or accessibility conformance.

The separately installed `core_pdf_validate.validate(source, *, profiles, backend=None)`
returns a `ValidationReport` with independent results for explicit profile IDs or
`profiles="declared"`. It runs a configured local validator against original bytes and
does not require successful core parsing for explicit targets. See
[standards interfaces and coverage](standards.md) and the
[validation companion](../packages/core-pdf-validate/README.md).
