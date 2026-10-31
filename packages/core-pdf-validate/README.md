# core-pdf-validate

Optional local conformance validation of original PDF bytes. Core never imports
or discovers this companion. Install it separately, along with a local
[veraPDF 1.30.2](https://github.com/veraPDF/veraPDF-library/releases/tag/v1.30.2)
installation and its required Java runtime. The package never downloads an engine.

```python
from core_pdf_validate import VeraPdfBackend, validate

report = validate(
    "document.pdf",  # paths or bytes
    profiles=["pdfa-3b", "pdfua-2"],
    backend=VeraPdfBackend("/opt/verapdf/verapdf", timeout=60),
)
for result in report.results:
    print(result.profile, result.execution_status, result.conformance)
```

`profiles` is a canonical identifier, a sequence of identifiers, or `"declared"`.
Explicit targets work even when core cannot parse the input. Declared targets are
read through public `PdfDocument.standards`; absent or unreadable claims return
`no_target_identified` in report diagnostics, with no arbitrary default target.
Repeated targets run once. Unknown or unsupported identifiers get their own
`unsupported_profile` / `not_checked` result.

The supported targets are PDF/A `pdfa-1a`, `pdfa-1b`, `pdfa-2a`, `pdfa-2b`,
`pdfa-2u`, `pdfa-3a`, `pdfa-3b`, `pdfa-3u`, `pdfa-4`, `pdfa-4e`, `pdfa-4f`;
PDF/UA `pdfua-1`, `pdfua-2`; and WTPDF `wtpdf-1.0-reuse`,
`wtpdf-1.0-accessibility`. Exact standards editions are advertised in
`VeraPdfBackend.supported_profiles`. PDF/X is recognized by core's standards
catalog but has no validation adapter here.

Every target is requested explicitly in a separate local subprocess. Input paths
are read once; private temporary copies contain identical bytes for each target.
`source_sha256` identifies those original bytes. Input files are never rewritten.
Profiles inferred from repaired reader objects are declarations only; the engine
always validates the original source snapshot.

Frozen reports retain execution status separately from machine conformance
(`pass`, `fail`, or `not_checked`), engine/version, rule references and locations,
original XML bytes, stderr, diagnostics, and coverage limitations. A machine pass
does not establish human-reviewed requirements, including semantic accuracy and
accessibility for PDF/UA and WTPDF. Missing engines, timeouts, unfamiliar engine
versions, contradictory or malformed reports, and incomplete checks never pass.

Only core and validation-model **1.30.2** report versions are accepted.
Supporting another engine or release requires an adapter;
implement `ValidationBackend` to provide one. Source I/O errors and invalid API
arguments raise normal Python exceptions.
