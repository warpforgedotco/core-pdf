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

Only core and validation-model **1.30.2** report versions are accepted. The XML
format and executable boundary are tested against recorded real pass/fail reports.
Supporting another engine or release requires an adapter with tested capabilities;
implement `ValidationBackend` to provide one. Source I/O errors and invalid API
arguments raise normal Python exceptions.

Run the companion suite with the workspace environment:

```sh
uv run --all-packages --group test pytest packages/core-pdf-validate/tests
```

Set `CORE_PDF_VERAPDF` to an installed executable to additionally run real-engine
integration tests against initialized reference corpora. Every one of the 15
advertised targets has an actual machine-pass fixture and an actual machine-fail
fixture, checking profile names, standards editions, rule references, and report
parsing. Nine small positive PDFs cover Unicode/ActualText, associated and embedded
files, optional content, and both generations of accessible tagging. Their pinned
upstream provenance, CC BY 4.0 attribution, hashes, and the single documented
PDF/A-3 derivative are recorded in [the fixture manifest](tests/fixtures/positive/manifest.json).
Authentic pass-report recordings cover all targets in ordinary tests without Java.
Once opted in, missing fixtures fail the tests; positive cases also verify that
the validator receives the expected source bytes and leaves them unchanged.

The dedicated `veraPDF 1.30.2 / Java 17` CI job downloads the exact official
[Greenfield 1.30.2 installer](https://software.verapdf.org/rel/1.30/verapdf-greenfield-1.30.2-installer.zip),
verifies its pinned SHA256, and installs only in runner temporary storage using
[upstream's automated installer](https://docs.verapdf.org/install/#automated-installation).
`CORE_PDF_REQUIRE_VERAPDF=1` makes missing engine configuration fail in this job.
The engine's own report versions are also checked by the adapter. To reproduce
the job locally with Java 17 on `PATH` and fixture submodules initialized:

```sh
python scripts/install_verapdf_ci.py /tmp/core-pdf-verapdf-ci
CORE_PDF_VERAPDF=/tmp/core-pdf-verapdf-ci/verapdf \
  CORE_PDF_REQUIRE_VERAPDF=1 \
  JAVA_TOOL_OPTIONS=-Duser.home=/tmp/core-pdf-verapdf-ci \
  uv run --locked --all-packages --extra unstructured --group test pytest \
  packages/core-pdf-validate/tests/test_verapdf_integration.py -ra
```

The destination must be new. `--archive /path/to/installer.zip` reuses a local
download with the same checksum requirement. This developer script remains
separate from the installed package; normal validation never installs software.
