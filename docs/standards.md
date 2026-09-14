# PDF versions and standards validation

`core-pdf-spec` shares its chapter algorithms across PDF versions. A document's
format version, specification edition, developer extensions, and conformance
claims are separate identities. Recognizing an identity does not imply complete
implementation of every feature or establish conformance.

## Document declarations

```python
from core_pdf import PdfDocument

with PdfDocument("document.pdf") as document:
    standards = document.standards
    print(standards.header_version)
    print(standards.catalog_version)
    print(standards.effective_version)
    print(standards.extensions)
    print(standards.profile_claims)
    print(standards.diagnostics)
```

`standards` is an immutable snapshot. Its numeric versions distinguish PDF
1.0–1.7 and 2.0, while retaining unknown version numbers. Original header and
catalog declaration strings are available separately. Missing or malformed
declarations produce diagnostics; they do not turn ordinary reading into
conformance validation.

Extension records retain decoded prefix, base version, level, URL, and revision.
Malformed extension dictionaries produce diagnostics; their arbitrary raw object
contents are not duplicated into this snapshot. The original PDF remains the
validation evidence.

The header is read before cross-reference processing. Catalog declarations are
resolved after authentication and decryption initialization, preserving mandatory
PDF MAC checks. The effective version incorporates the header, current catalog,
and earlier incremental catalog upgrades. A lower or removed `/Version` cannot
erase an earlier upgrade. If revision history cannot be inspected, the snapshot
reports that limitation. Extensions and profile claims describe the current
document, rather than an accumulation of obsolete declarations.

For PDF 1.0/1.1 headers, startup first preserves literal names in trailers and
security dictionaries. A separate probe inspects only unencrypted catalog version
names, including prior revisions, to recognize upgrades that themselves use escaped
names. It never decrypts content or discovers extensions or profile claims. The
selected security view is authenticated before full declaration discovery; changing
the name grammar afterward requires reparsing and authentication again.

If that version probe is inconclusive, core conservatively uses modern security
parsing and rejects aliases that could erase a literal authentication entry. This
can reject an otherwise readable legacy file combining a malformed private escaped
version field with private escaped security names. This bootstrap limitation avoids
silently dropping authentication requirements in ambiguous input.

Profile discovery reads namespace-preserving XMP from the document's metadata
stream and applicable Info entries. It runs once, when `standards` is requested.
The existing `metadata` projection remains unchanged. Multiple profile claims
remain separate; conflicting or unrecognized claims retain their source properties.

## Low-level semantics

```python
from core_pdf_spec.standards import PdfVersion, SemanticContext
from core_pdf_spec.s_07_syntax_primitives.text_string import decode_pdf_text_string

legacy = SemanticContext(PdfVersion(1, 7))
modern = SemanticContext(PdfVersion(2, 0))
raw = b"\xef\xbb\xbfExample"
assert decode_pdf_text_string(raw, context=legacy) == "ï»¿Example"
assert decode_pdf_text_string(raw, context=modern) == "Example"
```

The UTF-16BE BOM gained text-string meaning in PDF 1.2; the UTF-8 BOM did so in
PDF 2.0. Before each introduction, those bytes use PDFDocEncoding. This changes contextual
interpretation, not lexical string parsing. Algorithms receive context only where
their semantics differ. Existing calls omitting context retain their earlier behavior.
Core owns best-effort recovery for under-declared encodings and unknown versions.

PDFDocEncoding's `A0` slot was undefined before PDF 1.3. An explicit older context
rejects it in a PDFDocEncoding string; Unicode strings and no-context decoding keep
their established behavior. Core can recover under-declared Euro text explicitly.
Font WinAnsi slots `80`, `8E`, and `9E` likewise changed in PDF 1.3, from the prescribed
bullet fallback to Euro, Zcaron, and zcaron. `s_09_fonts.helpers.get_base_encoding_glyph_names`
and `build_simple_encoding_glyph_names` accept `context=` for these assignments.
Native font selection and extraction use the document context; explicit font-program
encodings, Differences, and ToUnicode retain their established roles. Compatibility
facades retain their reference libraries' encoding and positioning conventions.

`PdfLexer`, `ObjectResolver`, `ObjectStream`, `XRefScanner` parsing methods, and
`ContentInterpreter` accept `semantic_context=`. Names retain literal `#` through
PDF 1.1; hexadecimal name escapes apply from PDF 1.2. Strict parsing recognizes
NUL whitespace from PDF 1.3. `lexical_rules(context)` provides the same immutable
rules for standalone scanners. Content context reaches nested streams and inline
image boundaries. Core retains tolerant NUL handling and malformed-name recovery.

Changing `ObjectResolver.semantic_context` clears parsed object and object-stream
caches so later resolutions use the new rules. Finish active parsing before changing
it; objects already returned to the caller remain unchanged. `PdfLexer.select_lexical_rules`,
resolver `decode_text`, and interpreter `create_lexer` are public parser extension
points for reader adapters, as are the lexer's `handle_invalid_hex_string`,
`handle_invalid_name_escape`, `handle_dictionary_key_error`,
`handle_duplicate_dictionary_key`, and `handle_dictionary_entry_error` methods, the
`unknown_escape`/`eol_pair` hooks of `read_literal_string`, and the `node_type`,
`on_invalid_child`, `max_depth`, and `stop_at_malformed_parent` options of the page-tree
walkers. Their defaults keep the specified strict behaviour.

`s_08_graphics.color_spec.parse_color_space(..., context=...)` also applies the
PDF 1.3 change permitting Separation and DeviceN Indexed bases, including nested
color spaces. It does not validate the availability of every color-space feature.
DeviceN parsing also validates NChannel attributes and exposes a typed mapping from
DeviceN inputs to process-space components. Core uses that mapping for process-only
NChannel conversion. Mixed spot/process spaces can use core's documented sRGB approximation;
unsupported mixing inputs and malformed attributes retain the global tint fallback.
`s_08_graphics.color.color_space_paints` identifies prescribed `None`-colorant suppression
without selecting an output device; core keeps extraction and clipping while suppressing paint.
These object-selected semantics do not require a different
document parser for each PDF version.
The [chapter coverage audit](version-coverage.md) lists implemented differences,
shared algorithms, evidence, and remaining gaps.

`s_07_document.page.page_user_unit` validates the resolved leaf-page `UserUnit` entry.
The value defaults to 1, is not inherited, and scales default user space into physical
points. This is an object-level scale, not an alternative page parser selected by
the header. Core keeps raw geometry alongside the scale and applies it during
rasterization; see [page coordinate units](api.md#page-coordinates-and-physical-size).

`s_08_graphics.color_rendering.ColorRendering` retains rendering intent and PDF 2.0
black-point compensation as immutable values. Strict parsers enforce the three
compensation names; an unrecognized rendering-intent name selects RelativeColorimetric,
as prescribed by the standard. Absent or null ExtGState entries preserve current state.
`image_color_rendering` applies an image's own intent while preserving compensation;
stencils ignore image intent. These APIs describe PDF state, without choosing an output
profile. Core applies the state during its supported color conversions; its selected
defaults and transparency limits are described in [the architecture](architecture.md#device-colour).

`s_14_structure.roles.resolve_structure_role` follows transitive root and namespace
role maps and returns the terminal name, namespace, status, and path. It reports
permitted cycles without looping. An absent namespace ultimately uses the PDF 1.7
standard namespace even in a PDF 2.0 document; explicit namespaces select their own
maps. Core structure elements expose this through `role`, `role_namespace`,
`role_resolution`, and `role_error` while `type` retains the original tag.

`s_11_transparency.blend.blend_component` and `blend_components` implement ColorDodge
and ColorBurn before alpha compositing. Context selects the historical equations or
the revised PDF 2.0/Adobe 1.7 extension-level-5 equations. These new APIs default to
PDF 2.0 when context is omitted. Page rendering carries document context through
scalar and array paths. Image preparation preserves 16-bit sample precision through
Decode, masks, and color conversion before producing the selected 8-bit raster;
JPX Decode handling also follows the effective version.

Format selection does not replace object-level selectors such as encryption `/V`,
`/R`, or crypt-filter names. Extension coverage is recorded for exact declared
identities and implemented features; a higher extension level is not automatically
supported. `STANDARD_PROFILES` records known profile editions and base versions,
while `IMPLEMENTED_EXTENSIONS` records the explicitly audited extension features.

`PDF_2_0_BASELINE` pins ISO 32000-2:2020 reference and errata provenance. The 2017
and 2020 specification editions both describe PDF 2.0; the file header cannot
select an edition. The pinned reference is not a claim that every erratum has
already been implemented. A profile's additional constraints do not globally
replace PDF parsing; any profile-defined semantic extension belongs beside the
affected chapter algorithm with a citation and test.

## Optional external validation

Install `core-pdf-validate` separately and configure a local veraPDF 1.30.2
executable. The companion does not install Java or validator binaries at runtime.

```python
from core_pdf_validate import VeraPdfBackend, validate

report = validate(
    "document.pdf",
    profiles=("pdfa-2u", "pdfua-1"),
    backend=VeraPdfBackend(executable="verapdf", timeout=60),
)
for result in report.results:
    print(result.profile, result.execution_status, result.conformance)
    print(result.diagnostics, result.limitations)
```

Explicit targets do not require core to parse the document. The companion takes
an immutable snapshot of the original path or bytes, records its SHA-256 digest,
and supplies identical bytes to each backend invocation. Core's recovery,
substitute fonts, and extracted output never become validation evidence.

Use `profiles="declared"` to select the document's identified claims. Missing or
unreadable claims produce a report with no target and explanatory diagnostics;
there is no fallback to an arbitrary PDF/A profile. Every requested target gets
its own result, including unsupported targets. Duplicate targets run once.

The catalog also records PDF/E, PDF/VT, and PDF/R identities without claiming
native conformance checks. Profile identification support is documented separately
from catalog recognition; an explicit target can be requested even without a claim.

| Family | Automatic claim discovery | Bundled adapter |
| --- | --- | --- |
| PDF/A-1–4 | XMP identification properties | veraPDF |
| PDF/UA-1–2 | XMP identification properties | veraPDF |
| WTPDF 1.0 | PDF declarations in XMP | veraPDF |
| PDF/X | XMP and Info identification entries | None |
| PDF/E-1, PDF/VT-1–3 | XMP identification properties | None |
| PDF/R-1 | Catalog identity only; raster declaration comments are not parsed | None |

The veraPDF adapter supports PDF/A-1/2/3/4, PDF/UA-1/2, and WTPDF 1.0 targets
listed by `VeraPdfBackend.supported_profiles`. PDF/X declarations can be recognized,
but PDF/X validation requires another backend. Applications can provide a
`ValidationBackend` advertising exact profile identifiers and editions.

`execution_status="completed"` means the validator finished its machine checks.
`conformance` is `"pass"`, `"fail"`, or `"not_checked"`. Missing engines,
unsupported profiles or engine versions, timeouts, invalid reports, and incomplete
processing remain distinct execution results with `"not_checked"` conformance.
A machine pass does not satisfy requirements needing human judgment, including
semantic accessibility checks. Reports retain engine version, rule references,
profile edition, locations, raw XML, stderr, and coverage limitations.

The adapter requests each profile explicitly, verifies the report's profile and
engine/model versions, checks completion and rule counts, and compares those with
the exit status. Changing supported engine versions requires report-contract tests;
there is no unchecked version override.

A dedicated CI job installs checksum-pinned veraPDF 1.30.2 with Java 17 and runs
real-engine positive and negative tests for all 15 advertised targets. Positive
fixtures have pinned provenance, hashes, license notices, and recorded engine reports.
Missing engine configuration or fixtures fails that job. See the
[companion README](../packages/core-pdf-validate/README.md) for local reproduction.

## References and maintenance

- [ISO 32000-2 and corrected editions](https://pdfa.org/resource/iso-32000-2/).
- [Version and extension declaration errata](https://pdf-issues.pdfa.org/32000-2-2020/clause07.html).
- [UTF-8 text strings](https://pdfa.org/understanding-utf-8-in-pdf-2-0/).
- [Arlington's structural model and limitations](https://github.com/pdf-association/arlington-pdf-model).
- [veraPDF targets and report behavior](https://docs.verapdf.org/cli/validation/).

Arlington is a reference for structural/version metadata, not a runtime dependency
or a replacement for content-stream, lexical, encryption, or file-layout algorithms.
New semantic differences need chapter-local implementations and cited regression
tests. New validator targets need explicit backend coverage and realistic report
fixtures. The wheel checks verify that spec works without core and that explicit
validation does not import core or require an installed engine.
