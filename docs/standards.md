# PDF versions and standards validation

`core-pdf-spec` shares its chapter algorithms across PDF versions; the external-standard
kernels it wraps (JBIG2, predictors, the PostScript calculator, CMS and PDF MAC ciphers, and
the Adobe font formats) are their own distributions beneath it. A document's format
version, specification edition, developer extensions, and conformance claims are
separate identities. Recognizing one does not mean every feature is implemented or
that the document conforms.

## Document declarations

```python
from core_pdf import PdfDocument

with PdfDocument("document.pdf") as document:
    standards = document.standards
    print(standards.header_version, standards.catalog_version, standards.effective_version)
    print(standards.extensions, standards.profile_claims, standards.diagnostics)
```

`standards` is an immutable snapshot. Versions distinguish PDF 1.0 to 1.7 and 2.0 while
keeping unknown numbers; the original header and catalog strings are available too.
Missing or malformed declarations become diagnostics rather than turning ordinary
reading into validation. Extension records keep the decoded prefix, base version,
level, URL, and revision; a malformed extension dictionary yields a diagnostic and its
raw contents are not copied. The original PDF remains the validation evidence.

The header is read before cross-reference processing; catalog declarations are
resolved after authentication and decryption start, so mandatory PDF MAC checks run
first. The effective version combines the header, the current catalog, and earlier
incremental upgrades: a lower or removed `/Version` cannot undo an upgrade. If
revision history cannot be inspected, the snapshot says so. Extensions and profile
claims describe the current document only.

For PDF 1.0 and 1.1 headers, startup keeps literal names in trailers and security
dictionaries, and a separate probe reads only unencrypted catalog version names to
detect upgrades that use escaped names. The probe never decrypts content. If it is
inconclusive, core uses modern security parsing and rejects aliases that could erase
a literal authentication entry. That can reject a legacy file combining a malformed
escaped version field with escaped security names, which is preferred to silently
dropping an authentication requirement.

Profile discovery reads namespace-preserving XMP from the metadata stream and the
applicable Info entries, once, when `standards` is first requested. The `metadata`
projection is unchanged. Multiple claims stay separate, and conflicting or
unrecognized claims keep their source properties.

## Low-level semantics

```python
from core_pdf_spec.standards import PdfVersion, SemanticContext
from core_pdf_spec.s_07_syntax_primitives.text_string import decode_pdf_text_string

raw = b"\xef\xbb\xbfExample"
assert decode_pdf_text_string(raw, context=SemanticContext(PdfVersion(1, 7))) == "ï»¿Example"
assert decode_pdf_text_string(raw, context=SemanticContext(PdfVersion(2, 0))) == "Example"
```

Algorithms take a `SemanticContext` only where the standard's meaning changed between
versions. Calls that omit it keep their earlier behavior, and core owns best-effort
recovery for under-declared encodings and unknown versions. The version-sensitive
rules are:

- **Text strings.** The UTF-16BE BOM gained meaning in PDF 1.2 and the UTF-8 BOM in
  PDF 2.0; before that, those bytes are PDFDocEncoding. PDFDocEncoding slot `A0` was
  undefined before PDF 1.3, so an explicit older context rejects it.
- **Font encodings.** WinAnsi slots `80`, `8E`, and `9E` changed in PDF 1.3 from the
  bullet fallback to Euro, Zcaron, and zcaron.
  `s_09_fonts.helpers.get_base_encoding_glyph_names` and
  `build_simple_encoding_glyph_names` accept `context=`. Explicit font-program
  encodings, Differences, and ToUnicode keep their roles.
- **Lexing.** `PdfLexer`, `ObjectResolver`, `ObjectStream`, `XRefScanner`, and
  `ContentInterpreter` accept `semantic_context=`, and `lexical_rules(context)` gives
  standalone scanners the same immutable rules. Names keep a literal `#` through
  PDF 1.1 and use hex escapes from 1.2; strict parsing treats NUL as whitespace from
  1.3. Content context reaches nested streams and inline images. Changing
  `ObjectResolver.semantic_context` clears the object caches, so finish active parsing
  first.
- **Color.** `s_08_graphics.color_spec.parse_color_space(..., context=)` allows
  Separation and DeviceN Indexed bases from PDF 1.3, validates NChannel attributes,
  and exposes a typed DeviceN-to-process mapping. `color_space_paints` identifies
  `None`-colorant suppression without choosing a device.
  `color_rendering.ColorRendering` keeps rendering intent and PDF 2.0 black-point
  compensation; an unrecognized intent selects RelativeColorimetric as prescribed,
  and `image_color_rendering` applies an image's own intent.
- **Blending.** `s_11_transparency.blend.blend_component` and `blend_components`
  select the historical or revised PDF 2.0 ColorDodge and ColorBurn equations,
  defaulting to PDF 2.0. Image preparation keeps 16-bit precision through Decode,
  masks, and color conversion, and JPX handling follows the effective version.
- **Page scale.** `s_07_document.page.page_user_unit` validates the leaf page's
  `UserUnit` (default 1, not inherited). See
  [page coordinate units](api.md#page-coordinates-and-physical-size).
- **Structure roles.** `s_14_structure.roles.resolve_structure_role` follows
  transitive root and namespace role maps, reports cycles without looping, and falls
  back to the PDF 1.7 standard namespace when none is declared, even in PDF 2.0.
  Core exposes the result as `role`, `role_namespace`, `role_resolution`, and
  `role_error`, with `type` keeping the original tag.

Reader adapters extend the strict parsers through documented hooks whose defaults
keep the specified behavior: `PdfLexer.select_lexical_rules`, resolver `decode_text`,
interpreter `create_lexer`, the lexer's `handle_invalid_hex_string`,
`handle_invalid_name_escape`, `dictionary_end_length`, `handle_dictionary_key_error`,
`handle_duplicate_dictionary_key`, and `handle_dictionary_entry_error`, the
`unknown_escape` and `eol_pair` hooks of `read_literal_string`, and the `node_type`,
`on_invalid_child`, `max_depth`, and `stop_at_malformed_parent` options of the
page-tree walkers.

Format selection never replaces object-level selectors such as encryption `/V`, `/R`,
or crypt-filter names. `STANDARD_PROFILES` lists known profile editions and base
versions; `IMPLEMENTED_EXTENSIONS` lists audited extension features, and a higher
extension level is not automatically supported. `PDF_2_0_BASELINE` pins the ISO
32000-2:2020 reference and errata; both the 2017 and 2020 editions describe PDF 2.0
and the header cannot select between them. A profile's extra constraints belong
beside the affected chapter algorithm with a citation and test.

## Optional external validation

Install `core-pdf-validate` separately and point it at a local veraPDF 1.30.2
executable; it never installs Java or a validator itself.

```python
from core_pdf_validate import VeraPdfBackend, validate

report = validate(
    "document.pdf",
    profiles=("pdfa-2u", "pdfua-1"),
    backend=VeraPdfBackend(executable="verapdf", timeout=60),
)
for result in report.results:
    print(result.profile, result.execution_status, result.conformance)
```

Validation runs on an immutable snapshot of the original bytes, recorded by SHA-256,
so core's recovery, substitute fonts, and extracted output never become evidence, and
explicit targets do not need core to parse the file. `profiles="declared"` selects the
document's own claims; when none are readable the report has no target and explains
why, with no fallback profile. Every requested target gets its own result, including
unsupported ones, and duplicates run once.

| Family | Automatic claim discovery | Bundled adapter |
| --- | --- | --- |
| PDF/A-1 to 4 | XMP identification properties | veraPDF |
| PDF/UA-1 and 2 | XMP identification properties | veraPDF |
| WTPDF 1.0 | PDF declarations in XMP | veraPDF |
| PDF/X | XMP and Info identification entries | None |
| PDF/E-1, PDF/VT-1 to 3 | XMP identification properties | None |
| PDF/R-1 | Catalog identity only | None |

`VeraPdfBackend.supported_profiles` lists the veraPDF targets. Other families can be
recognized but need another `ValidationBackend`, which must advertise exact profile
identifiers and editions.

`execution_status="completed"` means the machine checks finished; `conformance` is
`"pass"`, `"fail"`, or `"not_checked"`. Missing engines, unsupported profiles or
engine versions, timeouts, and invalid or incomplete reports are distinct execution
results with `"not_checked"` conformance. A machine pass does not cover requirements
needing human judgment, such as semantic accessibility. Reports keep the engine
version, rule references, profile edition, locations, raw XML, stderr, and coverage
limitations. The adapter verifies the report's profile, engine and model versions,
completion, and rule counts against the exit status; changing supported engine
versions requires report-contract tests.

A dedicated CI job installs checksum-pinned veraPDF 1.30.2 with Java 17 and runs
real-engine positive and negative tests for every advertised target. See the
[companion README](../packages/core-pdf-validate/README.md) for local reproduction.

## References and maintenance

- [ISO 32000-2 and corrected editions](https://pdfa.org/resource/iso-32000-2/).
- [Version and extension declaration errata](https://pdf-issues.pdfa.org/32000-2-2020/clause07.html).
- [UTF-8 text strings](https://pdfa.org/understanding-utf-8-in-pdf-2-0/).
- [Arlington's structural model and limitations](https://github.com/pdf-association/arlington-pdf-model).
- [veraPDF targets and report behavior](https://docs.verapdf.org/cli/validation/).

Arlington is a reference for structural and version metadata, not a runtime dependency
or a substitute for the content-stream, lexical, encryption, or file-layout
algorithms. New semantic differences need chapter-local implementations with cited
regression tests; new validator targets need explicit backend coverage and realistic
report fixtures. The wheel checks verify that spec works without core and that
explicit validation imports neither core nor an installed engine.
