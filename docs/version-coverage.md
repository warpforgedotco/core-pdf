# PDF version coverage audit

This is a targeted audit of the chapter APIs currently present in `core-pdf-spec`,
reviewed on 2026-09-11. It records implemented differences, shared algorithms, and
known gaps. It is not an exhaustive inventory of every normative rule, a claim of
complete support for a PDF version, or a conformance report for a document.

The architecture and public usage are described in [standards.md](standards.md).
The implementation reference remains ISO 32000-2:2020 and the errata revision
pinned by `PDF_2_0_BASELINE`. Historical Adobe references establish earlier
behavior. An erratum can correct a shared rule without introducing a new format
version; changing the pinned reference requires its own review.

## How to read the audit

- **Context implemented**: a chapter API selects a tested rule using
  `SemanticContext`. This applies only to the behavior named in the row.
- **Shared / object-selected**: implemented algorithms use the relevant object
  fields or retain the same interpretation. Feature introduction and document
  conformance can require separate checks.
- **Partial / gap**: a known missing rule, incomplete algorithm, or unassessed
  historical behavior. Accepting an object does not establish support for it.

Existing low-level calls without context preserve their behavior; the new blend
API defaults to PDF 2.0 equations. Contextual chapter APIs reject unknown versions with `PdfUnsupportedError`;
declaration discovery retains unknown numeric versions. Core's
reader adapters may deliberately recover under-declared or malformed input; that
recovery does not establish conformance.

## Syntax, document declarations, and content

| Behavior | Version evidence | Implementation and limits |
| --- | --- | --- |
| Header, catalog version, revision history | Catalog `/Version` from PDF 1.4; a later revision cannot undo an earlier upgrade. ISO 32000-2 §7.5.2, §7.5.6, §7.7.2. | **Implemented declaration resolution.** `s_07_document.standards` supplies strict declaration rules; core composes the revision history. Malformed or unreadable history produces diagnostics. `test_standards.py` and core `test_document_standards.py`. |
| Developer extensions | ISO 32000-1:2008 (PDF 1.7) introduced extension dictionaries; PDF 2.0 also permits arrays of extension dictionaries and URL entries. ISO 32000-2 §7.12. | **Context implemented.** Strict parsers check base versions and declaration shapes. Exact prefix/base/level/revision coverage is separate from recognition. `test_standards.py`. |
| Unicode text-string markers | UTF-16BE from PDF 1.2, Adobe 1.2 §4.4; UTF-8 from PDF 2.0, ISO 32000-2 §7.9.2.2. | **Context implemented.** `decode_pdf_text_string` treats each BOM as ordinary PDFDocEncoding before its introduction. Resolver text decoding carries context. Core preserves under-declared Unicode recovery. `test_text_string_versions.py`, `test_standards.py`, and core `test_document_standards.py`. |
| Name `#xx` escapes | Adobe 1.2 §4.5 changes the interpretation of `#` in names. | **Context implemented.** `lexical_rules` selects literal `#` through 1.1 and escapes from 1.2. Context reaches names and dictionary keys through lexers, resolvers, object streams, xref/recovery, and nested content. Resolver context changes discard parsed caches. `test_lexical_versions.py`, `test_xref_versions.py`, and core lexical/xref integration tests. |
| NUL whitespace | Adobe 1.3 §3.1.1, Table 3.1; PDF Association's historical clarification linked below. | **Context implemented.** Strict lexers and scanners use five whitespace bytes through 1.2 and add NUL from 1.3. Tests cover token boundaries, arrays, hexadecimal strings, inline images, and xref data. Core deliberately retains NUL tolerance in older files. |
| PDFDocEncoding Euro slot | Adobe 1.3 Appendix D.1, note 1 introduces Euro at octal 240, previously unused. | **Context implemented.** Strict explicit pre-1.3 contexts reject byte `A0` in the PDFDocEncoding branch because its character is undefined. Unicode branches remain independent; no-context decoding and core's explicit under-declared-text recovery retain Euro. `test_text_string_versions.py`. |
| Object and cross-reference streams | Introduced in PDF 1.5; ISO 32000-1 §7.5.7–7.5.8. | **Shared / object-selected.** `ObjectStream` and `XRefScanner` decode their defined structures. Their availability in an older declared version is not a decoding override. `test_syntax_strict.py`, `test_xref_rows.py`. |
| Page tree, fields, labels, metadata, annotation appearance selection | ISO 32000-2 §7.7.3, §12.5–12.7, §12.4.2, §14.3. | **Partial.** The existing `s_07_document` helpers share current rules; these chapters have not received a complete historical audit. |
| Page UserUnit | PDF 1.6; ISO 32000-2 §8.3.2.3 and Table 31. | **Implemented geometry.** Strict `page_user_unit` validates a positive finite page-local number, with default 1 and no inheritance. Core retains raw geometry and exposes physical dimensions; rasterization applies the unit once, including crops, rotation, annotations, and fields. Structured output retains the scale; OCR budgets and image DPI use physical dimensions. Supported compatibility facades preserve their reference coordinate conventions. Spec and core `test_page_user_unit.py`; see [API coordinate units](api.md#page-coordinates-and-physical-size). |
| Content operators and compatibility sections | ISO 32000-2 §7.8.2 and chapter-specific operator definitions. | **Shared / partial.** `ContentInterpreter` carries optional semantic context through nested lexical parsing, validates operands and `BX`/`EX` scope, and preserves parser extension hooks. It does not exhaustively validate operator availability. `test_content*.py` and `test_lexical_versions.py`. |

Primary references: [Adobe PDF 1.2](https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/pdfreference1.2.pdf),
[Adobe PDF 1.3](https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/pdfreference1.3.pdf),
[historical NUL whitespace](https://pdfa.org/pdf-malformations-and-more/),
[ISO 32000-1:2008](https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/PDF32000_2008.pdf),
[ISO 32000-2 editions](https://pdfa.org/resource/iso-32000-2/), and
[syntax errata](https://pdf-issues.pdfa.org/32000-2-2020/clause07.html).

## Filters and security

| Behavior | Version evidence | Implementation and limits |
| --- | --- | --- |
| Filter identifiers | Flate from 1.2, JBIG2 from 1.4, JPX and Crypt from 1.5. ISO 32000-1 §7.4.1, Table 6. | **Shared / object-selected.** Registry and pipeline dispatch by filter name. Codec adapters and partial JBIG2 support do not constitute complete filter coverage. JBIG2 text, halftone, and refinement formats remain unsupported. `test_filter_decode_spec.py`, `test_graphics_filters.py`, `test_jbig2.py`. |
| Predictor parameter ranges | `Colors` limited to 1–4 before 1.3; 16-bit components from 1.5. ISO 32000-1 §7.4.4.3, Table 8. | **Partial.** Predictors accept current ranges. Historical parameter validation is missing; no alternative decoding formula was identified. `test_predictors.py` and core filter recovery tests. |
| Standard security handlers | ISO 32000-1 §7.6, Tables 20–22; ISO 32000-2 §7.6.3–7.6.4. | **Object-selected.** `/V`, `/R`, crypt filters, and cipher methods select RC4, AES-128, and AES-256 behavior. R5 and R6 password algorithms differ and are already separate. Header dispatch must not replace these selectors. Deprecation in PDF 2.0 does not justify rejecting old encryption during reading. |
| AES-256 R5 | Adobe PDF 1.7 extension level 3. | **Object-selected.** The exact extension identity is recorded. Strict dictionary tests and independent encrypted integration fixtures cover implemented handlers; additional standalone R5/R6 Unicode password known-answer tests remain useful. |
| AES-GCM R7 | ISO/TS 32003:2023 §5.1–5.2. | **Object-selected.** `/V 6`, `/R 7`, `/AESV4`, key/IV/tag lengths and the R6 password algorithm are implemented. This assessed subset is listed in exact extension coverage, not advertised as full cryptographic conformance. |
| PDF MAC | ISO/TS 32004:2024. | **Partial.** Permission bit, KDF salt, and standalone `AuthCode` verification are implemented and retained before standards discovery. Attached-signature MAC remains unsupported. |
| Public-key security | ISO 32000-2 §7.6.5. | **Gap.** The security factory explicitly reports unsupported public-key handlers. |

Security regression evidence includes strict R4 dictionary cases in spec
`test_security_integer_values.py` and R2–R7/MAC integration cases in core
`test_security_scalar_recovery.py`, with independent qpdf and pyHanko fixtures.
`test_security_version_bootstrap.py` pairs historical private names with real
security entries, password checks, escaped upgrades, recovered xrefs, and MAC tampering.
Legacy bootstrap selects the same name grammar for trailers and security objects,
preserves authentication entries during xref recovery, and authenticates again if
the final name grammar changes. Ambiguous version probes use the conservative
policy described in [standards.md](standards.md#document-declarations).
The extension references are available through the
[PDF Association extension registry](https://pdfa.org/extensions/).

## Graphics, fonts, and transparency

| Behavior | Version evidence | Implementation and limits |
| --- | --- | --- |
| Indexed Separation/DeviceN base | Allowed from PDF 1.3; Adobe 1.3 §4.5.5, pp. 181–182; ISO 32000-2 §8.6.6.3. | **Context implemented.** `parse_color_space(..., context=...)` enforces the older restriction recursively. Core's adapter retains tolerant parsing. `test_color_versions.py` pairs legacy errors, modern values, unchanged valid bases, and nested contexts. |
| Indexed lookup and color-space families | String lookup added in 1.2; ICCBased/DeviceN from 1.3. Adobe 1.2/1.3 color chapters. | **Shared / partial.** Parsing, component ranges, palettes, and tint inputs are shared. Earlier availability and lookup representation are not comprehensively validated. `test_color_spaces.py`, `test_indexed_color.py`. |
| ICC profile interpretation | ISO 32000-2 §8.6.5.5 and graphics errata. | **Object-selected.** The ICC header's v2/v4 version controls profile interpretation independently of the PDF header. Do not introduce a PDF-version gate in the core ICC adapter. |
| Sixteen-bit image samples | Added in 1.5; ISO 32000-2 §8.9.1, Table 87. | **Implemented decoding/rendering.** Full big-endian words pass through predictors, Decode arrays, color conversion, color-key masks, soft masks, and Matte unblending before output quantization. Raw and JPX paths include inline images and embedded JPX alpha. Codec support still bounds the accepted codestream formats; this is not complete image conformance validation. `test_sample_kernels.py`, `test_image_matte.py`, and core `test_image_16bit.py`. |
| JPX Decode arrays | ISO 32000-1 §7.4.9 and Table 89 versus ISO 32000-2 §7.4.9 and Table 87. | **Context implemented.** Earlier PDFs ignore Decode for JPX; PDF 2.0 applies it when ColorSpace is supplied. Image masks retain their prescribed exception. Context reaches image and mask preparation. `test_image_versions.py` and core `test_image_16bit.py`. |
| DeviceN NChannel | Added in 1.6; ISO 32000-2 §8.6.6.5. | **Partial.** Attributes are preserved; process and mixing semantics are not implemented. |
| Black-point compensation and rendering intent | `UseBlackPtComp` added in PDF 2.0; ISO 32000-2 §8.6.5.8–9, §11.7.5.3, Tables 57, 69, and 87. | **Implemented for current conversion paths.** Intent and compensation reach paint-time text/path colors, 8/16-bit images, shading, patterns, and nested Forms. Image intent remains local; stencils ignore it. Caches distinguish the settings; absolute colorimetric conversion ignores compensation while retaining its stored value. ICC profiles remain authoritative through Indexed and tint alternate spaces. Spec `test_color_rendering.py`, `test_color_math.py`, and core `test_color_rendering_reader.py`, `test_calibrated_colors.py` verify state and pixels. Selected defaults and arbitrary group blending-space limits are documented in [architecture](architecture.md#device-colour). |
| Halftone origin | PDF 2.0, ISO 32000-2 Table 57. | **Gap.** `HTO` requires a halftone renderer before its origin can affect raster output. |
| WinAnsi and MacRoman assignments | Adobe 1.2 Appendix C.1; Adobe 1.3 Appendix D.1, notes 1–3. | **Context implemented.** `get_base_encoding_glyph_names` selects bullet for pre-1.3 WinAnsi slots `80`, `8E`, and `9E`; 1.3 assigns Euro, Zcaron, and zcaron. Context reaches native font glyph selection and Unicode projection. Explicit built-in encodings and Differences retain precedence. PDF MacRoman retains currency at `DB` despite the platform encoding's later change, so no PDF-version branch is appropriate there. Spec and core `test_font_encoding_versions.py` cover decoding, advances, and facade text/geometry. |
| Standard 14 font dictionaries | Earlier dictionary omissions limited to PDF 1.0–1.7; ISO 32000-2 §9.6.2.2. | **Shared / partial.** Standard metrics remain available to readers. Checking omitted dictionary fields is a separate strict validation concern; removing metrics in 2.0 would be incorrect. A preexisting x-ray compatibility gap remains for Helvetica's implicit Euro width: core's retained table gives zero, while MuPDF supplies 556 units. Explicit PDF widths remain authoritative; this metric difference is separate from historical encoding selection. |
| TrueType character mapping | Adobe 1.3 §5.5.5; ISO 32000-1 §9.6.6.4; ISO 32000-2 §9.6.5.4. | **Historical audit: no universal earlier branch.** Adobe 1.3 describes Acrobat 4's mapping and explicitly allows earlier versions or implementations to differ. That caveat does not define a single alternative algorithm selected by an old PDF header. Core retains fontTools selection/recovery; spec owns symbolic-code helpers. Font-program selectors remain authoritative. |
| Composite and OpenType fonts | Adobe 1.2 Table 7.9; Adobe 1.3 Table 5.16; ISO 32000-2 Table 119, §9.7.6.2, §9.9. | **Historical audit: shared descendant rule.** The 1.2 table's generalized “one or more” wording is resolved by 1.3's explicit single-descendant requirement for all PDF versions through 1.3. Keep strict single-descendant parsing and core's existing malformed-array recovery. OpenType availability is still separate from decoding; this audit does not establish complete font-feature coverage. |
| Type 3/uncolored-pattern color operators | ISO 32000-2 §8.6.8 note 2 resolves earlier contradictory wording. | **Shared.** Existing ignore behavior follows the corrected rule. This is not a justified legacy/modern error switch. |
| Type 3 glyph programs | ISO 32000-2 §9.6.4. | **Gap.** Native rendering does not interpret Type 3 `CharProcs`. Color-control support for rendered glyphs does not establish procedural Type 3 rendering. |
| ColorDodge and ColorBurn | ISO 32000-1 §11.3.5, Table 136; Adobe 1.7 ExtensionLevel 5 §3.1; ISO 32000-2 §11.3.5, Table 134. | **Context implemented.** Scalar and array equations select the revised corners for PDF 2.0 or the exact audited Adobe extension identity. Dodge at backdrop 0/source 1 becomes 0; Burn at backdrop 1/source 0 becomes 1. Core carries context through page composition, paths, images, shading, patterns, and groups. Existing Normal/Multiply/Screen behavior is preserved. `test_blend_versions.py` and core `test_render_blend_versions.py`. |

References: [Adobe PDF 1.3](https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/pdfreference1.3.pdf),
[ISO 32000-1:2008](https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/PDF32000_2008.pdf),
[ISO 32000-2](https://pdfa.org/resource/iso-32000-2/), and
[graphics errata](https://pdf-issues.pdfa.org/32000-2-2020/clause08.html).
The [Adobe extension level 5 supplement](https://web.archive.org/web/20191221125453if_/https://www.adobe.com/content/dam/acom/en/devnet/acrobat/pdfs/adobe_supplement_iso32000_1.pdf)
documents the earlier introduction of the revised blend equations.

## Logical structure and profiles

| Behavior | Version evidence | Implementation and limits |
| --- | --- | --- |
| Structure dictionaries and attributes | ISO 32000-2 §14.7.3, §14.7.6. | **Shared / partial.** `parse_role_map`, revisioned `attribute_entries`, and `marked_content_id` expose strict local rules. Deprecated revision fields remain readable. This is not a complete structure validator. |
| Namespaces and transitive role mapping | ISO 32000-2 §14.7.3, §14.7.4.2, Table 356, §14.8.6. | **Context implemented.** `resolve_structure_role` follows root and explicit namespace maps, preserves namespace identities and paths, and terminates permitted cycles with a distinct status. An absent or null `NS` uses root mapping and defaults to the PDF 1.7 standard namespace even in PDF 2.0. Initial standard types may be remapped from PDF 1.5. Core exposes the resolved role, namespace, result, and malformed-input diagnostics. Vocabulary recognition is not hierarchy/profile validation. `test_structure_roles.py` and core `test_structure_role_namespaces.py`. |
| Profile claims | Identification mechanisms and editions listed in `STANDARD_PROFILES`. | **Implemented discovery, unverified claims.** XMP/Info discovery preserves claim identities and diagnostics. Recognition of PDF/A, PDF/UA, WTPDF, PDF/X, PDF/E, PDF/VT, or PDF/R is independent of validation support. See [the discovery matrix](standards.md#optional-external-validation). |
| PDF/A, PDF/UA, WTPDF machine checks | Exact editions in `VeraPdfBackend.supported_profiles`. | **External validation.** Optional veraPDF 1.30.2 adapter, 15 targets, original bytes, separate execution/conformance results. CI checks real passes and failures for all 15 advertised targets, with pinned, attributed positive fixtures and recorded reports. Human requirements remain outside the machine result. |
| PDF/X and other validators | Backend-specific profiles, editions, and report contracts. | **Gap.** No bundled PDF/X adapter yet. A locally installed engine is insufficient until profile selection and reports have been verified against real pass/fail fixtures. |

References: [ISO 32000-2](https://pdfa.org/resource/iso-32000-2/),
[structure errata](https://pdf-issues.pdfa.org/32000-2-2020/clause14.html),
[ISO/TS 32005 namespace inclusion rules](https://pdfa.org/resource/iso-32005/), and
[veraPDF validation targets](https://docs.verapdf.org/cli/validation/).

## Remaining work

The lexical, structure-role, 16-bit image, blend, positive-validator-fixture,
UserUnit, historical encoding, and current color-conversion controls above are implemented. The historical font
audit distinguishes actual encoding changes from clarified or implementation-specific
mapping rules. Further chapter work includes DeviceN NChannel, halftones, arbitrary
transparency-group blending spaces, Type 3 glyph programs, and the remaining partial/gap rows.

PDF/X integration is deferred. It requires an available backend, a usable license,
exact profile selection, and verified real pass/fail reports.

Each change should update its row and tests. New profiles stay separate from
format parsing, and new chapter algorithms stay in their owning spec module.
