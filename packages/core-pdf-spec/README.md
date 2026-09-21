# core-pdf-spec

Strict PDF and referenced-standard algorithms for Python 3.13+.

```sh
pip install core-pdf-spec
```

```python
from core_pdf_spec.s_07_syntax.lexer import PdfLexer
from core_pdf_spec.types import PdfName

lexer = PdfLexer(b"<< /Type /Page >>")
try:
    value = lexer.parse_dictionary_or_stream()
    assert value["Type"] == PdfName.of("Page")
finally:
    lexer.close()
```

Modules follow PDF chapters: syntax primitives, objects/xref, filters, content execution,
document semantics, security, graphics, fonts, and structure. A module's `__all__` lists its
supported exports; `internal_` names are private. Public parser methods document strict
parsing and the extension points consumers can implement.

The library contains implemented PDF semantics and the PDF-facing wrappers over the
referenced standards. The referenced-standard kernels themselves are separate floor
distributions that spec depends on: `core-predictors`, `core-postscript`, `core-jbig2`,
`core-pdf-crypto`, and `core-adobe-fonts` (which also carries the bundled CMaps and
standard font tables). It preserves standard-defined defaults and fallback rules. Malformed-input repair, retry/skip policy, substitute fonts, fontTools backends,
Unicode guesses, output-device choices, capture products, and rasterization belong to
applications such as `core-pdf`. Semantic sinks, stream decoders, and font providers are supplied
through typed interfaces. Unsupported algorithms report failure instead of selecting a backend.

There is no document facade, CLI, or complete conformance-validator claim. Core and OCR are
not dependencies and are never discovered or imported. Runtime dependencies are NumPy and
the five floor packages; cryptography and asn1crypto arrive through `core-pdf-crypto`, and
bundled font data retains its original notices inside `core-adobe-fonts`.

For Lab color conversion, `s_08_graphics.color_math.lab_components_to_xyz` accepts
NumPy float32 rows of actual `(L*, a*, b*)` components and a reference white point.
Image sample decoding and range enforcement belong to the caller. The existing
`lab_to_xyz` function retains its normalized input convention as a compatibility wrapper.

The current spec version is `0.5.0`, released independently of core. During `0.x`, breaking
changes to supported interfaces require a new minor version. Core currently accepts
`>=0.5.0,<0.6.0`; changes to that range require core integration and differential validation.
Release the spec wheel before a core release requiring a spec version that is not yet published.

Document format, specification edition, developer extensions, and conformance profiles are
separate identities in `core_pdf_spec.standards`. `PdfVersion` recognizes PDF 1.0 through 1.7
and 2.0 and retains unknown numeric declarations. `SemanticContext` supplies the effective
version and all extensions to version-sensitive algorithms. Recognition is not a claim of
complete feature support: `get_extension_coverage` lists only audited functionality for exact
prefix/base/level/revision combinations, and `STANDARD_PROFILES` is an identity catalog, not
a list of native validators.

The implementation reference is ISO 32000-2:2020 with the PDF Association errata pinned at
[`223821c58055f6ed2b93a5f40304c25ffa86459f`](https://github.com/pdf-association/pdf-issues/tree/223821c58055f6ed2b93a5f40304c25ffa86459f).
This records the reference baseline for implemented rules; it does not assert implementation
of every erratum. PDF 2.0's 2017 and 2020 editions use the same document version; earlier
references remain available in the [PDF specification archive](https://pdfa.org/resource/pdf-specification-archive/).
The catalog includes PDF/A, PDF/UA, WTPDF and PDF/X identities independently of any optional
validation backend's capabilities.

`s_07_document.standards` exposes strict header/catalog/extension declaration parsers and
`effective_pdf_version(header, catalog, previous=...)`. The latter preserves the highest
declared version across revisions; callers supply the preceding effective version. Parsing
helpers do not search for displaced headers, resolve indirect extension entries, or recover
invalid values. `parse_extensions(..., context=...)` additionally applies document-version
constraints. Reader orchestration and recovery belong to the caller.

```python
from core_pdf_spec.s_07_syntax_primitives.text_string import decode_pdf_text_string
from core_pdf_spec.standards import PdfVersion, SemanticContext

context = SemanticContext(PdfVersion.parse("2.0"))
assert decode_pdf_text_string(b"\xef\xbb\xbfHello", context=context) == "Hello"
```

UTF-16BE BOM text strings were introduced in PDF 1.2 and UTF-8 BOM text strings in PDF 2.0.
Before each introduction, those bytes use PDFDocEncoding. Omitting context preserves the
existing all-encodings API, while an explicit
unknown context raises `PdfUnsupportedError` instead of guessing a standard. `ObjectResolver`
accepts `semantic_context=` and exposes the same property so callers can initialize it after
security and catalog resolution. Changing it invalidates parsed object and object-stream
caches; callers must finish active parsing first. Object-specific algorithm selectors (such as encryption
`V` and `R`) remain authoritative.

`PdfLexer`, `ObjectStream`, `XRefScanner` parsing methods, and `ContentInterpreter`
also accept `semantic_context=`. Their names use literal `#` through PDF 1.1 and
hexadecimal escapes from PDF 1.2; strict whitespace includes NUL from PDF 1.3.
Standalone scanners accept the immutable result of `lexical_rules(context)`.
`PdfLexer.select_lexical_rules` and `ContentInterpreter.create_lexer` are public
extension points for reader adapters, alongside resolver `decode_text`.

`s_14_structure.roles.resolve_structure_role` returns a namespace-aware terminal
role and mapping path, including explicit cycle status. It follows root and
namespace maps without assuming that a PDF 2.0 file uses the PDF 2.0 tag namespace.
`s_11_transparency.blend` provides scalar and array ColorDodge/ColorBurn equations
selected by version and the exact audited Adobe extension identity; absent context
selects PDF 2.0. Source image helpers preserve 16-bit samples, and
`s_11_transparency.images` supplies strict Matte unblending. Codec selection and
output raster conversion stay in core.

`s_08_graphics.pdf_function.compile_pdf_function` supports Type 4 calculator streams using
the shared PDF 1.3+ semantics for all 42 operators, evaluated by `core_postscript.calculator`. The compiler accepts PDF numeric syntax,
comments, and conditional blocks, clips inputs to Domain and outputs to Range, and requires
exact numeric output arity. Evaluation uses finite float64 reals and signed 32-bit integers;
oversized integer literals and applicable arithmetic results promote to reals. The limits
are 100 operand-stack entries and 255 nested brace levels, including the outer braces.
Blocks are conditional syntax, not manipulable procedure objects. Compiled evaluators are
reusable and need no external runtime. Function types 0, 2, and 3 remain supported within
their existing limits: sampled functions require 8-bit samples and linear interpolation,
including the prescribed fallback for small cubic tables. This is not complete PDF function
conformance validation.

`s_11_transparency.groups.remove_group_backdrop` removes a transparency group's initial
backdrop using independently accumulated source alpha. It returns unquantized, unclipped
source components and alpha; callers own compositing state, raster storage, and output gamut
clipping. `ContentStreamFrame.group_isolated` preserves Form isolation for semantic sinks.
Actual transparency Forms default to non-isolated; legacy explicit `group_alpha` frames
retain their isolated default, and executor override signatures are unchanged.
`ContentStreamFrame.group_knockout` independently retains Form `/K`, defaulting to false.
`composite_knockout_element` in the same transparency module combines an element rendered
against the initial backdrop with previous group contributions, using separate shape and
source alpha. Its results are unquantized and unclipped; raster coverage and rounding remain
the caller's responsibility. `GraphicsState.alpha_is_shape` preserves `/AIS` through saved
and nested state; tiling pattern definitions retain their initial stream's value separately
from the alpha source selected when painting the pattern.

`GraphicsState.text_knockout` implements the initial `/TK true` value and saved-state behavior
from ISO 32000-2 §9.3.8. The interpreter ignores `/TK` entries inside `BT`/`ET`, including
invalid values and their indirect references, while applying other supported graphics-state
changes. Text-object scope is separate from `q`/`Q` and is restored across child streams.
`TilingPattern.text_knockout` retains the defining stream's initial value. Existing
`ContentSink.text_boundary` callbacks also emit `type3-glyph-begin` and `type3-glyph-end`
around each executed `CharProc`, including cleanup after a failed glyph stream. Sinks own
the implicit text/glyph compositing groups and deferred text clipping; these boundaries do
not reset graphics state. Type 3 modes 3 and 7 suppress program paint while text advances
remain active, following the shared contemporary rule in ISO 32000-2 §9.3.6. Type 3 glyphs
do not contribute outlines to text clipping. ISO 32000-1 named only mode 3; this correction
does not introduce a document-header branch.

`ContentStreamFrame.form_bbox` retains normalized, resolved local Form bounds for sinks that
need the transformed clipping quadrilateral. `clip_bbox` remains the enclosing transformed
rectangle, while `form_bbox_operand` preserves the original operand. The new field defaults
to `None` for legacy frames and does not change resolver or executor override signatures.

`s_11_transparency.soft_masks` exports immutable `SoftMask` descriptors and `parse_soft_mask`.
Descriptors retain the original transparency Form, the invoking CTM at `gs`, the transfer
function, and Luminosity backdrop/colour-space metadata. The Form Matrix is concatenated
when the group executes. Parsing preserves group/resource identity without traversing the
resource graph; Alpha ignores `/BC`, and transfer results are clipped to `[0, 1]`. `/None`
clears the mask, while omitted/null graphics-state entries preserve it. Graphics saves retain
the mask, and transparency Form entry resets it alongside alpha constants and blend mode.
Group execution and conversion to a raster mask remain the consumer's responsibility.
`ContentInterpreter.resolve_soft_mask` is an additive parsing extension for supplying function
compilation or recovery through `parse_soft_mask(..., compile_function=...)`; strict defaults
support function types 0, 2, 3, and 4 within the limits of `compile_pdf_function`, including
Type 4 Alpha-mask transfer functions without an injected compiler.
`ContentStreamExecutor.consume_frame` executes a prepared frame, including Form bounds and
transparency state, through the same error hooks and cleanup as `consume`. Existing parser
and executor signatures remain unchanged.

`s_07_document.page.page_user_unit` supplies strict page-local unit parsing with the
prescribed default; applications own physical geometry and raster-size limits.
`s_09_fonts.helpers.get_base_encoding_glyph_names` and
`build_simple_encoding_glyph_names` accept `context=` for historical WinAnsi assignments.
Explicit pre-1.3 PDFDocEncoding rejects its undefined `A0` slot instead of assigning
a later character. Omitting context preserves existing low-level behavior.

`s_08_graphics.color_rendering` defines immutable `ColorRendering` state and strict
rendering-intent/black-point-compensation helpers. The content interpreter carries this
state through saved and nested scopes; `ImageSource` retains it for deferred conversion.
`color_math` exposes XYZ/Lab conversion and the black-point endpoint scaling equation.
Output profiles, black-point detection, CMS selection, and raster conversion belong to
applications, without adding a color-management backend dependency to spec.

`s_08_graphics.color_spec.parse_color_space(..., context=...)` enforces the PDF 1.3
change permitting Separation/DeviceN bases in Indexed color spaces, recursively through
nested spaces. It does not enforce every feature's introduction version.

`ColorSpace.devicen_attributes` retains typed `DeviceNAttributes` and `DeviceNProcess`
metadata from ISO 32000-2:2020, 8.6.6.5 and Tables 70-71. The public
`color_spec.parse_device_n_attributes(attributes, colorants, *, context=None)` helper
checks this metadata independently of the outer alternate space and tint transform.
Process `component_indices` map process-space order to DeviceN input positions;
omitted CMYK channels have `None` indices. NChannel requires complete, contiguous,
naturally ordered non-CMYK process components and matching Separation definitions for
all spot components. Process definitions take precedence over `Colorants` entries.
The original attributes remain in `ColorSpace.params["Attributes"]`; optional mixing
hints remain opaque and are not compiled or fully validated. This API parses metadata
without choosing a blending algorithm. Ordinary DeviceN `Colorants` metadata was already
defined in Adobe PDF 1.3, Table 4.20; parsing does not impose a PDF 1.6 availability gate.
`s_08_graphics.color.color_space_paints` identifies Separation `/None` and all-`/None`
DeviceN spaces that discard output, including through Indexed and uncolored Pattern
bases. Mixed DeviceN spaces retain every input component for their alternate tint transform.

When migrating to `0.5.0`, import the referenced-standard kernels from their own packages;
the spec modules keep only the PDF wrappers:

- `s_07_filters.predictors` keeps `apply_predictor`, `apply_tiff_predictor`,
  `apply_png_predictor`, and `SUPPORTED_PREDICTOR_BITS`; `png_predict`, `tiff_predict*`,
  `PredictorError`, and `UnsupportedPngFilterError` are in `core_predictors.{png,tiff,errors}`,
  and the sub-byte unpacker is public as `core_predictors.samples.unpack_subbyte_rows`.
- `s_08_graphics.calculator` keeps the Domain/Range reader; the language is
  `core_postscript.calculator.compile_calculator(source, domains, ranges)`.
- `s_07_filters.jbig2` is now a module exporting `decode_jbig2`; the decoder, segment
  parsers, and bitmap kernels are `core_jbig2.codec` and `core_jbig2.bitmap`.
  `JBIG2PageDecoder.finish()` returns T.88 polarity (1 = black); `decode_jbig2` applies the
  ISO 32000-1 7.4.7 inversion through `core_jbig2.bitmap.invert_packed_bitmap`.
- `s_07_security.ciphers` and `s_07_security.pdf_mac` keep the PDF wrappers over
  `core_pdf_crypto.ciphers` (public `aes_*`, `rc4_crypt`) and `core_pdf_crypto.pdf_mac`
  (`validate_pdf_mac_token`, `validate_authenticated_data`, `digest*`, `parse_der`).
  Unsupported digest algorithms raise `core_pdf_crypto.errors.UnsupportedAlgorithmError`,
  which the wrapper maps to `PdfUnsupportedError`.
- `s_09_fonts.font_program`, `font_program_type1`, `cmap_tokenizer`, `cmap_ranges`,
  `cmap_decoder`, `cmap_resources`, `glyphs`, `data.core14`, `data.zapf_dingbats`, and
  `_vendor.font_data` moved to `core_adobe_fonts.{cff.font, cff.charstrings, type1.program,
  cmap.tokenizer, cmap.ranges, cmap.decoder, cmap.resources, agl.mapping, afm.core14,
  agl.glyph_list, agl.zapf_dingbats, _vendor.font_data}`; the three base encoding name
  tables are re-exported by `core_adobe_fonts.encodings`. `CFFFont.font_matrix` returns
  `CffFontMatrix`, a plain six-float tuple, instead of `s_08_graphics.matrix.Matrix`;
  `internal_validate_pdf_codespace` is public as `cmap.ranges.validate_effective_codespace`.
  `s_09_fonts.cmap_tounicode`, `font_program_truetype`, `dictionaries`, `widths`, `metrics`,
  `helpers`, `service`, and `data.base_encodings` stay in spec.

When migrating to `0.4.0`:

- Construct `ObjectResolver(data, xref, decipher=...)` without the removed `trailer` argument;
  retain the trailer in the calling application instead of accessing `resolver.trailer`.
- Remove access to `JBIG2MQDecoder.ctx`, an unused attribute that has been removed.
- Use `CMapDecoder.decode_entries(bytes(data))` and extract the CIDs from the returned
  `(code_bytes, cid)` pairs in place of the removed `decode_cids_array` method.

From the workspace, run the standalone tests with:

```sh
uv run --locked --all-packages --group test pytest packages/core-pdf-spec/tests
```

To prove spec works with only its declared dependencies installed:

```sh
uv sync --locked --package core-pdf-spec --group test
uv run --locked --no-sync python scripts/check_package_isolation.py core-pdf-spec
uv run --locked --no-sync pytest packages/core-pdf-spec/tests
```
