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

The library contains implemented PDF semantics and algorithms from referenced standards,
including bundled CMaps and standard font tables. It preserves standard-defined defaults and
fallback rules. Malformed-input repair, retry/skip policy, substitute fonts, fontTools backends,
Unicode guesses, output-device choices, capture products, and rasterization belong to
applications such as `core-pdf`. Semantic sinks, stream decoders, and font providers are supplied
through typed interfaces. Unsupported algorithms report failure instead of selecting a backend.

There is no document facade, CLI, or complete conformance-validator claim. Core and OCR are
not dependencies and are never discovered or imported. Runtime dependencies are NumPy,
cryptography, and asn1crypto; bundled font data retains its original notices.

For Lab color conversion, `s_08_graphics.color_math.lab_components_to_xyz` accepts
NumPy float32 rows of actual `(L*, a*, b*)` components and a reference white point.
Image sample decoding and range enforcement belong to the caller. The existing
`lab_to_xyz` function retains its normalized input convention as a compatibility wrapper.

The current spec version is `0.4.1`, released independently of core. During `0.x`, breaking
changes to supported interfaces require a new minor version. Core currently accepts
`>=0.4.1,<0.5.0`; changes to that range require core integration and differential validation.
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

`s_08_graphics.color_spec.parse_color_space(..., context=...)` enforces the PDF 1.3
change permitting Separation/DeviceN bases in Indexed color spaces, recursively through
nested spaces. It does not enforce every feature's introduction version. The repository's
[version coverage audit](../../docs/version-coverage.md) distinguishes implemented rules,
shared algorithms, and remaining historical or feature gaps.

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
