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

The current spec version is `0.4.0`, released independently of core. During `0.x`, breaking
changes to supported interfaces require a new minor version. Core currently accepts
`>=0.4.0,<0.5.0`; changes to that range require core integration and differential validation.
Release the spec wheel before a core release requiring a spec version that is not yet published.

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
