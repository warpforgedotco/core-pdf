# core-pdf-crypto

RC4, AES-CBC, and AES-GCM (ISO/TS 32003) cipher primitives and ISO/TS 32004 PDF MAC validation over RFC 5652 CMS AuthenticatedData, RFC 5869 HKDF, and RFC 3394 key wrap. Takes bytes; knows no PDF objects.

This distribution is a workspace floor package of `core-pdf`: it imports nothing from
`core-pdf`, `core-pdf-spec`, `core-pdf-ocr`, or `core-pdf-validate`. It may import another floor package it declares as a
dependency, such as `core-records` for its value classes. PDF-specific glue (stream parameters, PDF objects, PDF exceptions) lives
in the `core-pdf-spec` chapter that ISO 32000 assigns.

Run its tests without the rest of the workspace installed:

```sh
uv sync --locked --package core-pdf-crypto --group test
uv run --locked --no-sync pytest packages/core-pdf-crypto/tests
```
