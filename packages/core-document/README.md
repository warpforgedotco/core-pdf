# core-document

Format-neutral immutable document records and derived JSON, Markdown, and semantic HTML views.

PDF parsing and extraction orchestration remain in `core-pdf`.

`DocumentAdapter` provides an optional post-extraction enrichment boundary. Adapters
receive an immutable `Document` and return a new one, so OCR and other specialized
providers can remain separate packages.

`Document.edit()` opens a transactional IR editor. Changes are isolated until
`commit()` and can be discarded with `rollback()`; PDF byte serialization is a
separate writer layer owned by the format-specific package.
