# Specification fixtures

Specifications serve two distinct purposes here: as **normative references** for
the subsystems the engine implements, and as **parser fixtures** — large,
real-world PDFs produced by the tooling of their era.

They are split into three tiers by redistribution rights. Fetch a tier with
`./scripts/fetch_pdf_specs.sh <tier>`, or all of them with no argument.

The PDF Association's documents sit behind Cloudflare, which rejects curl even
for direct PDF URLs, so they are fetched by `scripts/fetch_pdfa_docs.py` using
Playwright. `fetch_pdf_specs.sh` delegates to it; run
`uvx --from playwright playwright install chromium` first. Those documents land
in `reference/pdfa/` or `restricted/pdfa/` according to their own licences.

| Tier | Committed? | Why |
| --- | --- | --- |
| `reference/` | **yes** | CC BY-ND / BSD / IETF / Unicode / W3C — verbatim redistribution permitted |
| `PDF/` | no | Adobe & ISO reserve all reproduction rights |
| `restricted/` | no | ITU-T and ICC distribute freely but reserve copyright |

`core-pdf` is a public AGPL-3.0 repository, so committing a document means
redistributing it. Only the `reference/` tier is licensed for that. The other
two tiers are gitignored and rebuilt on demand from their canonical sources.

Before adding anything, read the actual copyright notice in the document. The
PDF References carry a section titled "Copyright Permission" that grants the
right to *implement* PDF and not the right to redistribute the specification —
see `PDF/README.md`.

## A note on producer diversity

All eight documents in `PDF/` are FrameMaker → Acrobat Distiller output spanning
1993–2008. That is a genuine axis of coverage (Distiller 3 through 8), and one
of them, ISO 32000-1, is RC4-encrypted and so exercises `s_07_security`. But it
is a single producer lineage. Fixtures from pdfTeX, Word, Ghostscript, Chrome
print-to-PDF and scanner output would each exercise materially different code
paths; the `restricted/` tier incidentally adds some non-Adobe producers.
