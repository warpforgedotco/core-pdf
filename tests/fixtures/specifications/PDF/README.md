# PDF specification fixtures

Real-world, large, structurally diverse PDFs spanning every published version of
the format -- useful as parser fixtures precisely because each document was
produced by the tooling of its own era.

Populate this directory with:

```sh
./scripts/fetch_pdf_specs.sh
```

Fetch only the two public documents exercised by the security specification
evidence tests with `./scripts/fetch_pdf_specs.sh security`.

## These files are deliberately not committed

Every document below is Adobe- or ISO-copyrighted and **none of them grant
redistribution rights**. Adobe serves them publicly at no cost, so downloading a
working copy is fine, but committing them into this (public, AGPL-3.0)
repository would be redistribution. Hence `.gitignore` and the fetch script.

The "Copyright Permission" clause carried in PDF Reference 1.3 through 1.7 is
easy to misread: it grants permission to *implement* PDF -- to write producers
and consumers, and to copy the operator/data-structure tables and example code
"to the extent necessary" to do so. It grants nothing with respect to
redistributing the specification document itself, and the front matter of each
edition states the opposite explicitly:

> No part of this publication may be reproduced, stored in a retrieval system,
> or transmitted, in any form or by any means [...] without the prior written
> consent of the publisher.

ISO 32000-1:2008 is Adobe's authorized copy of the ISO standard and is likewise
restricted: "Requests for permission to reproduce this document for any purpose
should be arranged with ISO."

## Contents

| Fixture | Spec | Year | Copyright |
| --- | --- | --- | --- |
| `PDFReference-1.0-Adobe-1993.pdf` | PDF 1.0 | 1993 | Adobe |
| `PDFReference-1.2-Adobe-1996.pdf` | PDF 1.2 | 1996 | Adobe |
| `PDFReference-1.3-Adobe-2000.pdf` | PDF 1.3 | 2000 | Adobe |
| `PDFReference-1.4-Adobe-2001.pdf` | PDF 1.4 | 2001 | Adobe |
| `PDFReference-1.5-Adobe-2003.pdf` | PDF 1.5 | 2003 | Adobe |
| `PDFReference-1.6-Adobe-2004.pdf` | PDF 1.6 | 2004 | Adobe |
| `PDFReference-1.7-Adobe-2006.pdf` | PDF 1.7 | 2006 | Adobe |
| `ISO32000-1-2008-PDF-1.7.pdf` | ISO 32000-1 (PDF 1.7) | 2008 | ISO, authorized Adobe copy |
| `ISO32000-2-2020-PDF-2.0-EC3.pdf` | ISO 32000-2 (PDF 2.0), Errata Collection 3 | 2020 | ISO, sponsored access |
| `ISO-TS-32001-2022-Hash-Algorithms-EC3.pdf` | ISO/TS 32001 — hash algorithms (SHA-3) | 2022 | ISO, sponsored access |
| `ISO-TS-32002-2022-Digital-Signatures-EC3.pdf` | ISO/TS 32002 — digital signatures (EdDSA) | 2022 | ISO, sponsored access |
| `ISO-TS-32003-2023-AES-GCM.pdf` | ISO/TS 32003 — AES-GCM encryption | 2023 | ISO, sponsored access |
| `ISO-TS-32004-2024-Integrity-Protection.pdf` | ISO/TS 32004 — integrity protection | 2024 | ISO, sponsored access |
| `ISO-TS-32005-2023-Structure-Namespaces.pdf` | ISO/TS 32005 — structure namespaces | 2023 | ISO, sponsored access |

PDF 1.1 is not among them: Adobe does not host that edition, and no
authoritative copy is publicly served. Every other published version of PDF,
1.0 through 2.0, is present.

## Not obtainable

- **The ISO 32000-2 bundle is a manual step.** The PDF Association sponsors free
  access to PDF 2.0 and its five extensions, but only through a cart and
  checkout on <https://www.pdfa-inc.org/product/iso-32000-2-pdf-2-0-bundle-sponsored-access/>,
  which cannot be scripted. The bundle is already in place here; re-download it
  by hand if this directory is ever rebuilt from scratch. ISO reserves all
  rights over these documents -- "no part of this publication may be reproduced"
  -- so they are gitignored like everything else in this tier.
- **The conformance subsets** -- PDF/A (ISO 19005), PDF/X (ISO 15930), PDF/UA
  (ISO 14289), PDF/E (ISO 24517), PDF/VT (ISO 16612), PDF/raster (ISO 23504).
  All are sold by ISO at roughly CHF 200 per part and none may be redistributed.

For freely-licensed PDF 2.0 sample files that *are* committed, see the
`tests/fixtures/pdf20examples` submodule (CC BY-SA 4.0, PDF Association).
