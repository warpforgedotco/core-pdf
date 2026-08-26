#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
"""Fetch the PDF Association's specifications and technical notes.

pdfa.org sits behind Cloudflare, which rejects plain HTTP clients even for
direct PDF URLs, so these documents need a real browser engine. Everything
else is fetched by scripts/fetch_pdf_specs.sh with curl.

Documents are split by redistribution rights, matching the tiers described in
tests/fixtures/specifications/README.md:

  reference/pdfa   CC BY 4.0 -- redistributable, committed to the repo
  restricted/pdfa  no redistribution grant -- gitignored

Usage:
    uv run --with playwright python scripts/fetch_pdfa_docs.py [reference|restricted|all]
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

SPECS_ROOT = Path(__file__).resolve().parent.parent / "tests/fixtures/specifications"
PUB = "https://pdfa.org/download-area/publications"
SPEC = "https://pdfa.org/download-area/specifications"
UPL = "https://pdfa.org/wp-content/uploads"

# tier -> {local filename: url}
DOCUMENTS: dict[str, dict[str, str]] = {
    # CC BY 4.0. Attribution is recorded in reference/NOTICE.md.
    "reference": {
        "Matterhorn-Protocol-1.1.pdf": f"{PUB}/Matterhorn-Protocol-1-1.pdf",
        "Tagged-PDF-Best-Practice-Guide-Syntax.pdf": f"{PUB}/Tagged-PDF-Best-Practice-Guide.pdf",
        "Well-Tagged-PDF-WTPDF-1.0.pdf": f"{SPEC}/Well-Tagged-PDF-WTPDF-1.0.pdf",
        "Conforming-to-PDFA-and-PDFUA.pdf": f"{PUB}/Conforming-to-both-PDFA-&#038;-PDFUA.pdf",
        "PDF-Declarations-1.0.pdf": f"{SPEC}/PDF-Declarations.pdf",
        "BPG-Math-in-PDF.pdf": f"{PUB}/BPG-Math-in-PDF.pdf",
        "Custom-Metadata-Structures-in-PDF.pdf": f"{PUB}/Including-custom-metadata-structures-in-PDF.pdf",
        "PDF-Extension-Brotli.pdf": f"{PUB}/pdf-extension-brotli.pdf",
        "EA-PDF-v1.pdf": f"{SPEC}/EA-PDF-v1.pdf",
    },
    # The PDF/A Competence Center TechNotes state that redistribution requires
    # written approval; the WTPDF examples carry no licence grant at all.
    "restricted": {
        "TN0001-PDFA1-and-Namespaces.pdf": f"{UPL}/2011/08/tn0001_pdfa-1_and_namespaces_2008-03-182.pdf",
        "TN0003-Metadata-in-PDFA1.pdf": f"{UPL}/2011/08/tn0003_metadata_in_pdfa-1_2008-03-182.pdf",
        "TN0008-Predefined-XMP-Properties.pdf": f"{UPL}/2011/08/tn0008_predefined_xmp_properties_in_pdfa-1_2008-03-20.pdf",
        "TN0009-XMP-Extension-Schemas.pdf": f"{UPL}/2011/09/tn0009_xmp_extension_schemas_in_pdfa-1_2008-03-20.pdf",
        "TN0010-Clarifications-ISO19005.pdf": f"{UPL}/2017/07/TechNote0010.pdf",
        "example-WTPDF-apryse-itext.pdf": "https://pdfa.org/download-area/examples/WTPDF/apryse-itext-wtpdf.pdf",
        "example-WTPDF-BFO-PDFUA2.pdf": "https://pdfa.org/download-area/examples/WTPDF/2024-03-04_BFO-WTPDF-PDFua2.pdf",
    },
}

# Cloudflare's rules for pdfa.org are picky about this exact string: the
# headless default and a "Chrome/126.0.0.0" build number are both rejected with
# a 403, while "Chrome/126" is served. If these downloads start failing, this
# line is the first thing to check.
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126 Safari/537.36"
)


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "playwright is not installed. Run:\n"
            "  uvx --from playwright playwright install chromium\n"
            "  uv run --with playwright python scripts/fetch_pdfa_docs.py",
            file=sys.stderr,
        )
        return 2

    tier = sys.argv[1] if len(sys.argv) > 1 else "all"
    wanted = DOCUMENTS if tier == "all" else {tier: DOCUMENTS.get(tier, {})}

    fetched = cached = failed = 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(user_agent=UA)
        # Load the site once so the Cloudflare clearance cookie is set. The page
        # must stay open: closing it drops the clearance and every request 403s.
        page = ctx.new_page()
        page.goto("https://pdfa.org/", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)

        for tier_name, docs in wanted.items():
            dest = SPECS_ROOT / tier_name / "pdfa"
            dest.mkdir(parents=True, exist_ok=True)
            for name, url in docs.items():
                target = dest / name
                if target.exists():
                    cached += 1
                    continue
                try:
                    resp = ctx.request.get(html.unescape(url), timeout=120000)
                    body = resp.body()
                except Exception as exc:  # noqa: BLE001 - report and continue
                    print(f"  ERR  {name}: {type(exc).__name__}", file=sys.stderr)
                    failed += 1
                    continue
                if not resp.ok or not body.startswith(b"%PDF-"):
                    print(f"  WARN {name}: status={resp.status}, not a PDF", file=sys.stderr)
                    failed += 1
                    continue
                target.write_bytes(body)
                print(f"==> {tier_name:11s} {name}")
                fetched += 1
                page.wait_for_timeout(1500)
        browser.close()

    print(f"\ntier={tier}: {fetched} fetched, {cached} cached, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
