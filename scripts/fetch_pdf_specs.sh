#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Fetch the specification documents used as parser fixtures and normative
# references, in three tiers distinguished by redistribution rights:
#
#   reference   redistributable, COMMITTED to the repo (CC BY-ND / BSD /
#               IETF Trust / Unicode / W3C terms -- verbatim copies allowed)
#   pdf         the core PDF specs; free to download, NOT redistributable,
#               so gitignored (see tests/fixtures/specifications/PDF/README.md)
#   restricted  ITU-T and ICC specs; free to download, NOT redistributable,
#               so gitignored
#
# The PDF Association's documents live behind Cloudflare, which rejects curl even
# for direct PDF URLs, so they are fetched separately by scripts/fetch_pdfa_docs.py
# (Playwright). This script delegates to it.
#
# Usage: scripts/fetch_pdf_specs.sh [all|reference|pdf|restricted|pdfa]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPECS_ROOT="${PROJECT_ROOT}/tests/fixtures/specifications"

TIER="${1:-all}"

ADOBE="https://opensource.adobe.com/dc-acrobat-sdk-docs"
FONTNOTES="https://adobe-type-tools.github.io/font-tech-notes/pdfs"
AGL="https://raw.githubusercontent.com/adobe-type-tools/agl-aglfn/master"
RFC="https://www.rfc-editor.org/rfc"
UCD="https://www.unicode.org/Public/UCD/latest/ucd"
ITU="https://www.itu.int/rec/dologin_pub.asp?lang=e&type=items&id"

# tier|subdirectory|url|local filename
MANIFEST=(
  # ---- reference: redistributable, committed -------------------------------
  # Adobe font technical notes (CC BY-ND 4.0) -- CFF, charstrings, Type 1, CID.
  "reference|fonts|${FONTNOTES}/5176.CFF.pdf|TN5176-CFF.pdf"
  "reference|fonts|${FONTNOTES}/5177.Type2.pdf|TN5177-Type2-Charstring.pdf"
  "reference|fonts|${FONTNOTES}/T1_SPEC.pdf|Type1-Font-Format.pdf"
  "reference|fonts|${FONTNOTES}/5015.Type1_Supp.pdf|TN5015-Type1-Supplement.pdf"
  "reference|fonts|${FONTNOTES}/5014.CIDFont_Spec.pdf|TN5014-CMap-CIDFont.pdf"
  "reference|fonts|${FONTNOTES}/5094.CJK_CID.pdf|TN5094-CJK-Collections.pdf"
  "reference|fonts|${FONTNOTES}/5180.sfnt.pdf|TN5180-sfnt.pdf"
  # Adobe Glyph List (BSD-3-Clause) -- glyph-name to Unicode resolution.
  "reference|glyph-lists|${AGL}/glyphlist.txt|glyphlist.txt"
  "reference|glyph-lists|${AGL}/aglfn.txt|aglfn.txt"
  "reference|glyph-lists|${AGL}/LICENSE.md|LICENSE-AGL.md"
  # RFCs (IETF Trust, unlimited redistribution) -- Flate filter, signatures.
  "reference|rfc|${RFC}/rfc1950.txt|rfc1950-zlib.txt"
  "reference|rfc|${RFC}/rfc1951.txt|rfc1951-deflate.txt"
  "reference|rfc|${RFC}/rfc5652.txt|rfc5652-cms.txt"
  "reference|rfc|${RFC}/rfc3161.txt|rfc3161-timestamp.txt"
  # Unicode character data (Unicode License) -- text segmentation, ToUnicode.
  "reference|unicode|${UCD}/UnicodeData.txt|UnicodeData.txt"
  "reference|unicode|${UCD}/LineBreak.txt|LineBreak.txt"
  "reference|unicode|https://www.unicode.org/license.txt|LICENSE-Unicode.txt"
  # PNG (W3C Document License) -- the predictor algorithms used by FlateDecode.
  "reference|png|https://www.w3.org/TR/png-3/|png-3rd-edition.html"

  # ---- pdf: free download, not redistributable, gitignored -----------------
  "pdf|.|${ADOBE}/pdfstandards/pdfreference1.0.pdf|PDFReference-1.0-Adobe-1993.pdf"
  "pdf|.|${ADOBE}/pdfstandards/pdfreference1.2.pdf|PDFReference-1.2-Adobe-1996.pdf"
  "pdf|.|${ADOBE}/pdfstandards/pdfreference1.3.pdf|PDFReference-1.3-Adobe-2000.pdf"
  "pdf|.|${ADOBE}/pdfstandards/pdfreference1.4.pdf|PDFReference-1.4-Adobe-2001.pdf"
  "pdf|.|${ADOBE}/pdfstandards/pdfreference1.5_v6.pdf|PDFReference-1.5-Adobe-2003.pdf"
  "pdf|.|${ADOBE}/pdfstandards/pdfreference1.6.pdf|PDFReference-1.6-Adobe-2004.pdf"
  "pdf|.|${ADOBE}/pdfstandards/pdfreference1.7old.pdf|PDFReference-1.7-Adobe-2006.pdf"
  "pdf|.|${ADOBE}/standards/pdfstandards/pdf/PDF32000_2008.pdf|ISO32000-1-2008-PDF-1.7.pdf"

  # ---- restricted: free download, not redistributable, gitignored ----------
  # ITU-T recommendations behind the image filters. T.81 is taken from the
  # long-standing W3C mirror; itu.int serves no scriptable PDF for it.
  "restricted|itu-t|https://www.w3.org/Graphics/JPEG/itu-t81.pdf|T.81-JPEG.pdf"
  "restricted|itu-t|${ITU}=T-REC-T.88-200002-S!!PDF-E|T.88-JBIG2.pdf"
  "restricted|itu-t|${ITU}=T-REC-T.4-200307-I!!PDF-E|T.4-Group3-Fax.pdf"
  "restricted|itu-t|${ITU}=T-REC-T.6-198811-I!!PDF-E|T.6-Group4-Fax.pdf"
  "restricted|itu-t|${ITU}=T-REC-T.800-201511-S!!PDF-E|T.800-JPEG2000.pdf"
  # ICC colour profile specification -- ICCBased, Lab, CalRGB, DeviceN.
  "restricted|icc|https://www.color.org/specification/ICC1v43_2010-12.pdf|ICC.1-2010.pdf"
  "restricted|icc|https://www.color.org/specification/ICC.1-2022-05.pdf|ICC.1-2022.pdf"
)

fetched=0 cached=0 failed=0

for entry in "${MANIFEST[@]}"; do
  IFS='|' read -r tier subdir url name <<<"${entry}"
  [[ "${TIER}" == "all" || "${TIER}" == "${tier}" ]] || continue

  case "${tier}" in
    pdf) dest="${SPECS_ROOT}/PDF" ;;
    *)   dest="${SPECS_ROOT}/${tier}/${subdir}" ;;
  esac
  mkdir -p "${dest}"

  if [[ -f "${dest}/${name}" ]]; then
    cached=$((cached + 1))
    continue
  fi

  printf '==> %-12s %s\n' "${tier}" "${name}"
  if ! curl -fsSL --retry 3 -m 300 -o "${dest}/${name}.part" "${url}"; then
    rm -f "${dest}/${name}.part"
    echo "    WARN: download failed -- fetch by hand from ${url}" >&2
    failed=$((failed + 1))
    continue
  fi

  # Content check: PDFs must carry the magic, text/HTML must be non-trivial.
  ok=1
  if [[ "${name}" == *.pdf && "$(head -c 5 "${dest}/${name}.part")" != "%PDF-" ]]; then
    ok=0
  elif [[ ! -s "${dest}/${name}.part" ]]; then
    ok=0
  fi

  if (( ok )); then
    mv "${dest}/${name}.part" "${dest}/${name}"
    fetched=$((fetched + 1))
  else
    rm -f "${dest}/${name}.part"
    echo "    WARN: ${url} did not return the expected content -- fetch by hand" >&2
    failed=$((failed + 1))
  fi
done

echo
echo "tier=${TIER}: ${fetched} fetched, ${cached} cached, ${failed} failed"
(( failed == 0 )) || echo "Some documents need manual download; see the tier README."

# The pdfa.org documents need a browser engine; hand them to the Playwright
# fetcher, which sorts them into the reference and restricted tiers itself.
if [[ "${TIER}" == "all" || "${TIER}" == "pdfa" ]]; then
  echo
  echo "==> PDF Association documents (via Playwright)"
  if ! uv run --with playwright python "${SCRIPT_DIR}/fetch_pdfa_docs.py" all; then
    echo "    Some PDF Association documents were not fetched." >&2
    echo "    Ensure the browser is installed: uvx --from playwright playwright install chromium" >&2
  fi
fi
