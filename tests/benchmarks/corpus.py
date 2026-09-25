"""A profiling corpus for core-pdf, chosen by measurement rather than by size.

Every one of the 795 PDFs under `tests/fixtures` was opened and had its first
pages extracted; the results were ranked by cost and grouped by content shape.
File size turned out to be a poor predictor: the corpus spans two orders of
magnitude per page (p50 62 ms, p95 362 ms, max 5.6 s), and an 80 KB file can
cost more per page than a 7.9 MB one. Content shape predicts it much better --
text-heavy pages run a median 195 ms, sparse pages 12 ms.

The samples below are all well-formed: no xref recovery, no page-tree recovery,
page_class "native". They measure real work rather than error-recovery paths.

`native_ms` is the approximate cost of the benchmarked work on the profiling
VM. It is a budgeting aid, not an assertion -- callgrind runs roughly 45x
slower than native, so a 500 ms sample costs about 22 s of simulation. That
factor is why the heaviest documents are listed in DEEP_DIVE and left out of
the default suite: extracting all 619 pages of DA-619p takes 346 s natively,
which is over four hours under simulation.
"""

from pathlib import Path
from typing import NamedTuple

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"


class Sample(NamedTuple):
    """One corpus entry, with the measurement that earned it a place."""

    path: str
    shape: str
    native_ms: int
    note: str

    @property
    def file(self) -> Path:
        return FIXTURE_ROOT / self.path

    @property
    def id(self) -> str:
        return f"{self.shape}-{Path(self.path).stem}"[:48]


# Document open only: xref parsing, trailer, page tree. No content extraction.
OPEN_SAMPLES = (
    Sample(
        "llama_index/docs/examples/query_engine/pdf_tables/billionaires_page.pdf",
        "xref",
        822,
        "worst open cost in the corpus, 4.7x the next file",
    ),
    Sample(
        "unstructured/example-docs/pdf/pdf2image-memory-error-test-400p.pdf",
        "pagetree",
        115,
        "400 pages; open cost that scales with the page tree",
    ),
    Sample(
        "pdf20examples/Simple PDF 2.0 file.pdf",
        "control",
        1,
        "trivial document; catches harness noise and startup regressions",
    ),
)

# First-page extraction, chosen to span the shapes that drive cost.
EXTRACT_SAMPLES = (
    Sample(
        "llama_index/docs/examples/data/10k/lyft_2021.pdf",
        "text",
        93,
        "2648 chars/pg across 238 pages; the dominant real-world shape",
    ),
    Sample(
        "pdfminer.six/samples/nonfree/i1040nr.pdf",
        "text",
        152,
        "6344 chars/pg, the densest text in the corpus",
    ),
    Sample(
        "SCORE-Bench/src/fhhd0346-p009.pdf",
        "tables",
        554,
        "8 tables on one 60 KB page; isolates table detection",
    ),
    Sample(
        "llama_index/docs/examples/query_engine/pdf_tables/billionaires_page.pdf",
        "tables",
        191,
        "tables plus the heaviest open in the corpus",
    ),
    Sample(
        "pypdf/resources/issue-301.pdf",
        "anomaly",
        1231,
        "120 KB and 93 chars/pg, yet 1.2 s across 21 blocks",
    ),
    Sample(
        "PyMuPDF/tests/resources/test_3789.pdf",
        "sparse",
        46,
        "21 chars/pg; a near-empty page isolates per-page fixed overhead",
    ),
    Sample(
        "pdf20examples/Simple PDF 2.0 file.pdf",
        "control",
        16,
        "trivial page; the floor for any extraction change",
    ),
)

# Multi-page extraction, for costs that only appear across pages: shared font
# and resource caches, and whatever state survives a page boundary.
SLICE_PAGES = 3
SLICE_SAMPLES = (
    Sample(
        "llama_index/docs/examples/data/10k/lyft_2021.pdf",
        "text",
        279,
        "three consecutive text-dense pages",
    ),
    Sample(
        "unstructured/example-docs/pdf/pdf2image-memory-error-test-400p.pdf",
        "mixed",
        174,
        "three pages of moderate content from a 400-page document",
    ),
)

# Rendering, which until now had no benchmark at all. It is a sibling of
# extraction rather than a stage of it -- both consume the captured page
# program, and only rendering turns it into pixels -- so a change can move one
# and not the other, and several already have.
#
# The set is deliberately small, and smaller than the extraction one. A
# recorded run is dominated by per-benchmark valgrind startup rather than by
# sample cost, so the count is what to economise on: three shapes, each
# measured twice (compose and rasterize), is six more benchmarks against the
# suite's twelve. lyft_2021 is left out because i1040nr covers the same text
# shape more densely.
RENDER_SAMPLES = (
    Sample(
        "pdfminer.six/samples/nonfree/i1040nr.pdf",
        "text",
        420,
        "densest text in the corpus; dominated by glyph outline fills",
    ),
    Sample(
        "llama_index/docs/examples/query_engine/pdf_tables/billionaires_page.pdf",
        "tables",
        274,
        "rules and fills rather than glyph outlines",
    ),
    Sample(
        "SCORE-Bench/src/NASA-SNA-8-D-027III-Rev2-CsmLmSpacecraftOperationalDataBook-"
        "Volume3-MassProperties-Pg54.pdf",
        "jbig2",
        90,
        "a 2550x3300 JBIG2 scan; the arithmetic decoder was 90% of rasterize",
    ),
    Sample(
        "pdf20examples/Simple PDF 2.0 file.pdf",
        "control",
        8,
        "trivial page; the floor for any rendering change",
    ),
)


# Too slow for a routine suite, kept here because they are the best targets for
# a one-off profile. Run these by hand, not under the default benchmark run.
DEEP_DIVE = (
    Sample(
        "PyMuPDF/tests/resources/test_3806.pdf",
        "raster",
        5634,
        "6.4 MB, 24 figures, zero text; the slowest page in the corpus",
    ),
    Sample(
        "PyMuPDF/tests/resources/test_3362.pdf",
        "anomaly",
        4870,
        "80 KB -> 4.9 s; 8661 chars and 49 blocks on a single page",
    ),
    Sample(
        "PyMuPDF/tests/resources/test_3450.pdf",
        "anomaly",
        5426,
        "1.7 MB, 3 tables, 21 blocks -> 5.4 s",
    ),
    Sample(
        "PyMuPDF/tests/resources/test_2904.pdf",
        "figures",
        1142,
        "39 figures/pg with almost no text",
    ),
    Sample(
        "pdfminer.six/samples/nonfree/kampo.pdf",
        "cjk",
        1170,
        "30 KB -> 1.2 s of CJK text",
    ),
    Sample(
        "unstructured/example-docs/pdf/DA-619p.pdf",
        "long",
        559,
        "619 pages; 346 s for the whole document, so always slice it",
    ),
)
