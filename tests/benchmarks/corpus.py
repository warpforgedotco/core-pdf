from pathlib import Path
from typing import NamedTuple

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"


class Sample(NamedTuple):
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
