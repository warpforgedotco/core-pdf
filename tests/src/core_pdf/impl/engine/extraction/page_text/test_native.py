from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from core_document import Document, DocumentAdapter
from core_layout.impl.layout.models import TextRun

from core_pdf.impl.engine.extraction.common.observation_resolver import ResolvedTextLine
from core_pdf.impl.engine.extraction.common.page_geometry import PageObservation
from core_pdf.impl.engine.extraction.document import PdfDocument
from core_pdf.impl.engine.extraction.page_text.engine import (
    build_page_document,
    build_page_extraction_result,
)
from core_pdf.impl.engine.extraction.page_text.native import (
    discard_overlapping_nested_xobject_runs,
    native_text_runs_for_extraction,
    normalize_fullwidth_ascii_text,
    normalize_latin_ligatures,
)
from core_pdf.impl.engine.extraction.page_text.snapshots import native_snapshot

TESTS_DIR = Path(__file__).parents[6]
SAMPLE_PDF = TESTS_DIR / "fixtures" / "SCORE-Bench" / "src" / "global-AIDS-strategy-p74-75-p001.pdf"
SNAPSHOT_DIR = TESTS_DIR / "snapshots" / "native"


def result_text(result: Any) -> str:
    return "\n".join(line.text for block in result.blocks for line in block.lines)


def image_only_pdf() -> BytesIO:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 10 10] "
            b"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>"
        ),
        (
            b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 "
            b"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Length 3 >>\n"
            b"stream\n\xff\x00\x00\nendstream"
        ),
        b"<< /Length 24 >>\nstream\nq\n10 0 0 10 0 0 cm\n/Im0 Do\nQ\nendstream",
    ]
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode())
        data.extend(obj)
        data.extend(b"\nendobj\n")
    xref_offset = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    data.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    return BytesIO(data)


def text_run(
    text: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    visible: bool = True,
    provenance: tuple[tuple[str, object], ...] = (),
) -> TextRun:
    return TextRun(
        text,
        x0,
        y0,
        x1,
        y1,
        x0,
        y0,
        10.0,
        4.0,
        0,
        0,
        0,
        visible=visible,
        provenance=provenance,
    )


def test_native_extraction_returns_pdf_text_without_external_services() -> None:
    with PdfDocument.open(SAMPLE_PDF) as document:
        page = cast(Any, document.pages[0])
        result = build_page_extraction_result(page)
        text = result_text(result)

        assert text.strip()
        assert hasattr(page, "extract_text")
        assert page.extraction_cache is not None
        assert "native_output_lines" in page.extraction_cache


def test_structured_page_result_reports_native_route() -> None:
    with PdfDocument.open(SAMPLE_PDF) as document:
        page = cast(Any, document.pages[0])
        result = build_page_extraction_result(page)

        assert result_text(result).strip()
        assert result.base_route in {"native_fast", "native_layout"}
        assert result.resolved_lines


def test_document_extract_returns_core_document_ir() -> None:
    with PdfDocument.open(SAMPLE_PDF) as document:
        result = document.extract()

    assert isinstance(result, Document)
    assert result.pages[0].width > 0
    assert result.pages[0].blocks
    assert isinstance(result.pages[0].tables, tuple)
    assert isinstance(result.pages[0].figures, tuple)
    assert isinstance(result.pages[0].links, tuple)
    assert isinstance(result.pages[0].annotations, tuple)
    assert isinstance(result.pages[0].form_fields, tuple)
    assert result.to_json_dict()["schema_version"] == "1.0"
    assert "GLOBAL AIDS STRATEGY" in result.to_markdown()
    assert 'data-schema-version="1.0"' in result.to_html()


def test_page_builder_returns_core_document_page_directly() -> None:
    with PdfDocument.open(SAMPLE_PDF) as document:
        page = cast(Any, document.pages[0])
        result = build_page_document(page)

    assert result.page_number == 1
    assert result.blocks


def test_document_extract_accepts_optional_immutable_adapters() -> None:
    class Adapter:
        def apply(self, document: Document) -> Document:
            return Document(
                pages=document.pages,
                metadata={**document.metadata, "adapter": "test"},
                diagnostics=document.diagnostics,
                schema_version=document.schema_version,
            )

    adapter: DocumentAdapter = Adapter()
    with PdfDocument.open(SAMPLE_PDF) as document:
        result = document.extract(adapters=(adapter,))

    assert result.metadata["adapter"] == "test"


def test_structured_extraction_embeds_canonical_document_ir() -> None:
    with PdfDocument.open(SAMPLE_PDF) as document:
        structured = cast(dict[str, Any], document.extract_structured())
        selected = cast(dict[str, Any], document.extract_structured(pages=[1]))

    assert structured["schema_version"] == "1.0"
    assert structured["document"]["schema_version"] == "1.0"
    assert len(structured["document"]["pages"]) == 1
    assert len(selected["document"]["pages"]) == 1


def test_document_outputs_use_canonical_extraction_ir() -> None:
    with PdfDocument.open(SAMPLE_PDF) as document:
        json_output = document.to_json()
        html_output = document.to_html()

    assert '"schema_version": "1.0"' in json_output
    assert 'data-schema-version="1.0"' in html_output


def test_page_markdown_uses_canonical_page_ir() -> None:
    with PdfDocument.open(SAMPLE_PDF) as document:
        page = cast(Any, document.pages[0])

        assert page.to_markdown() == page.extract().to_markdown()


def test_page_exposes_coordinated_text_extraction() -> None:
    with PdfDocument.open(SAMPLE_PDF) as document:
        page = cast(Any, document.pages[0])

        assert hasattr(page, "extract_text")
        assert page.extract_text()


def test_ocr_layer_extracts_image_only_text(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = TESTS_DIR / "fixtures" / "SCORE-Bench" / "src" / "Enron_Attendee_List-Grainy-p001.pdf"
    monkeypatch.setenv("CORE_PDF_OCR", "1")

    with PdfDocument.open(fixture) as document:
        result = cast(Any, document.pages[0]).extract()

    assert result.page_class == "image"
    assert "Enron Corporation" in result.text


def test_image_only_page_does_not_attempt_text_extraction() -> None:
    with PdfDocument(image_only_pdf()) as document:
        page = cast(Any, document.pages[0])
        result = build_page_extraction_result(page)

        assert result_text(result) == ""
        assert result.page_class == "image"
        assert result.resolved_lines == ()


def test_native_bounds_keep_long_text_with_oversized_font_metrics() -> None:
    run = text_run(
        "A valid line whose reported width exceeds the page",
        60.0,
        100.0,
        1_200.0,
        112.0,
    )

    from core_pdf.impl.engine.extraction.page_text.native import (
        native_text_runs_inside_page_bounds,
    )

    assert native_text_runs_inside_page_bounds([run], (0.0, 0.0, 612.0, 792.0)) == [run]


def test_native_extraction_drops_duplicate_invisible_text_layer() -> None:
    painted = text_run("Hello native PDF text", 10.0, 10.0, 100.0, 20.0)
    invisible = text_run(
        "Hello native PDF text",
        10.0,
        10.0,
        100.0,
        20.0,
        visible=False,
        provenance=(("text_render_mode", 3),),
    )

    assert native_text_runs_for_extraction([painted, invisible]) == [painted]


def test_native_extraction_drops_repeated_garbage_invisible_text_layer() -> None:
    painted = text_run(
        "Source: https://www.industrydocuments.ucsf.edu/docs/pqlp0022",
        10.0,
        10.0,
        220.0,
        20.0,
    )
    garbage = text_run(
        "~" * 80,
        10.0,
        30.0,
        220.0,
        40.0,
        visible=False,
        provenance=(("text_render_mode", 3),),
    )

    assert native_text_runs_for_extraction([painted, garbage]) == [painted]


def test_native_extraction_normalizes_narrow_misdecoded_space_glyphs() -> None:
    narrow_space = text_run("a", 10.0, 10.0, 12.0, 20.0)
    narrow_space_2 = text_run("a", 14.0, 10.0, 16.0, 20.0)
    narrow_space_3 = text_run("a", 18.0, 10.0, 20.0, 20.0)
    for run in (narrow_space, narrow_space_2, narrow_space_3):
        run.space_width = 2.0
    real_letter = text_run("a", 20.0, 10.0, 24.0, 20.0)
    real_letter.space_width = 2.0

    normalized = native_text_runs_for_extraction(
        [narrow_space, narrow_space_2, narrow_space_3, real_letter]
    )

    assert [run.text for run in normalized] == [" ", " ", " ", "a"]


def test_native_extraction_normalizes_unicode_checkbox_glyphs() -> None:
    normalized = native_text_runs_for_extraction(
        [text_run("☒", 10.0, 10.0, 16.0, 16.0), text_run("☐", 20.0, 10.0, 26.0, 16.0)]
    )

    assert [run.text for run in normalized] == ["[x]", "[]"]


def test_native_extraction_normalizes_fullwidth_ascii_on_latin_pages() -> None:
    def line(text: str) -> ResolvedTextLine:
        observation = PageObservation("text", "native", bbox=(0.0, 0.0, 1.0, 1.0), text=text)
        return ResolvedTextLine(text, observation)

    normalized = normalize_fullwidth_ascii_text((line("Rubber （CR） ， ５０"),))
    cjk = normalize_fullwidth_ascii_text((line("日本語 （CR）"),))

    assert normalized[0].text == "Rubber (CR) , 50"
    assert cjk[0].text == "日本語 （CR）"


def test_native_extraction_normalizes_latin_ligatures() -> None:
    def line(text: str) -> ResolvedTextLine:
        observation = PageObservation("text", "native", bbox=(0.0, 0.0, 1.0, 1.0), text=text)
        return ResolvedTextLine(text, observation)

    normalized = normalize_latin_ligatures((line("ﬁnal ﬂow oﬃce eﬄuent ﬁﬂ"),))

    assert [item.text for item in normalized] == ["final flow office effluent fifl"]


@pytest.mark.parametrize(
    (
        "fixture_name",
        "expected_rotation",
        "expected_text",
        "ordered_markers",
        "expected_block_kinds",
    ),
    [
        (
            "BarrowArchAnalysis_Alaska1984-p076.pdf",
            90,
            "PORT CAPACITY AT ANCHORAGE",
            ("Port Anchorage", "PORT CAPACITY AT ANCHORAGE", "General Cargo"),
            (),
        ),
        (
            "global-AIDS-strategy-p74-75-p001.pdf",
            0,
            "GLOBAL AIDS STRATEGY 2021–2026",
            ("GLOBAL AIDS STRATEGY", "leadership can play", "Financial an"),
            ("heading",),
        ),
        (
            "korean_power_system_challenges-p003.pdf",
            0,
            "This document was prepared as an account of work",
            (
                "Korean Power System Challenges",
                "Disclaimer",
                "This document was prepared",
            ),
            (),
        ),
        (
            "Employee_Health_Benefits_Assess-p006.pdf",
            180,
            "Data Findings Presentation",
            ("Data Findings Presentation", "Provide an oral presentation", "The presentation"),
            ("list",),
        ),
        (
            "Index_to_Positions_table_vertical_text-p063.pdf",
            0,
            "Guards",
            ("2758- 82", "Guards", "Clk-otenograpbers"),
            (),
        ),
        (
            "CV_RenyuHu_2023p4-4.pdf",
            0,
            "EXTERNALLY SPONSORED RESEARCH PROJECTS",
            ("Renyu\tHu", "EXTERNALLY SPONSORED RESEARCH PROJECTS", "Principal\tInvestigator"),
            (),
        ),
        (
            "Financing-the-big-investment-p002.pdf",
            0,
            "The Grantham Research Institute",
            (
                "The Grantham Research Institute",
                "The Brookings Institution",
                "Financing a big investment push",
            ),
            (),
        ),
        (
            "IRS-2023-Form-1095-A-p002.pdf",
            0,
            "Form 1095-A",
            ("Form 1095-A", "Part I Recipient Information", "33 Annual Totals"),
            (),
        ),
    ],
)
def test_native_extraction_quality_corpus(
    fixture_name: str,
    expected_rotation: int,
    expected_text: str,
    ordered_markers: tuple[str, ...],
    expected_block_kinds: tuple[str, ...],
) -> None:
    fixture = TESTS_DIR / "fixtures" / "SCORE-Bench" / "src" / fixture_name

    with PdfDocument.open(fixture) as document:
        page = cast(Any, document.pages[0])
        result = build_page_extraction_result(page)
        text = result_text(result)

        assert page.rotation == expected_rotation
        assert expected_text.casefold() in text.casefold()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        assert lines
        if result.page_class != "table":
            assert all(left != right for left, right in zip(lines, lines[1:], strict=False))
        positions = [text.casefold().index(marker.casefold()) for marker in ordered_markers]
        assert positions == sorted(positions)
        assert expected_block_kinds == tuple(
            sorted({block.kind for block in result.blocks if block.kind != "paragraph"})
        )

        snapshot = SNAPSHOT_DIR / f"{Path(fixture_name).stem}.md"
    assert native_snapshot(fixture_name, page, result) == snapshot.read_text()


def test_native_table_text_joins_tightly_split_words() -> None:
    fixture = (
        TESTS_DIR
        / "fixtures"
        / "SCORE-Bench"
        / "src"
        / (
            "NASA-SNA-8-D-027III-Rev2-CsmLmSpacecraftOperationalDataBook-"
            "Volume3-MassProperties-pg856.pdf"
        )
    )

    with PdfDocument.open(fixture) as document:
        result = build_page_extraction_result(cast(Any, document.pages[0]))
        text = result_text(result)

    assert "Tank Volume" in text
    assert "Tank Vo lume" not in text
    assert "(in³)" in text
    assert text.count("in³") >= 6
    assert "F. Volume of S/C GHE Line (in³)" in text
    assert "Q. Tank Volume (in³)" in text
    assert "SERVICE MODULE" in text
    assert "Loading Temperature" in text
    assert "Secondary Oxidizer" in text
    assert "Initial Weight" in text


def test_native_multicolumn_text_does_not_merge_same_baseline_columns() -> None:
    fixture = TESTS_DIR / "fixtures" / "SCORE-Bench" / "src" / "GlobalTrends_2040p10-17-p004.pdf"

    with PdfDocument.open(fixture) as document:
        result = build_page_extraction_result(cast(Any, document.pages[0]))
        text = result_text(result)

    assert "ex- Several global" not in text
    assert "technolog-ical" not in text
    assert "exacerbate risks to human and national" in text
    assert "technological developments will increase" in text
    assert "Several global economic trends" in text
    assert "The pace and reach of technological developments" in text


def test_native_patent_columns_keep_adjacent_rows_separate() -> None:
    fixture = TESTS_DIR / "fixtures" / "SCORE-Bench" / "src" / "pet-display-patent-p001.pdf"

    with PdfDocument.open(fixture) as document:
        result = build_page_extraction_result(cast(Any, document.pages[0]))
        lines = result_text(result).splitlines()

    assert not any("Brooklyn" in line and "5,445,302" in line for line in lines)


def test_native_sidebar_column_is_emitted_after_main_column() -> None:
    fixture = (
        TESTS_DIR / "fixtures" / "SCORE-Bench" / "src" / "healthcare-workforce-strategies-p002.pdf"
    )
    with PdfDocument.open(fixture) as document:
        text = document.extract().text

    assert text.index("Exercise contractual agreements") < text.index("RELATED ASPR")
    assert text.index("EMAC is implemented") < text.index("Alternate Care Sites")


def test_native_recovers_horizontal_rule_from_misleading_combining_mapping() -> None:
    fixture = (
        TESTS_DIR / "fixtures" / "SCORE-Bench" / "src" / "healthcare-workforce-strategies-p003.pdf"
    )
    with PdfDocument.open(fixture) as document:
        text = document.extract().text

    assert "–\tNational Disaster Medical System" in text
    assert "̛\tNational Disaster Medical System" not in text


def test_native_table_headings_preserve_tight_letter_runs() -> None:
    fixture = TESTS_DIR / "fixtures" / "SCORE-Bench" / "src" / "EPD-p001.pdf"

    with PdfDocument.open(fixture) as document:
        result = build_page_extraction_result(cast(Any, document.pages[0]))
        text = result_text(result)

    assert "Gross Accumulated" in text
    assert "G ro ss" not in text


def test_native_same_run_camel_case_does_not_gain_a_space() -> None:
    fixture = (
        TESTS_DIR
        / "fixtures"
        / "SCORE-Bench"
        / "src"
        / "USDC-compression-vit-2310.11117-p7-p007.pdf"
    )

    with PdfDocument.open(fixture) as document:
        result = build_page_extraction_result(cast(Any, document.pages[0]))
        text = result_text(result)

    assert "DeiT-S" in text
    assert "Dei T-S" not in text


def test_native_sentence_boundaries_restore_missing_space() -> None:
    fixture = (
        TESTS_DIR
        / "fixtures"
        / "SCORE-Bench"
        / "src"
        / "USDC-compression-vit-2310.11117-p7-p004.pdf"
    )

    with PdfDocument.open(fixture) as document:
        result = build_page_extraction_result(cast(Any, document.pages[0]))
        text = result_text(result)

    assert "resource loss Lres. We" in text
    assert "\ndesign the resource loss" in text
    assert "resource loss Lres.We design" not in text


def test_native_formula_fonts_recover_untrusted_unicode_mappings() -> None:
    fixture = (
        TESTS_DIR
        / "fixtures"
        / "SCORE-Bench"
        / "src"
        / "USDC-compression-vit-2310.11117-p7-p004.pdf"
    )

    with PdfDocument.open(fixture) as document:
        result = build_page_extraction_result(cast(Any, document.pages[0]))
        text = result_text(result)

    assert "FLOPs difference" in text
    assert "γ Lres" in text
    assert "−ft" in text
    assert "Fffn" in text


def test_native_nested_xobject_text_does_not_interleave_page_text() -> None:
    fixture = TESTS_DIR / "fixtures" / "SCORE-Bench" / "src" / "water-15-0151828729_p3-3.pdf"

    with PdfDocument.open(fixture) as document:
        result = build_page_extraction_result(cast(Any, document.pages[0]))
        text = result_text(result)

    assert "Water 2023, 15, 1518 3 of 21" in text
    assert "x FOR PEER REVIEW" not in text
    assert "Section 3, “Rainwater Harvesting Systems”" in text
    assert "The top journals publishing papers on RWH" not in text


def test_native_discards_alternate_clipped_review_layer() -> None:
    fixture = TESTS_DIR / "fixtures" / "SCORE-Bench" / "src" / "ijerph-19-00825-p020.pdf"

    with PdfDocument.open(fixture) as document:
        result = build_page_extraction_result(cast(Any, document.pages[0]))
        text = result_text(result)

    assert "Int. J. Environ. Res. Public Health 2022, 19, 825" in text
    assert "x FOR PEER REVIEW" not in text
    assert "# 3 49.686" in text


def test_nested_duplicate_layer_is_removed_by_token_overlap() -> None:
    page_runs = [
        SimpleNamespace(text=f"shared token {index}", xobject_depth=0) for index in range(30)
    ]
    nested_runs = [
        SimpleNamespace(text=f"shared token {index}", xobject_depth=1) for index in range(30)
    ]

    assert discard_overlapping_nested_xobject_runs([*page_runs, *nested_runs]) == page_runs


def test_native_formula_page_preserves_visual_reading_order() -> None:
    fixture = (
        TESTS_DIR
        / "fixtures"
        / "SCORE-Bench"
        / "src"
        / "OptimalEstimationMethodologies-for-PanelDataRegressionModels-pg9-12-p003.pdf"
    )

    with PdfDocument.open(fixture) as document:
        result = build_page_extraction_result(cast(Any, document.pages[0]))
        text = result_text(result)

    assert text.index("Then, it follows that") < text.index("(2.18)")
    assert text.index("(2.18)") < text.index("Denote")
    assert text.index("Denote") < text.index("(2.19)")
    assert text.index("(2.19)") < text.index("Then, the K statistic")
    assert "∂Q T(θ)/∂θ =D T(θ)" in text
    assert "\n∂θ f f T" not in text
    assert "∂Q T(θ)/∂θ" in text
    assert "1/√" in text
    assert "t/T" in text
    assert "s/T" in text
    assert "V−1/ff" not in text
    assert "K T(θ₀) =( ∂Q T(θ₀)/∂θ)" in text
    assert "∂φ(θ₀)/∂θ = ∂φ(θ)/∂θ" in text
    assert "θ₀" in text


def test_native_formula_script_variables_preserve_token_boundaries() -> None:
    fixture = (
        TESTS_DIR
        / "fixtures"
        / "SCORE-Bench"
        / "src"
        / "OptimalEstimationMethodologies-for-PanelDataRegressionModels-pg9-12-p002.pdf"
    )

    with PdfDocument.open(fixture) as document:
        result = build_page_extraction_result(cast(Any, document.pages[0]))
        text = result_text(result)

    assert "V ff(θ)" in text
    assert "f(Y t,θ)" in text
    assert "Vff(θ)" not in text


def test_native_paragraphs_join_soft_hyphenated_line_breaks() -> None:
    fixture = (
        TESTS_DIR
        / "fixtures"
        / "SCORE-Bench"
        / "src"
        / "USDC-compression-vit-2310.11117-p7-p007.pdf"
    )

    with PdfDocument.open(fixture) as document:
        result = build_page_extraction_result(cast(Any, document.pages[0]))
        text = result_text(result)

    assert "on random, and our recursive" in text
    assert "dynamic compression of USDC" in text
    assert "ran-\ndom" not in text
    assert "dy-\nnamic" not in text


def test_native_tables_preserve_row_order_when_table_columns_are_localized() -> None:
    fixture = (
        TESTS_DIR
        / "fixtures"
        / "SCORE-Bench"
        / "src"
        / "USDC-compression-vit-2310.11117-p7-p007.pdf"
    )

    with PdfDocument.open(fixture) as document:
        result = build_page_extraction_result(cast(Any, document.pages[0]))
        text = result_text(result)

    assert text.index("Model B Top-1 Accuracy (%)") < text.index("Avg-32 Avg-8 Random Ours")
    assert text.index("Avg-32 Avg-8 Random Ours") < text.index("256 77.13")
    assert text.index("256 77.13") < text.index("USDC 32 77.10")
    assert text.index("USDC 32 77.10") < text.index("(DeiT-S) 8 77.05")


def test_native_table_drops_dense_vertical_margin_labels() -> None:
    fixture = (
        TESTS_DIR
        / "fixtures"
        / "SCORE-Bench"
        / "src"
        / "Management-By-Objectives-Food-Exhibit-3-p004.pdf"
    )

    with PdfDocument.open(fixture) as document:
        result = build_page_extraction_result(cast(Any, document.pages[0]))
        text = result_text(result)

    assert text.startswith("Attend all food industry Dietary Guidelines (DG)")
    assert "\nCD\n" not in f"\n{text}\n"


def test_native_flattened_form_checkboxes_are_explicit() -> None:
    fixture = TESTS_DIR / "fixtures" / "SCORE-Bench" / "src" / "SEC-FORM-D-OAG-p002.pdf"

    with PdfDocument.open(fixture) as document:
        result = build_page_extraction_result(cast(Any, document.pages[0]))
        text = result_text(result)

    assert "Relationship: [x] Executive Officer [] Director [] Promoter" in text
    assert "Relationship: X Executive Officer" not in text
    assert "xslFormDX01/primary_doc.xml" in text
    assert "2/5" in text
