from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from core_pdf import PdfDocument


def test_parse_pipeline_records_route_and_stage_timings() -> None:
    fixture = (
        Path(__file__).parent
        / "fixtures"
        / "SCORE-Bench"
        / "src"
        / "global-AIDS-strategy-p74-75-p001.pdf"
    )
    with PdfDocument.open(fixture) as document:
        page = document.pages[0]
        page.extract()
        cache = page.extraction_cache
        assert cache is not None
        metrics = cast(dict[str, Any], cache["parse_metrics"])

    assert metrics["route"] in {"native", "hybrid", "ocr"}
    assert metrics["preflight_class"] in {
        "native-text",
        "image-only",
        "vector-diagram",
        "mixed",
        "likely-malformed",
    }
    assert metrics["content_stream_passes"] == 1
    assert metrics["page_program_events"] >= 0
    assert metrics["preflight_native_route_mismatch"] in {0, 1}
    assert metrics["preflight_image_route_mismatch"] in {0, 1}
    assert metrics["preflight_vector_route_mismatch"] in {0, 1}
    assert "preflight" in cache
    for key in ("capture_seconds", "layout_seconds", "ocr_seconds"):
        value = metrics[key]
        assert isinstance(value, (int, float))
        assert value >= 0.0


def test_minor_rotated_native_text_does_not_force_page_ocr() -> None:
    fixture = (
        Path(__file__).parent
        / "fixtures"
        / "SCORE-Bench"
        / "src"
        / "GlobalTrends_2040p10-17-p004.pdf"
    )
    with PdfDocument.open(fixture) as document:
        page = document.pages[0]
        page.extract()
        cache = page.extraction_cache
        assert cache is not None
        metrics = cast(dict[str, Any], cache["parse_metrics"])

    assert metrics["route"] == "native"
    assert metrics["ocr_raster_pixels"] == 0


def test_high_confidence_image_region_supplement_skips_fallback() -> None:
    filename = (
        "NASA-SNA-8-D-027III-Rev2-CsmLmSpacecraftOperationalDataBook-"
        "Volume3-MassProperties-Pg54.pdf"
    )
    fixture = Path(__file__).parent / "fixtures" / "SCORE-Bench" / "src" / filename
    with PdfDocument.open(fixture) as document:
        page = document.pages[0]
        page.extract()
        cache = page.extraction_cache
        assert cache is not None
        diagnostics = cast(tuple[dict[str, object], ...], cache["ocr_pass_diagnostics"])
        metrics = cast(dict[str, Any], cache["parse_metrics"])

    assert [item["name"] for item in diagnostics] == [
        "image-regions",
    ]
    assert diagnostics[0]["region_stage"] == "direct-image-regions"
    assert diagnostics[0]["task_count"] == 1
    assert diagnostics[0]["raster_pixels"] == 2_927 * 1_680
    assert diagnostics[0]["render_timings"] == {}
    assert metrics["ocr_full_page_fallback"] == 0


def test_near_axis_photo_supplement_decodes_source_image() -> None:
    fixture = (
        Path(__file__).parent
        / "fixtures"
        / "SCORE-Bench"
        / "src"
        / "SPUR_Future_Of_Transportation-p001.pdf"
    )
    with PdfDocument.open(fixture) as document:
        page = document.pages[0]
        text = page.extract().text
        cache = page.extraction_cache
        assert cache is not None
        diagnostics = cast(tuple[dict[str, object], ...], cache["ocr_pass_diagnostics"])
        metrics = cast(dict[str, Any], cache["parse_metrics"])

    assert [item["name"] for item in diagnostics] == ["image-regions"]
    assert diagnostics[0]["region_stage"] == "direct-image-regions"
    assert diagnostics[0]["task_count"] == 1
    assert diagnostics[0]["raster_pixels"] == 3_416 * 892
    assert diagnostics[0]["render_timings"] == {}
    assert "TEMPO" in text
    assert "2305" in text
    assert metrics["ocr_full_page_fallback"] == 0


def test_blank_image_supplement_is_rejected_before_ocr() -> None:
    fixture = (
        Path(__file__).parent
        / "fixtures"
        / "SCORE-Bench"
        / "src"
        / "Carbone-4-for-Bird-E-Scooter.pdf"
    )
    with PdfDocument.open(fixture) as document:
        page = document.pages[0]
        page.extract()
        cache = page.extraction_cache
        assert cache is not None
        diagnostics = cast(tuple[dict[str, object], ...], cache["ocr_pass_diagnostics"])
        metrics = cast(dict[str, Any], cache["parse_metrics"])

    assert [item["name"] for item in diagnostics] == ["image-regions"]
    assert diagnostics[0]["region_stage"] == "image-text-preflight"
    assert diagnostics[0]["task_count"] == 0
    assert diagnostics[0]["raster_pixels"] == 0
    assert cast(int, diagnostics[0]["skipped_raster_pixels"]) == 2_191 * 2_489
    preflight = cast(tuple[dict[str, object], ...], diagnostics[0]["image_text_preflight"])
    assert preflight[0]["reason"] == "low-edge-density"
    assert metrics["ocr_raster_pixels"] == 0


def test_opaque_grayscale_soft_mask_uses_one_direct_ocr_pass() -> None:
    fixture = Path(__file__).parent / "fixtures" / "SCORE-Bench" / "src" / "sydd0278.pdf"
    with PdfDocument.open(fixture) as document:
        page = document.pages[0]
        page.extract()
        cache = page.extraction_cache
        assert cache is not None
        diagnostics = cast(tuple[dict[str, object], ...], cache["ocr_pass_diagnostics"])
        metrics = cast(dict[str, Any], cache["parse_metrics"])

    assert [item["name"] for item in diagnostics] == ["primary-page"]
    assert cast(int, diagnostics[0]["characters"]) >= 32
    assert cast(int, diagnostics[0]["raster_pixels"]) == 1_700 * 2_269
    assert metrics["ocr_full_page_fallback"] == 0


def test_stroked_vector_text_uses_one_packed_seed_raster() -> None:
    fixture = (
        Path(__file__).parent / "fixtures" / "SCORE-Bench" / "src" / "VCAs_REV2_SCHEMATIC-p002.pdf"
    )
    with PdfDocument.open(fixture) as document:
        page = document.pages[0]
        text = page.extract().text
        tables = page.extract_tables()
        cache = page.extraction_cache
        assert cache is not None
        diagnostics = cast(tuple[dict[str, object], ...], cache["ocr_pass_diagnostics"])
        metrics = cast(dict[str, Any], cache["parse_metrics"])
        plan = cast(dict[str, Any], cache["parse_plan"])
        packed = cast(dict[str, Any], cache["stroked_vector_packed"])

    assert plan["reason"] == "stroked-vector-text"
    assert [item["name"] for item in diagnostics] == ["stroked-vector-text"]
    assert diagnostics[0]["scope"] == "stroked-vector-text"
    assert diagnostics[0]["region_stage"] == "packed-stroked-vector-text"
    assert diagnostics[0]["task_count"] == 1
    assert cast(int, diagnostics[0]["raster_pixels"]) < 700_000
    render_timings = cast(dict[str, Any], diagnostics[0]["render_timings"])
    assert render_timings["raster_mode"] == "packed-stroked-vector-text"
    assert render_timings["raster_kernel"] == "wu"
    assert cast(int, render_timings["packed_cells"]) >= 100
    assert render_timings["horizontal_padding"] == 4.0
    assert render_timings["vertical_padding"] == 2.0
    assert cast(int, render_timings["display_items"]) < 20
    assert cast(int, packed["symbol_observations"]) >= 400
    assert metrics["stroked_vector_text_trusted"] == 1
    assert cast(int, metrics["stroked_vector_candidate_paths"]) >= 1_000
    assert cast(int, metrics["stroked_vector_packed_cells"]) >= 100
    assert metrics["stroked_vector_packed_fallback"] == 0
    assert cast(int, metrics["stroked_vector_decoded_runs"]) >= 100
    assert cast(int, metrics["stroked_vector_decode_additions"]) >= 40
    assert cast(int, metrics["stroked_vector_decode_corrections"]) >= 1
    assert cast(int, metrics["stroked_vector_approximate_signatures"]) >= 10
    assert cast(float, metrics["uncovered_vector_area"]) < 1_000.0
    assert tables == []
    assert "R15" in text
    assert "R21" in text
    assert {"D5", "D6", "D7", "D10", "D12", "R16", "R32", "1k"} <= set(text.split())


def test_layered_soft_mask_scan_uses_one_composited_word_pass() -> None:
    fixture = (
        Path(__file__).parent
        / "fixtures"
        / "SCORE-Bench"
        / "src"
        / "Roosevelt-Letter-Oppenheimer-p001.pdf"
    )
    with PdfDocument.open(fixture) as document:
        page = document.pages[0]
        page.extract()
        cache = page.extraction_cache
        assert cache is not None
        diagnostics = cast(tuple[dict[str, object], ...], cache["ocr_pass_diagnostics"])
        metrics = cast(dict[str, Any], cache["parse_metrics"])

    assert [item["name"] for item in diagnostics] == ["primary-page"]
    assert diagnostics[0]["recognize_words"] is True
    assert cast(int, diagnostics[0]["characters"]) >= 500
    assert cast(int, diagnostics[0]["raster_pixels"]) < 6_000_000
    assert metrics["ocr_full_page_fallback"] == 0
