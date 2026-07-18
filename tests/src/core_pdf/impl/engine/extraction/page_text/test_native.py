from io import BytesIO
from pathlib import Path
from typing import Any, cast

import pytest
from core_layout.impl.layout.models import TextRun

from core_pdf.impl.engine.extraction.document import PdfDocument
from core_pdf.impl.engine.extraction.page_text.engine import build_page_extraction_result
from core_pdf.impl.engine.extraction.page_text.native import native_text_runs_for_extraction

TESTS_DIR = Path(__file__).parents[6]
SAMPLE_PDF = TESTS_DIR / "fixtures" / "SCORE-Bench" / "src" / "global-AIDS-strategy-p74-75-p001.pdf"
SNAPSHOT_DIR = TESTS_DIR / "snapshots" / "native"


def native_snapshot(fixture_name: str, page: Any, result: Any) -> str:
    lines = [
        "---",
        f"fixture: {fixture_name}",
        f"rotation: {page.rotation}",
        f"page_class: {result.page_class}",
        f"base_route: {result.base_route}",
        f"block_count: {len(result.blocks)}",
        f"line_count: {len(result.resolved_lines)}",
        "---",
        "",
    ]
    for index, line in enumerate(result.resolved_lines, 1):
        lines.extend(
            (
                f"<!-- line: {index:03d}; break_before: {line.break_before}; "
                f"kind: {line.kind}; source: {line.source} -->",
                "```text",
                line.text,
                "```",
            )
        )
    return "\n".join(lines) + "\n"


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
        assert not hasattr(page, "extract_text")
        assert page.extraction_cache is not None
        assert "native_output_lines" in page.extraction_cache
        assert "page_extraction_snapshot" not in page.extraction_cache


def test_structured_page_result_reports_native_route() -> None:
    with PdfDocument.open(SAMPLE_PDF) as document:
        page = cast(Any, document.pages[0])
        result = build_page_extraction_result(page)

        assert result_text(result).strip()
        assert result.base_route in {"native_fast", "native_layout"}
        assert result.resolved_lines


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


@pytest.mark.parametrize(
    ("fixture_name", "expected_rotation", "expected_text", "ordered_markers"),
    [
        (
            "BarrowArchAnalysis_Alaska1984-p076.pdf",
            90,
            "PORT CAPACITY AT ANCHORAGE",
            ("Port Anchorage", "PORT CAPACITY AT ANCHORAGE", "General Cargo"),
        ),
        (
            "global-AIDS-strategy-p74-75-p001.pdf",
            0,
            "GLOBAL AIDS STRATEGY 2021–2026",
            ("GLOBAL AIDS STRATEGY", "leadership can play", "Financial an"),
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
        ),
        (
            "Employee_Health_Benefits_Assess-p006.pdf",
            180,
            "Data Findings Presentation",
            ("Data Findings Presentation", "Provide an oral presentation", "The presentation"),
        ),
    ],
)
def test_native_extraction_quality_corpus(
    fixture_name: str,
    expected_rotation: int,
    expected_text: str,
    ordered_markers: tuple[str, ...],
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
        assert all(left != right for left, right in zip(lines, lines[1:], strict=False))
        positions = [text.casefold().index(marker.casefold()) for marker in ordered_markers]
        assert positions == sorted(positions)

        snapshot = SNAPSHOT_DIR / f"{Path(fixture_name).stem}.md"
        assert native_snapshot(fixture_name, page, result) == snapshot.read_text()
