# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import sys
import tracemalloc
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import pytest

from core_pdf import PdfDocument
from core_pdf.impl.engine import parse as ocr
from core_pdf.impl.engine.rendering import RasterImage, RenderOptions
from core_pdf.impl.engine.spec.s_08_graphics import image_decode

FIXTURES = Path(__file__).parents[1] / "fixtures" / "SCORE-Bench" / "src"
PAGE_CASES = (
    pytest.param(
        "global-AIDS-strategy-p74-75-p001.pdf",
        id="hybrid",
    ),
    pytest.param(
        "esp32_s3_circuit_schematic.pdf",
        id="ocr",
    ),
    pytest.param(
        "Employee_Health_Benefits_Assess-p006.pdf",
        id="native",
    ),
)

IMAGE_CONSUMER_CASES = (
    pytest.param(
        "153rd-Omaha-Pow-Wow-p001.pdf",
        id="jpeg2000-scan",
    ),
)

TYPE3_VECTOR_OCR_CASES = (
    pytest.param(
        "005-CISA-AA22-076-Strengthening-Cybersecurity-p1-p4-p001.pdf",
        id="type3-vector-ocr-p001",
    ),
    pytest.param(
        "005-CISA-AA22-076-Strengthening-Cybersecurity-p1-p4-p004.pdf",
        id="type3-vector-ocr-p004",
    ),
)


def internal_fixture(name: str) -> Path:
    path = FIXTURES / name
    if not path.exists():
        pytest.skip()
    return path


def internal_score_bench_fixtures() -> tuple[Path, ...]:
    paths = tuple(sorted(FIXTURES.glob("*.pdf")))
    if not paths:
        pytest.skip()
    return paths


def internal_build_page_program(path: Path) -> int:
    with PdfDocument.open(path) as document:
        return len(document.pages[0].get_page_program().events.sequence)


def internal_extract_page(path: Path) -> dict[str, Any]:
    with PdfDocument.open(path) as document:
        page = document.pages[0]
        page.extract()
        cache = page.extraction_cache
        assert cache is not None
        metrics = cache.get("parse_metrics")
        assert isinstance(metrics, dict)
        return cast(dict[str, Any], metrics)


def internal_render_type3_vector_ocr_page(path: Path) -> dict[str, int]:
    with PdfDocument.open(path) as document:
        rendered = document.pages[0].render(RenderOptions(include_text=False))
        raster = rendered.rasterize(
            scale=3.5,
            crop=(0.0, 0.0, 612.0, 792.0),
            cache=False,
        )
        image_timings = rendered.metadata.get("__core_pdf_raster_image_timings__", {})
        decoders = tuple(document.decoder_cache.values())
        return {
            "raster_pixels": raster.width * raster.height,
            "type3_cache_hits": sum(decoder.type3_charproc_cache_hits for decoder in decoders),
            "type3_cache_misses": sum(decoder.type3_charproc_cache_misses for decoder in decoders),
            "type3_compiled_programs": sum(
                decoder.type3_charproc_compiled_programs for decoder in decoders
            ),
            "type3_compiled_operations": sum(
                decoder.type3_charproc_compiled_operations for decoder in decoders
            ),
            "type3_unsafe_fallbacks": sum(
                decoder.type3_charproc_unsafe_fallbacks for decoder in decoders
            ),
            "tiled_affine_blits": int(image_timings.get("tiled_affine_blit_count", 0)),
            "tiled_affine_peak_scratch_bytes": int(
                image_timings.get("tiled_affine_peak_scratch_bytes", 0)
            ),
        }


def internal_calibrate_preflight(paths: tuple[Path, ...]) -> dict[str, Any]:
    class_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    pairs: Counter[tuple[str, str]] = Counter()
    native_mismatches: list[str] = []
    image_mismatches: list[str] = []
    vector_mismatches: list[str] = []
    for path in paths:
        with PdfDocument.open(path) as document:
            page = document.pages[0]
            capture = ocr.capture_page(page)
            preflight = ocr.preflight_page(page, capture)
            plan = ocr.plan_page(capture)
        page_class = preflight.recommendation.page_class.value
        route = plan.route.value
        class_counts[page_class] += 1
        route_counts[route] += 1
        pairs[(page_class, route)] += 1
        if page_class == "native-text" and route != "native":
            native_mismatches.append(path.name)
        if page_class == "image-only" and route != "ocr":
            image_mismatches.append(path.name)
        if page_class == "vector-diagram" and route == "native":
            vector_mismatches.append(path.name)
    return {
        "total": len(paths),
        "preflight_class_counts": dict(class_counts),
        "planned_route_counts": dict(route_counts),
        "preflight_route_pairs": {f"{key[0]}->{key[1]}": value for key, value in pairs.items()},
        "native_text_route_mismatches": tuple(native_mismatches),
        "image_only_route_mismatches": tuple(image_mismatches),
        "vector_diagram_route_mismatches": tuple(vector_mismatches),
    }


def internal_profile_page_program(
    path: Path,
) -> tuple[int, int, int, int, int, dict[str, dict[str, int]]]:
    tracemalloc.start()
    try:
        with PdfDocument.open(path) as document:
            program = document.pages[0].get_page_program()
            event_count = len(program.events.sequence)
            column_bytes = sum(
                getattr(column, "nbytes", 0)
                for column in (
                    program.events.sequence,
                    program.events.kind,
                    program.events.bbox,
                    program.events.payload,
                    program.events.visible,
                )
            )
            product_groups = (
                ("runs", program.products.runs),
                ("glyphs", program.products.glyphs),
                ("drawings", program.products.drawings),
                ("inline_images", program.products.inline_images),
                ("lines", program.products.lines),
            )
            product_breakdown = {
                name: {
                    "count": len(group),
                    "tuple_bytes": sys.getsizeof(group),
                    "storage_bytes": getattr(group, "nbytes", 0),
                    "object_bytes_shallow": (
                        0
                        if hasattr(group, "nbytes")
                        else sum(sys.getsizeof(item) for item in group)
                    ),
                }
                for name, group in product_groups
            }
            product_tuple_bytes = sum(
                values["tuple_bytes"] for values in product_breakdown.values()
            )
            product_object_bytes = sum(
                values["object_bytes_shallow"] for values in product_breakdown.values()
            )
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return (
        event_count,
        peak_bytes,
        column_bytes,
        product_tuple_bytes,
        product_object_bytes,
        product_breakdown,
    )


@pytest.mark.parametrize("fixture_name", PAGE_CASES)
def test_cold_page_program_construction_benchmark(benchmark, fixture_name: str) -> None:
    event_count = benchmark.pedantic(
        internal_build_page_program,
        args=(internal_fixture(fixture_name),),
        iterations=1,
        rounds=3,
    )

    assert event_count > 0


@pytest.mark.parametrize("fixture_name", PAGE_CASES)
def test_end_to_end_page_extraction_benchmark(benchmark, fixture_name: str) -> None:
    metrics = benchmark.pedantic(
        internal_extract_page,
        args=(internal_fixture(fixture_name),),
        iterations=1,
        rounds=1,
    )

    assert metrics["route"] in {"native", "hybrid", "ocr"}
    assert metrics["page_program_events"] >= 0
    assert metrics["content_stream_passes"] == 1
    assert metrics["preflight_native_route_mismatch"] in {0, 1}
    assert metrics["preflight_image_route_mismatch"] in {0, 1}
    assert metrics["preflight_vector_route_mismatch"] in {0, 1}
    benchmark.extra_info.update(
        {
            "route": metrics["route"],
            "preflight_class": metrics.get("preflight_class"),
            "capture_seconds": metrics.get("capture_seconds"),
            "preflight_seconds": metrics.get("preflight_seconds"),
            "page_program_events": metrics["page_program_events"],
            "ocr_seconds": metrics.get("ocr_seconds"),
            "preflight_native_route_mismatch": metrics["preflight_native_route_mismatch"],
            "preflight_image_route_mismatch": metrics["preflight_image_route_mismatch"],
            "preflight_vector_route_mismatch": metrics["preflight_vector_route_mismatch"],
        }
    )


@pytest.mark.parametrize("fixture_name", TYPE3_VECTOR_OCR_CASES)
def test_type3_vector_ocr_raster_benchmark(benchmark, fixture_name: str) -> None:
    metrics = benchmark.pedantic(
        internal_render_type3_vector_ocr_page,
        args=(internal_fixture(fixture_name),),
        iterations=1,
        rounds=3,
    )
    benchmark.extra_info.update(metrics)

    assert metrics["raster_pixels"] > 0
    assert metrics["type3_cache_hits"] > metrics["type3_cache_misses"]
    assert metrics["type3_compiled_programs"] > 0
    assert metrics["type3_compiled_operations"] > 0
    assert metrics["type3_unsafe_fallbacks"] == 0
    assert metrics["tiled_affine_blits"] > 0
    assert metrics["tiled_affine_peak_scratch_bytes"] <= 1 << 20


def test_score_bench_preflight_calibration_benchmark(benchmark) -> None:
    summary = benchmark.pedantic(
        internal_calibrate_preflight,
        args=(internal_score_bench_fixtures(),),
        iterations=1,
        rounds=1,
    )
    benchmark.extra_info.update(summary)

    assert summary["total"] > 0


@pytest.mark.parametrize("fixture_name", PAGE_CASES)
def test_page_program_memory_profile_benchmark(benchmark, fixture_name: str) -> None:
    (
        event_count,
        peak_bytes,
        column_bytes,
        product_tuple_bytes,
        product_object_bytes,
        product_breakdown,
    ) = benchmark.pedantic(
        internal_profile_page_program,
        args=(internal_fixture(fixture_name),),
        iterations=1,
        rounds=1,
    )
    benchmark.extra_info.update(
        {
            "event_count": event_count,
            "tracemalloc_peak_bytes": peak_bytes,
            "event_column_bytes": column_bytes,
            "product_tuple_bytes": product_tuple_bytes,
            "product_object_bytes_shallow": product_object_bytes,
            "product_breakdown": product_breakdown,
        }
    )

    assert event_count > 0
    assert peak_bytes >= 0
    assert column_bytes > 0
    assert product_tuple_bytes > 0


def internal_consume_image_page(path: Path) -> dict[str, Any]:
    with PdfDocument.open(path) as document:
        page = document.pages[0]
        started = perf_counter()
        page_program = page.get_page_program()
        page_program_ms = (perf_counter() - started) * 1000
        started = perf_counter()
        images = page.extract_images()
        image_extraction_ms = (perf_counter() - started) * 1000
        started = perf_counter()
        page.extract()
        text_extraction_ms = (perf_counter() - started) * 1000
        metrics = page.extraction_cache.get("parse_metrics") if page.extraction_cache else None
        assert isinstance(metrics, dict)
        metrics = cast(dict[str, Any], metrics)
        started = perf_counter()
        page.render().rasterize(scale=0.25)
        rasterization_ms = (perf_counter() - started) * 1000
        return {
            "events": len(page_program.events.sequence),
            "images": len(images),
            "page_program_ms": round(page_program_ms),
            "image_extraction_ms": round(image_extraction_ms),
            "text_extraction_ms": round(text_extraction_ms),
            "rasterization_ms": round(rasterization_ms),
            "ocr_seconds": metrics["ocr_seconds"],
            "ocr_raster_pixels": metrics["ocr_raster_pixels"],
            "ocr_observations": metrics["ocr_observations"],
        }


@pytest.mark.parametrize("fixture_name", IMAGE_CONSUMER_CASES)
def test_shared_image_consumers_benchmark(benchmark, fixture_name: str) -> None:
    result = benchmark.pedantic(
        internal_consume_image_page,
        args=(internal_fixture(fixture_name),),
        iterations=1,
        rounds=1,
    )
    benchmark.extra_info.update(
        {
            key: value
            for key, value in result.items()
            if key.endswith("internal_ms") or key.startswith("ocr_")
        }
    )

    assert result["events"] > 0
    assert result["images"] > 0


def test_shared_image_decode_once(monkeypatch: pytest.MonkeyPatch) -> None:
    path = internal_fixture("153rd-Omaha-Pow-Wow-p001.pdf")
    decode_count = 0
    original_decode = image_decode.decode_pdf_image

    def counted_decode(*args: Any, **kwargs: Any) -> Any:
        nonlocal decode_count
        decode_count += 1
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(image_decode, "decode_pdf_image", counted_decode)
    with PdfDocument.open(path) as document:
        page = document.pages[0]
        page.extract_images()
        page.extract()
        page.render().rasterize(scale=0.25)

    assert decode_count == 1


def internal_recognize_same_raster_regions(path: Path, *, batched: bool, region_count: int) -> int:
    with PdfDocument.open(path) as document:
        page = document.pages[0]
        page_program = page.get_page_program()
        drawing = next(
            drawing for drawing in page_program.products.drawings if drawing.kind == "image"
        )
        source = drawing.image_source
        assert source is not None
        raster = source.decode()
        assert raster is not None
        image = RasterImage(
            raster.array.tobytes(),
            raster.width,
            raster.height,
            raster.channels,
        )
        height = raster.height
        region_height = height // (region_count + 1)
        tasks = tuple(
            ocr.internal_OcrTask(
                mode=11,
                image=image,
                rectangle=(0, index * region_height, raster.width, region_height),
                page_box=(0.0, 0.0, float(page.width), float(page.height)),
                resolution=300,
            )
            for index in range(region_count)
        )
        ocr.internal_OCR_LOCAL.api = None
        candidates = (
            ocr.internal_recognize_group(tasks)
            if batched
            else tuple(ocr.internal_recognize(task) for task in tasks)
        )
        return len(candidates)


@pytest.mark.parametrize("batched", [False, True], ids=("independent", "batched"))
@pytest.mark.parametrize("region_count", [2, 4, 8], ids=("2-regions", "4-regions", "8-regions"))
def test_multi_region_ocr_batch_benchmark(
    benchmark,
    batched: bool,
    region_count: int,
) -> None:
    result = benchmark.pedantic(
        internal_recognize_same_raster_regions,
        args=(internal_fixture("DEA_Compliance-Rotat_Table-Form_Img-p001.pdf"),),
        kwargs={"batched": batched, "region_count": region_count},
        iterations=1,
        rounds=1,
    )

    assert result == region_count
