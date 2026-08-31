import json
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from core_pdf.impl.document import PdfDocument
from core_pdf.impl.parse import ParsedPage
from core_pdf.impl.parse import pipeline as parse_pipeline
from core_pdf.impl.primitives import PdfName, PdfReference
from core_pdf.impl.runtime.execution import ExecutionRuntime, RuntimeConfig, TaskScope, WorkStage
from core_pdf.impl.spec.s_07_syntax.stream import PdfStream
from core_pdf.impl.writing import serialize_pdf_file

TESTS_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = TESTS_DIR / "SCORE-Bench" / "src" / "global-AIDS-strategy-p74-75-p001.pdf"
PAGE_OCR_PDF = TESTS_DIR / "SCORE-Bench" / "src" / "SFG-Content-Marketing-2021-p001.pdf"


def internal_many_page_pdf(page_count: int) -> bytes:
    objects: dict[int, object] = {
        1: {PdfName.of("Type"): PdfName.of("Catalog"), PdfName.of("Pages"): PdfReference(2)},
        3: {
            PdfName.of("Type"): PdfName.of("Font"),
            PdfName.of("Subtype"): PdfName.of("Type1"),
            PdfName.of("BaseFont"): PdfName.of("Helvetica"),
        },
    }
    kids: list[PdfReference] = []
    for page_index in range(page_count):
        page_object = 4 + page_index * 2
        content_object = page_object + 1
        kids.append(PdfReference(page_object))
        objects[page_object] = {
            PdfName.of("Type"): PdfName.of("Page"),
            PdfName.of("Parent"): PdfReference(2),
            PdfName.of("MediaBox"): [0, 0, 612, 792],
            PdfName.of("Resources"): {PdfName.of("Font"): {PdfName.of("F1"): PdfReference(3)}},
            PdfName.of("Contents"): PdfReference(content_object),
        }
        objects[content_object] = PdfStream(
            {}, f"BT /F1 10 Tf 36 750 Tm (Page {page_index}) Tj ET".encode()
        )
    objects[2] = {
        PdfName.of("Type"): PdfName.of("Pages"),
        PdfName.of("Kids"): kids,
        PdfName.of("Count"): page_count,
    }
    return serialize_pdf_file(objects, trailer={PdfName.of("Root"): PdfReference(1)})


@pytest.mark.parametrize(
    ("fixture", "expected_pass"),
    [
        pytest.param(SAMPLE_PDF, "image-regions", id="image-region"),
        pytest.param(PAGE_OCR_PDF, "primary-page", id="page"),
    ],
)
def test_ocr_extraction_can_start_in_an_application_worker(
    fixture: Path,
    expected_pass: str,
) -> None:
    script = textwrap.dedent(
        f"""
        import json
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path
        from core_pdf import PdfDocument

        fixture = Path({str(fixture)!r})

        def extract():
            with PdfDocument.open(fixture) as document:
                extracted = document.extract()
                report = document.pages[0].parse_report
                assert report is not None
                return {{
                    "characters": len(extracted.text),
                    "passes": [item["name"] for item in report.recognition.passes],
                    "worker": threading.current_thread() is not threading.main_thread(),
                }}

        with ThreadPoolExecutor(max_workers=1) as executor:
            print(json.dumps(executor.submit(extract).result()))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    assert result["characters"] > 0
    assert expected_pass in result["passes"]
    assert result["worker"] is True


def test_worker_first_ocr_initialization_has_an_actionable_error() -> None:
    script = textwrap.dedent(
        """
        from concurrent.futures import ThreadPoolExecutor

        def load_pdf_document():
            from core_pdf import PdfDocument
            return PdfDocument

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(load_pdf_document).result()
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode != 0
    assert "initialize OCR on the main thread" in completed.stderr


def test_run_on_each_worker_reaches_every_pooled_thread() -> None:
    runtime = ExecutionRuntime(RuntimeConfig(parent_workers=3, ocr_workers=3))
    try:
        threads: set[int] = set()
        lock = threading.Lock()

        def record() -> None:
            with lock:
                threads.add(threading.get_ident())

        assert runtime.run_on_each_worker(record) == 3
        assert len(threads) == 3
    finally:
        runtime.shutdown()


def test_run_on_each_worker_does_not_hang_when_workers_are_busy() -> None:
    runtime = ExecutionRuntime(RuntimeConfig(parent_workers=4, ocr_workers=4))
    release = threading.Event()
    busy_workers = threading.Barrier(3)
    try:

        def occupy_worker() -> None:
            busy_workers.wait(timeout=2)
            release.wait(30)

        for _ in range(2):
            runtime.internal_get_executor().submit(occupy_worker)
        busy_workers.wait(timeout=2)
        warmed = 0
        lock = threading.Lock()

        def record() -> None:
            nonlocal warmed
            with lock:
                warmed += 1

        started = time.perf_counter()
        # The barrier can never fill while two workers are blocked, so the wait
        # must break rather than deadlock the pool.
        runtime.run_on_each_worker(record, timeout=1.0)
        assert time.perf_counter() - started < 10.0
        assert warmed >= 1
    finally:
        release.set()
        runtime.shutdown()


def test_runtime_maps_in_order_with_bounded_workers() -> None:
    runtime = ExecutionRuntime()
    runtime.configure(RuntimeConfig(parent_workers=2))
    thread_ids: set[int] = set()

    def work(value: int) -> int:
        thread_ids.add(threading.get_ident())
        return value * 2

    assert list(runtime.map_ordered(work, range(8))) == [0, 2, 4, 6, 8, 10, 12, 14]
    assert thread_ids


def test_runtime_maps_in_completion_order_with_input_indexes() -> None:
    runtime = ExecutionRuntime()
    runtime.configure(RuntimeConfig(parent_workers=2))
    release_first = threading.Event()

    def work(value: int) -> int:
        if value == 0:
            assert release_first.wait(timeout=2)
        return value * 2

    results = runtime.map_completed(work, [0, 1])
    first = next(results)
    release_first.set()
    remainder = list(results)

    assert (first.index, first.value) == (1, 2)
    assert [(result.index, result.value) for result in remainder] == [(0, 0)]
    runtime.shutdown()


def test_completion_order_iterator_does_not_submit_more_work_when_closed() -> None:
    runtime = ExecutionRuntime()
    runtime.configure(RuntimeConfig(parent_workers=1))
    started: list[int] = []

    def work(value: int) -> int:
        started.append(value)
        return value

    with runtime.task_scope() as context:
        results = context.map_completed(work, range(8))
        first = next(results)
        results.close()

    assert (first.index, first.value) == (0, 0)
    assert len(started) <= 2
    runtime.shutdown()


def test_nested_maps_run_without_pool_deadlock() -> None:
    runtime = ExecutionRuntime()
    runtime.configure(RuntimeConfig(parent_workers=2))

    def inner(value: int) -> int:
        return value + 1

    def outer(value: int) -> list[int]:
        return list(runtime.map_ordered(inner, [value, value + 1]))

    assert list(runtime.map_ordered(outer, [0, 10])) == [[1, 2], [11, 12]]
    runtime.shutdown()


def test_stage_budget_limits_ocr_without_blocking_page_workers() -> None:
    runtime = ExecutionRuntime()
    runtime.configure(RuntimeConfig(parent_workers=4, ocr_workers=2))
    lock = threading.Lock()
    active = 0
    peak = 0

    def work(value: int) -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return value

    values = list(runtime.map_ordered(work, range(8), stage=WorkStage.OCR))
    metrics = runtime.metrics()

    assert values == list(range(8))
    assert peak == 2
    assert metrics.ocr_capacity == 2
    assert metrics.ocr_active == 0
    runtime.shutdown()


def test_nested_page_to_ocr_stages_run_without_deadlock() -> None:
    runtime = ExecutionRuntime()
    runtime.configure(RuntimeConfig(parent_workers=2, ocr_workers=1))

    def outer(value: int) -> list[int]:
        return list(
            runtime.map_ordered(
                lambda inner: inner + 1,
                (value,),
                stage=WorkStage.OCR,
            )
        )

    assert list(runtime.map_ordered(outer, (0, 10), stage=WorkStage.PAGE)) == [[1], [11]]
    runtime.shutdown()


def test_worker_nested_map_uses_idle_workers_and_caller() -> None:
    runtime = ExecutionRuntime()
    runtime.configure(RuntimeConfig(parent_workers=2))
    rendezvous = threading.Barrier(2)
    thread_ids: set[int] = set()

    def inner(value: int) -> int:
        thread_ids.add(threading.get_ident())
        rendezvous.wait(timeout=2)
        return value

    future = runtime.submit(lambda: list(runtime.map_ordered(inner, (1, 2))))

    assert future.result(timeout=3) == [1, 2]
    assert len(thread_ids) == 2
    runtime.shutdown()


def test_runtime_round_robins_pending_document_work() -> None:
    runtime = ExecutionRuntime()
    runtime.configure(RuntimeConfig(parent_workers=1))
    first_started = threading.Event()
    release_first = threading.Event()
    started: list[str] = []

    def first() -> str:
        started.append("a0")
        first_started.set()
        assert release_first.wait(timeout=2)
        return "a0"

    def record(label: str) -> str:
        started.append(label)
        return label

    with runtime.task_scope() as first_context:
        with runtime.task_scope() as second_context:
            futures = [first_context.submit(first)]
            assert first_started.wait(timeout=2)
            futures.extend(
                (
                    first_context.submit(record, "a1"),
                    second_context.submit(record, "b0"),
                    first_context.submit(record, "a2"),
                    second_context.submit(record, "b1"),
                )
            )
            release_first.set()
            assert [future.result(timeout=2) for future in futures] == [
                "a0",
                "a1",
                "b0",
                "a2",
                "b1",
            ]

    assert started == ["a0", "a1", "b0", "a2", "b1"]
    runtime.shutdown()


def test_raster_budget_blocks_until_the_active_lease_is_released() -> None:
    from core_pdf.impl.runtime import execution as runtime_module

    runtime = ExecutionRuntime()
    runtime.internal_raster_budget = runtime_module.internal_ResourceBudget(10)
    acquired = threading.Event()

    with runtime.task_scope() as context:

        def reserve() -> None:
            with context.reserve_raster(1):
                acquired.set()

        with context.reserve_raster(10):
            thread = threading.Thread(target=reserve)
            thread.start()
            assert not acquired.wait(timeout=0.05)
        assert acquired.wait(timeout=2)
        thread.join(timeout=2)

    runtime.shutdown()


def test_page_workers_waiting_on_ocr_do_not_nest_page_work_holding_raster_leases() -> None:
    """Regression: a page worker blocked on its OCR groups must not "help" by
    nesting another page parse on the same thread. Each page parse holds a raster
    lease across that wait, so with ``workers * lease == budget`` every worker
    would end up waiting for a lease held only by the waiting workers themselves.
    """
    lease = 16
    workers = 2
    runtime = ExecutionRuntime()
    runtime.configure(
        RuntimeConfig(
            parent_workers=workers,
            ocr_workers=workers,
            raster_budget_bytes=workers * lease,
        )
    )
    stop = threading.Event()
    nesting = threading.local()
    nested_page_parses = 0
    nested_lock = threading.Lock()

    with runtime.task_scope(cancelled=stop.is_set) as context:

        def ocr_group(value: int) -> int:
            time.sleep(0.005)
            return value

        def parse_page(value: int) -> list[int]:
            nonlocal nested_page_parses
            if getattr(nesting, "depth", 0):
                with nested_lock:
                    nested_page_parses += 1
            nesting.depth = getattr(nesting, "depth", 0) + 1
            try:
                with context.reserve_raster(lease):
                    return list(context.map_ordered(ocr_group, (value,), stage=WorkStage.OCR))
            finally:
                nesting.depth -= 1

        try:
            futures = [
                context.submit(parse_page, index, stage=WorkStage.PAGE)
                for index in range(workers * 3)
            ]
            assert [future.result(timeout=5) for future in futures] == [
                [index] for index in range(workers * 3)
            ]
        finally:
            stop.set()

    assert nested_page_parses == 0
    runtime.shutdown()


def test_context_tracks_scheduler_metrics_and_worker_state() -> None:
    runtime = ExecutionRuntime()
    runtime.configure(RuntimeConfig(parent_workers=2))

    with runtime.task_scope(metrics=True) as context:
        values = list(context.map_ordered(lambda value: (value, runtime.in_worker), range(4)))
        metrics = context.metrics()

    assert [value for value, internal_in_worker in values] == [0, 1, 2, 3]
    assert all(in_worker for internal_value, in_worker in values)
    assert metrics.submitted == 4
    assert metrics.completed == 4
    assert metrics.peak_workers > 0
    runtime.shutdown()


def test_runtime_does_not_classify_same_message_runtime_error_as_cancellation() -> None:
    runtime = ExecutionRuntime(RuntimeConfig(parent_workers=1))

    def fail() -> None:
        raise RuntimeError("PDF extraction was cancelled")

    with runtime.task_scope(metrics=True) as context:
        future = context.submit(fail)
        with pytest.raises(RuntimeError, match="PDF extraction was cancelled"):
            future.result(timeout=2)
        metrics = context.metrics()

    assert metrics.completed == 1
    assert metrics.cancelled == 0
    runtime.shutdown()


def test_child_cancellation_is_immutable_and_shares_parent_metrics() -> None:
    runtime = ExecutionRuntime(RuntimeConfig(parent_workers=1))
    stop = threading.Event()

    with runtime.task_scope(metrics=True) as parent:
        child = parent.with_cancellation(stop.is_set)
        assert not parent.cancelled()
        assert not child.cancelled()
        stop.set()
        assert not parent.cancelled()
        assert child.cancelled()
        future = child.submit(lambda: None)
        with pytest.raises(RuntimeError, match="PDF extraction was cancelled"):
            future.result(timeout=2)
        metrics = parent.metrics()

    assert metrics.submitted == 1
    assert metrics.completed == 1
    assert metrics.cancelled == 1
    runtime.shutdown()


def test_document_extraction_chunks_capture_and_parses_native_pages_inline() -> None:
    runtime = ExecutionRuntime(RuntimeConfig(parent_workers=4))
    try:
        page_count = 128
        with PdfDocument.open(internal_many_page_pdf(page_count)) as document:
            with runtime.task_scope(metrics=True) as context:
                extracted = document.extract(context=context)
                metrics = context.metrics()

        assert len(extracted.pages) == page_count
        assert [page.text for page in extracted.pages] == [
            f"Page {page_index}" for page_index in range(page_count)
        ]
        assert metrics.submitted == len(
            parse_pipeline.internal_page_chunks(tuple(range(page_count)), runtime.max_workers)
        )
        assert metrics.submitted < page_count
    finally:
        runtime.shutdown()


def test_close_defers_resource_release_until_operation_finishes() -> None:
    document = PdfDocument.open(SAMPLE_PDF)
    operation = document.acquire_operation()
    document.close()

    assert document.closed
    assert document.raw_data

    operation.release()

    assert document.raw_data == b""
    with pytest.raises(ValueError, match="closed"):
        document.acquire_operation()


def test_resolver_is_safe_for_concurrent_same_object_reads() -> None:
    with PdfDocument.open(SAMPLE_PDF) as document:
        root = document.trailer_dict["Root"]
        with ThreadPoolExecutor(max_workers=4) as executor:
            resolved = list(executor.map(document.resolve, [root] * 16))

    for value in resolved:
        assert isinstance(value, dict)
        assert (PdfName.of("Type"), PdfName.of("Catalog")) in value.items()


def test_same_document_extraction_is_single_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    original = parse_pipeline.internal_PageExtraction.internal_build_parsed_page
    calls = 0
    calls_lock = threading.Lock()

    def counted_parse(
        extraction: parse_pipeline.internal_PageExtraction,
        context: TaskScope,
    ) -> ParsedPage:
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return original(extraction, context)

    monkeypatch.setattr(
        parse_pipeline.internal_PageExtraction,
        "internal_build_parsed_page",
        counted_parse,
    )
    with PdfDocument.open(SAMPLE_PDF) as document:
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda internal_index: document.extract().text, range(4)))

    assert results[0]
    assert results == [results[0]] * 4
    assert calls == 1


def test_document_and_page_share_the_emitted_page() -> None:
    with PdfDocument.open(SAMPLE_PDF) as document:
        extracted_document = document.extract()
        extracted_page = document.pages[0].extract()

    assert extracted_document.pages[0] is extracted_page


def test_concurrent_document_extracts_share_the_emitted_document() -> None:
    with PdfDocument.open(SAMPLE_PDF) as document:
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda internal_index: document.extract(), range(4)))

    assert all(result is results[0] for result in results)


def internal_multi_page_pdf() -> bytes:
    from core_pdf import serialize_document_to_pdf
    from core_pdf.impl.structured import Block, BlockKind, Document, Page, TextLine

    return serialize_document_to_pdf(
        Document(
            pages=tuple(
                Page(
                    page_number=page_number,
                    width=300.0,
                    height=400.0,
                    blocks=(
                        Block(
                            page_number,
                            BlockKind.PARAGRAPH,
                            (TextLine(f"page {page_number} payload"),),
                        ),
                    ),
                )
                for page_number in range(1, 4)
            )
        )
    )


def test_document_extract_parses_only_the_selected_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    original = parse_pipeline.internal_PageExtraction.internal_build_parsed_page
    parsed_page_numbers: list[int] = []

    def counted_parse(
        extraction: parse_pipeline.internal_PageExtraction,
        context: TaskScope,
    ) -> ParsedPage:
        parsed_page_numbers.append(extraction.page.page_number)
        return original(extraction, context)

    monkeypatch.setattr(
        parse_pipeline.internal_PageExtraction,
        "internal_build_parsed_page",
        counted_parse,
    )
    with PdfDocument.open(internal_multi_page_pdf()) as document:
        selected = document.extract(pages=2)
        cached = document.extract(pages=[2])

    assert selected is cached
    assert tuple(page.page_number for page in selected.pages) == (2,)
    assert "page 2 payload" in selected.text
    assert "page 1 payload" not in selected.text
    assert parsed_page_numbers == [2]


def test_distinct_page_selections_can_extract_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = parse_pipeline.internal_PageExtraction.internal_build_parsed_page
    rendezvous = threading.Barrier(2)

    def concurrent_parse(
        extraction: parse_pipeline.internal_PageExtraction,
        context: TaskScope,
    ) -> ParsedPage:
        rendezvous.wait(timeout=3)
        return original(extraction, context)

    monkeypatch.setattr(
        parse_pipeline.internal_PageExtraction,
        "internal_build_parsed_page",
        concurrent_parse,
    )
    with PdfDocument.open(internal_multi_page_pdf()) as document:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(document.extract, pages=1)
            second = executor.submit(document.extract, pages=2)
            results = first.result(timeout=10), second.result(timeout=10)

    assert [page.pages[0].page_number for page in results] == [1, 2]


def test_overlapping_page_selections_share_base_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = parse_pipeline.capture_page
    calls: dict[int, int] = {}
    calls_lock = threading.Lock()

    def counted_capture(page: object) -> parse_pipeline.CapturedPage:
        page_number = int(getattr(page, "page_number"))
        with calls_lock:
            calls[page_number] = calls.get(page_number, 0) + 1
        if page_number == 2:
            time.sleep(0.05)
        return original(page)

    monkeypatch.setattr(parse_pipeline, "capture_page", counted_capture)
    with PdfDocument.open(internal_multi_page_pdf()) as document:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(document.extract, pages=[1, 2])
            second = executor.submit(document.extract, pages=[2, 3])
            results = first.result(timeout=10), second.result(timeout=10)

    assert [tuple(page.page_number for page in result.pages) for result in results] == [
        (1, 2),
        (2, 3),
    ]
    assert calls == {1: 1, 2: 1, 3: 1}
