import json
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from core_pdf.impl.document import PdfDocument
from core_pdf.impl.extract import pipeline as parse_pipeline
from core_pdf.impl.extract import selection as parse_selection
from core_pdf.impl.output import Page
from core_pdf.impl.primitives import PdfName
from core_pdf.impl.runtime.execution import ExecutionRuntime, RuntimeConfig, TaskScope, WorkStage
from tests.helpers.paths import SCORE_BENCH
from tests.helpers.pdf_bytes import text_pages_pdf

SAMPLE_PDF = SCORE_BENCH / "global-AIDS-strategy-p74-75-p001.pdf"
PAGE_OCR_PDF = SCORE_BENCH / "SFG-Content-Marketing-2021-p001.pdf"


def test_ocr_extraction_can_start_in_an_application_worker() -> None:
    """Both OCR entry points (image regions and the primary page) start off-main-thread.

    One interpreter runs both fixtures: the property under test is that the
    engine initialises from a worker thread, which one process demonstrates.
    """
    cases = {
        str(SAMPLE_PDF): "image-regions",
        str(PAGE_OCR_PDF): "primary-page",
    }
    script = textwrap.dedent(
        f"""
        import json
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path
        from core_pdf import PdfDocument
        from core_pdf.impl.extract.pipeline import internal_PageExtraction

        def extract(fixture):
            with PdfDocument.open(Path(fixture)) as document:
                extracted = document.extract()
                return {{
                    "characters": len(extracted.text),
                    "passes": [
                        item.name
                        for item in internal_PageExtraction(document.pages[0]).plan().ocr_passes
                    ],
                    "worker": threading.current_thread() is not threading.main_thread(),
                }}

        with ThreadPoolExecutor(max_workers=1) as executor:
            results = [executor.submit(extract, fixture).result() for fixture in {list(cases)!r}]
        print(json.dumps(results))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=True,
        text=True,
    )

    for result, expected_pass in zip(json.loads(completed.stdout), cases.values(), strict=True):
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
        runtime.run_on_each_worker(record, timeout=0.1)
        assert time.perf_counter() - started < 5.0
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

    assert values == list(range(8))
    assert peak == 2
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


def test_runtime_preserves_same_message_runtime_error() -> None:
    runtime = ExecutionRuntime(RuntimeConfig(parent_workers=1))

    def fail() -> None:
        raise RuntimeError("PDF extraction was cancelled")

    with runtime.task_scope() as context:
        future = context.submit(fail)
        with pytest.raises(RuntimeError, match="PDF extraction was cancelled"):
            future.result(timeout=2)

    runtime.shutdown()


def test_child_cancellation_is_immutable_and_shares_parent_queue() -> None:
    runtime = ExecutionRuntime(RuntimeConfig(parent_workers=1))
    stop = threading.Event()

    with runtime.task_scope() as parent:
        child = parent.with_cancellation(stop.is_set)
        assert not parent.cancelled()
        assert not child.cancelled()
        stop.set()
        assert not parent.cancelled()
        assert child.cancelled()
        future = child.submit(lambda: None)
        with pytest.raises(RuntimeError, match="PDF extraction was cancelled"):
            future.result(timeout=2)

    runtime.shutdown()


def test_document_extraction_chunks_capture_and_parses_native_pages_inline() -> None:
    runtime = ExecutionRuntime(RuntimeConfig(parent_workers=4))
    try:
        page_count = 128
        pages = tuple(f"Page {page_index}" for page_index in range(page_count))
        with PdfDocument.open(text_pages_pdf(pages)) as document:
            with runtime.task_scope() as context:
                extracted = document.extract(context=context)

        assert len(extracted.pages) == page_count
        assert [page.text for page in extracted.pages] == [
            f"Page {page_index}" for page_index in range(page_count)
        ]
        chunks = parse_selection.internal_page_chunks(tuple(range(page_count)), runtime.max_workers)
        assert len(chunks) < page_count
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


def test_concurrent_same_document_extractions_are_consistent() -> None:
    with PdfDocument.open(SAMPLE_PDF) as document:
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda internal_index: document.extract(), range(4)))

    assert results[0].text
    assert results == [results[0]] * 4


def test_document_and_page_emit_the_same_page() -> None:
    with PdfDocument.open(SAMPLE_PDF) as document:
        extracted_document = document.extract()
        extracted_page = document.pages[0].extract()

    assert extracted_document.pages[0] == extracted_page


def internal_multi_page_pdf() -> bytes:
    return text_pages_pdf(tuple(f"page {page_number} payload" for page_number in range(1, 4)))


def test_document_extract_parses_only_the_selected_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    original = parse_pipeline.internal_PageExtraction.assembled_page
    parsed_page_numbers: list[int] = []

    def counted_assembly(
        extraction: parse_pipeline.internal_PageExtraction,
        context: TaskScope,
    ) -> Page:
        parsed_page_numbers.append(extraction.page.page_number)
        return original(extraction, context)

    monkeypatch.setattr(
        parse_pipeline.internal_PageExtraction,
        "assembled_page",
        counted_assembly,
    )
    with PdfDocument.open(internal_multi_page_pdf()) as document:
        selected = document.extract(pages=2)

    assert tuple(page.page_number for page in selected.pages) == (2,)
    assert "page 2 payload" in selected.text
    assert "page 1 payload" not in selected.text
    assert parsed_page_numbers == [2]


def test_distinct_page_selections_can_extract_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = parse_pipeline.internal_PageExtraction.assembled_page
    rendezvous = threading.Barrier(2)

    def concurrent_assembly(
        extraction: parse_pipeline.internal_PageExtraction,
        context: TaskScope,
    ) -> Page:
        rendezvous.wait(timeout=3)
        return original(extraction, context)

    monkeypatch.setattr(
        parse_pipeline.internal_PageExtraction,
        "assembled_page",
        concurrent_assembly,
    )
    with PdfDocument.open(internal_multi_page_pdf()) as document:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(document.extract, pages=1)
            second = executor.submit(document.extract, pages=2)
            results = first.result(timeout=10), second.result(timeout=10)

    assert [page.pages[0].page_number for page in results] == [1, 2]


def test_overlapping_page_selections_extract_independently(
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
    assert calls == {1: 1, 2: 2, 3: 1}
