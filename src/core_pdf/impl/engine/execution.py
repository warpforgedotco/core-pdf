# SPDX-License-Identifier: AGPL-3.0-only
"""Process-wide bounded execution shared by parsing and rendering."""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Generic, TypeVar

internal_InputT = TypeVar("internal_InputT")
internal_ResultT = TypeVar("internal_ResultT")


def internal_worker_count() -> int:
    configured = os.environ.get("CORE_PDF_THREADS")
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    # Page parsing can submit OCR work while its page worker waits. A larger
    # pool therefore oversubscribes the same process even before native OCR
    # work is counted; four workers keeps that nested pipeline bounded.
    return max(1, min(4, os.process_cpu_count() or 1))


def internal_raster_budget_bytes() -> int:
    configured = os.environ.get("CORE_PDF_RASTER_BUDGET_MIB")
    if configured:
        try:
            return max(1, int(configured)) * 1024 * 1024
        except ValueError:
            pass
    return 256 * 1024 * 1024


class WorkStage(StrEnum):
    GENERAL = "general"
    PAGE = "page"
    OCR = "ocr"


def internal_ocr_worker_count() -> int:
    configured = os.environ.get("CORE_PDF_OCR_WORKERS")
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    return max(1, min(os.process_cpu_count() or 1, 4))


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Explicit resource limits for the shared thread runtime."""

    parent_workers: int = field(default_factory=internal_worker_count)
    ocr_workers: int = field(default_factory=internal_ocr_worker_count)
    raster_budget_bytes: int = field(default_factory=internal_raster_budget_bytes)
    prewarm: bool = False

    def __post_init__(self) -> None:
        if self.parent_workers < 1:
            raise ValueError("parent_workers must be positive")
        if self.ocr_workers < 1:
            raise ValueError("ocr_workers must be positive")
        if self.raster_budget_bytes < 1:
            raise ValueError("raster_budget_bytes must be positive")


@dataclass(frozen=True, slots=True)
class RuntimeMetrics:
    submitted: int = 0
    completed: int = 0
    cancelled: int = 0
    peak_workers: int = 0
    queue_seconds: float = 0.0
    execution_seconds: float = 0.0
    parent_workers: int = 0
    active_workers: int = 0
    queued_tasks: int = 0
    raster_capacity_bytes: int = 0
    raster_available_bytes: int = 0
    page_capacity: int = 0
    page_active: int = 0
    ocr_capacity: int = 0
    ocr_active: int = 0


@dataclass(frozen=True, slots=True)
class CompletedResult(Generic[internal_ResultT]):
    index: int
    value: internal_ResultT


@dataclass(slots=True)
class internal_QueuedWork:
    future: Future[Any]
    function: Callable[..., Any]
    args: tuple[object, ...]
    kwargs: dict[str, object]
    context: TaskScope | None
    submitted_at: float
    stage: WorkStage


class internal_ResourceBudget:
    """A cancellation-aware weighted semaphore measured in bytes."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, capacity)
        self.internal_available = self.capacity
        self.internal_condition = threading.Condition()

    def acquire(self, amount: int, cancelled: Callable[[], bool]) -> int:
        amount = min(self.capacity, max(1, amount))
        with self.internal_condition:
            while self.internal_available < amount:
                if cancelled():
                    raise RuntimeError("PDF extraction was cancelled")
                self.internal_condition.wait(timeout=0.05)
            self.internal_available -= amount
        return amount

    def release(self, amount: int) -> None:
        with self.internal_condition:
            self.internal_available = min(self.capacity, self.internal_available + amount)
            self.internal_condition.notify_all()

    @property
    def available(self) -> int:
        with self.internal_condition:
            return self.internal_available


class internal_ResourceLease(AbstractContextManager["internal_ResourceLease"]):
    def __init__(
        self,
        budget: internal_ResourceBudget,
        amount: int,
        cancelled: Callable[[], bool],
    ) -> None:
        self.internal_budget = budget
        self.internal_amount = amount
        self.internal_cancelled = cancelled
        self.internal_acquired = 0

    def __enter__(self) -> internal_ResourceLease:
        self.internal_acquired = self.internal_budget.acquire(
            self.internal_amount, self.internal_cancelled
        )
        return self

    def __exit__(self, *internal_args: object) -> None:
        if self.internal_acquired:
            self.internal_budget.release(self.internal_acquired)
            self.internal_acquired = 0


class TaskScope(AbstractContextManager["TaskScope"]):
    """Cancellation and metrics scope sharing the process-wide executor."""

    def __init__(
        self,
        runtime: ExecutionRuntime,
        *,
        cancelled: Callable[[], bool] | None = None,
        metrics: bool = False,
    ) -> None:
        self.runtime = runtime
        self.internal_cancelled = cancelled
        self.internal_metrics_enabled = metrics
        self.internal_lock = threading.Lock()
        self.internal_submitted = 0
        self.internal_completed = 0
        self.internal_cancelled_count = 0
        self.internal_active = 0
        self.internal_peak = 0
        self.internal_queue_seconds = 0.0
        self.internal_execution_seconds = 0.0

    def __enter__(self) -> TaskScope:
        return self

    def __exit__(self, *internal_args: object) -> None:
        self.runtime.internal_close_context(self)

    def internal_bind_cancelled(self, cancelled: Callable[[], bool]) -> Callable[[], bool] | None:
        previous = self.internal_cancelled
        self.internal_cancelled = cancelled
        return previous

    def internal_restore_cancelled(self, cancelled: Callable[[], bool] | None) -> None:
        self.internal_cancelled = cancelled

    def cancelled(self) -> bool:
        return bool(self.internal_cancelled and self.internal_cancelled())

    def raise_if_cancelled(self) -> None:
        if self.cancelled():
            raise RuntimeError("PDF extraction was cancelled")

    def map_ordered(
        self,
        function: Callable[[internal_InputT], internal_ResultT],
        values: Iterable[internal_InputT],
        *,
        stage: WorkStage = WorkStage.GENERAL,
    ) -> Iterator[internal_ResultT]:
        return self.runtime.map_ordered(function, values, context=self, stage=stage)

    def map_completed(
        self,
        function: Callable[[internal_InputT], internal_ResultT],
        values: Iterable[internal_InputT],
        *,
        stage: WorkStage = WorkStage.GENERAL,
    ) -> Generator[CompletedResult[internal_ResultT], None, None]:
        return self.runtime.map_completed(function, values, context=self, stage=stage)

    def submit(
        self,
        function: Callable[..., internal_ResultT],
        /,
        *args: object,
        stage: WorkStage = WorkStage.GENERAL,
        **kwargs: object,
    ) -> Future[internal_ResultT]:
        return self.runtime.submit(function, *args, context=self, stage=stage, **kwargs)

    def reserve_raster(self, amount: int) -> AbstractContextManager[object]:
        return internal_ResourceLease(self.runtime.internal_raster_budget, amount, self.cancelled)

    def metrics(self) -> RuntimeMetrics:
        with self.internal_lock:
            runtime_metrics = self.runtime.metrics()
            return RuntimeMetrics(
                submitted=self.internal_submitted,
                completed=self.internal_completed,
                cancelled=self.internal_cancelled_count,
                peak_workers=self.internal_peak,
                queue_seconds=self.internal_queue_seconds,
                execution_seconds=self.internal_execution_seconds,
                parent_workers=runtime_metrics.parent_workers,
                active_workers=runtime_metrics.active_workers,
                queued_tasks=runtime_metrics.queued_tasks,
                raster_capacity_bytes=runtime_metrics.raster_capacity_bytes,
                raster_available_bytes=runtime_metrics.raster_available_bytes,
                page_capacity=runtime_metrics.page_capacity,
                page_active=runtime_metrics.page_active,
                ocr_capacity=runtime_metrics.ocr_capacity,
                ocr_active=runtime_metrics.ocr_active,
            )

    def internal_record_submit(self) -> None:
        if self.internal_metrics_enabled:
            with self.internal_lock:
                self.internal_submitted += 1

    def internal_record_start(self) -> None:
        if self.internal_metrics_enabled:
            with self.internal_lock:
                self.internal_active += 1
                self.internal_peak = max(self.internal_peak, self.internal_active)

    def internal_record_done(self, queue: float, execution: float, cancelled: bool) -> None:
        if self.internal_metrics_enabled:
            with self.internal_lock:
                self.internal_active = max(0, self.internal_active - 1)
                self.internal_completed += 1
                self.internal_cancelled_count += int(cancelled)
                self.internal_queue_seconds += queue
                self.internal_execution_seconds += execution


class ExecutionRuntime:
    """Fair bounded threads shared by parsing, rendering, and native OCR."""

    internal_default: ExecutionRuntime | None = None
    internal_default_lock = threading.Lock()

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.internal_config = config or RuntimeConfig()
        self.internal_max_workers = self.internal_config.parent_workers
        self.internal_stage_limits = {
            WorkStage.GENERAL: self.internal_max_workers,
            WorkStage.PAGE: self.internal_max_workers,
            WorkStage.OCR: min(
                self.internal_max_workers,
                self.internal_config.ocr_workers,
            ),
        }
        self.internal_stage_active = dict.fromkeys(WorkStage, 0)
        self.internal_executor: ThreadPoolExecutor | None = None
        self.internal_lock = threading.RLock()
        self.internal_local = threading.local()
        self.internal_anonymous_context = object()
        self.internal_pending: dict[object, deque[internal_QueuedWork]] = {}
        self.internal_round_robin: deque[object] = deque()
        self.internal_active = 0
        self.internal_raster_budget = internal_ResourceBudget(
            self.internal_config.raster_budget_bytes
        )
        self.internal_submitted = 0
        self.internal_completed = 0
        self.internal_cancelled_count = 0
        self.internal_peak = 0
        self.internal_queue_seconds = 0.0
        self.internal_execution_seconds = 0.0
        if self.internal_config.prewarm:
            self.prewarm()

    @classmethod
    def default(cls) -> ExecutionRuntime:
        with cls.internal_default_lock:
            if cls.internal_default is None:
                cls.internal_default = cls()
            return cls.internal_default

    @property
    def max_workers(self) -> int:
        return self.internal_max_workers

    @property
    def in_worker(self) -> bool:
        return bool(getattr(self.internal_local, "in_worker", False))

    @property
    def current_stage(self) -> WorkStage | None:
        return getattr(self.internal_local, "stage", None)

    def task_scope(
        self,
        *,
        cancelled: Callable[[], bool] | None = None,
        metrics: bool = False,
    ) -> TaskScope:
        return TaskScope(self, cancelled=cancelled, metrics=metrics)

    def configure(self, config: RuntimeConfig) -> None:
        self.shutdown(wait=True)
        with self.internal_lock:
            self.internal_config = config
            self.internal_max_workers = config.parent_workers
            self.internal_stage_limits = {
                WorkStage.GENERAL: config.parent_workers,
                WorkStage.PAGE: config.parent_workers,
                WorkStage.OCR: min(config.parent_workers, config.ocr_workers),
            }
            self.internal_stage_active = dict.fromkeys(WorkStage, 0)
            self.internal_raster_budget = internal_ResourceBudget(config.raster_budget_bytes)
        if config.prewarm:
            self.prewarm()

    def internal_get_executor(self) -> ThreadPoolExecutor:
        with self.internal_lock:
            if self.internal_executor is None:
                self.internal_executor = ThreadPoolExecutor(
                    max_workers=self.internal_max_workers,
                    thread_name_prefix="core-pdf",
                )
            return self.internal_executor

    def prewarm(self) -> None:
        """Start the shared executor before latency-sensitive work arrives."""
        self.internal_get_executor()

    def run_on_each_worker(
        self,
        function: Callable[[], object],
        *,
        timeout: float = 60.0,
    ) -> int:
        """Run ``function`` once on every pooled worker thread.

        Used to move thread-local setup — a Tesseract API costs a few tens of
        milliseconds per thread — off the critical path of the first page that
        needs it. Each task waits on a barrier so no thread can claim two of
        them and leave another cold.

        A worker already busy with other work never reaches the barrier, so the
        wait is bounded: on timeout the barrier breaks, every queued task runs
        anyway, and the threads that were busy stay cold. ``function`` must
        therefore be idempotent per thread. Returns the number of tasks that
        completed without raising.
        """
        workers = self.internal_max_workers
        executor = self.internal_get_executor()
        barrier = threading.Barrier(workers)

        def internal_warm() -> None:
            with suppress(threading.BrokenBarrierError, threading.ThreadError):
                barrier.wait(timeout=timeout)
            function()

        futures = [executor.submit(internal_warm) for _ in range(workers)]
        warmed = 0
        for future in futures:
            try:
                future.result(timeout=timeout)
            except Exception:
                continue
            warmed += 1
        return warmed

    def internal_run(
        self,
        function: Callable[..., internal_ResultT],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        context: TaskScope | None,
        submitted_at: float,
        stage: WorkStage,
    ) -> internal_ResultT:
        started = time.perf_counter()
        previous = self.in_worker
        previous_stage = self.current_stage
        self.internal_local.in_worker = True
        self.internal_local.stage = stage
        if context is not None:
            context.internal_record_start()
        with self.internal_lock:
            self.internal_peak = max(self.internal_peak, self.internal_active)
        cancelled = False
        try:
            if context is not None:
                context.raise_if_cancelled()
            return function(*args, **kwargs)
        except RuntimeError as exc:
            cancelled = str(exc) == "PDF extraction was cancelled"
            raise
        finally:
            finished = time.perf_counter()
            if context is not None:
                context.internal_record_done(started - submitted_at, finished - started, cancelled)
            with self.internal_lock:
                self.internal_completed += 1
                self.internal_cancelled_count += int(cancelled)
                self.internal_queue_seconds += started - submitted_at
                self.internal_execution_seconds += finished - started
            self.internal_local.in_worker = previous
            self.internal_local.stage = previous_stage

    def submit(
        self,
        function: Callable[..., internal_ResultT],
        /,
        *args: object,
        context: TaskScope | None = None,
        stage: WorkStage = WorkStage.GENERAL,
        **kwargs: object,
    ) -> Future[internal_ResultT]:
        if context is not None:
            context.internal_record_submit()
        submitted_at = time.perf_counter()
        future: Future[internal_ResultT] = Future()
        work = internal_QueuedWork(
            future=future,
            function=function,
            args=args,
            kwargs=dict(kwargs),
            context=context,
            submitted_at=submitted_at,
            stage=stage,
        )
        key: object = context if context is not None else self.internal_anonymous_context
        with self.internal_lock:
            self.internal_submitted += 1
            queue = self.internal_pending.get(key)
            if queue is None:
                queue = deque()
                self.internal_pending[key] = queue
                self.internal_round_robin.append(key)
            queue.append(work)
            self.internal_dispatch_locked()
        return future

    def internal_dispatch_locked(self) -> None:
        while self.internal_active < self.internal_max_workers and self.internal_round_robin:
            selected = self.internal_take_eligible_locked(helping=False)
            if selected is None:
                break
            work, internal_stage_acquired = selected
            self.internal_active += 1
            self.internal_get_executor().submit(self.internal_execute_queued, work)

    def internal_take_eligible_locked(
        self,
        *,
        helping: bool,
        stage: WorkStage | None = None,
    ) -> tuple[internal_QueuedWork, bool] | None:
        """Select one queued task.

        A helping worker is a pool thread blocked on futures of ``stage``. It may
        only run work of that stage: running anything else nests an unrelated
        task on a thread that may still hold that task's exclusive resources (a
        page worker waiting on its OCR groups holds a raster lease; nesting a
        second page parse on it would wait for a lease that only the waiting
        workers hold). Work of the helper's own stage bypasses the stage limit
        because the helper already occupies one of that stage's slots.
        """
        current_stage = self.current_stage if helping else None
        remaining_contexts = len(self.internal_round_robin)
        while remaining_contexts:
            key = self.internal_round_robin.popleft()
            queue = self.internal_pending.get(key)
            if not queue:
                self.internal_pending.pop(key, None)
                remaining_contexts -= 1
                continue
            selected: internal_QueuedWork | None = None
            stage_acquired = False
            for internal_position in range(len(queue)):
                work = queue.popleft()
                if work.future.cancelled():
                    continue
                if helping and work.stage is not stage:
                    queue.append(work)
                    continue
                same_stage_help = helping and current_stage is work.stage
                if same_stage_help or (
                    self.internal_stage_active[work.stage] < self.internal_stage_limits[work.stage]
                ):
                    selected = work
                    stage_acquired = not same_stage_help
                    if stage_acquired:
                        self.internal_stage_active[work.stage] += 1
                    break
                queue.append(work)
            if queue:
                self.internal_round_robin.append(key)
            else:
                self.internal_pending.pop(key, None)
            if selected is not None:
                return selected, stage_acquired
            remaining_contexts -= 1
        return None

    def internal_take_pending_for_help(
        self,
        stage: WorkStage,
    ) -> tuple[internal_QueuedWork, bool] | None:
        """Take one fair queued task of ``stage`` for cooperative execution by a waiting worker."""
        with self.internal_lock:
            return self.internal_take_eligible_locked(helping=True, stage=stage)

    def internal_execute_helped(self, work: internal_QueuedWork, stage_acquired: bool) -> bool:
        if not work.future.set_running_or_notify_cancel():
            if stage_acquired:
                self.internal_stage_slot_released(work.stage)
            return False
        try:
            result = self.internal_run(
                work.function,
                work.args,
                work.kwargs,
                work.context,
                work.submitted_at,
                work.stage,
            )
        except BaseException as exc:
            work.future.set_exception(exc)
        else:
            work.future.set_result(result)
        finally:
            if stage_acquired:
                self.internal_stage_slot_released(work.stage)
        return True

    def internal_help_once(self, stage: WorkStage) -> bool:
        selected = self.internal_take_pending_for_help(stage)
        return selected is not None and self.internal_execute_helped(*selected)

    def internal_result(
        self,
        future: Future[internal_ResultT],
        stage: WorkStage,
    ) -> internal_ResultT:
        if not self.in_worker:
            return future.result()
        while not future.done():
            if not self.internal_help_once(stage):
                wait((future,), timeout=0.001, return_when=FIRST_COMPLETED)
        return future.result()

    def internal_execute_queued(self, work: internal_QueuedWork) -> None:
        if not work.future.set_running_or_notify_cancel():
            self.internal_worker_slot_released(work.stage)
            return
        result: Any = None
        error: BaseException | None = None
        try:
            result = self.internal_run(
                work.function,
                work.args,
                work.kwargs,
                work.context,
                work.submitted_at,
                work.stage,
            )
        except BaseException as exc:
            error = exc
        finally:
            self.internal_worker_slot_released(work.stage)
        if error is not None:
            work.future.set_exception(error)
        else:
            work.future.set_result(result)

    def internal_stage_slot_released(self, stage: WorkStage) -> None:
        with self.internal_lock:
            self.internal_stage_active[stage] = max(0, self.internal_stage_active[stage] - 1)
            self.internal_dispatch_locked()

    def internal_worker_slot_released(self, stage: WorkStage) -> None:
        with self.internal_lock:
            self.internal_active = max(0, self.internal_active - 1)
            self.internal_stage_active[stage] = max(0, self.internal_stage_active[stage] - 1)
            self.internal_dispatch_locked()

    def internal_close_context(self, context: TaskScope) -> None:
        with self.internal_lock:
            queue = self.internal_pending.pop(context, None)
            if queue is None:
                return
            self.internal_round_robin = deque(
                key for key in self.internal_round_robin if key is not context
            )
            for work in queue:
                work.future.cancel()

    def map_ordered(
        self,
        function: Callable[[internal_InputT], internal_ResultT],
        values: Iterable[internal_InputT],
        *,
        context: TaskScope | None = None,
        stage: WorkStage = WorkStage.GENERAL,
    ) -> Iterator[internal_ResultT]:
        iterator = iter(values)
        futures: deque[Future[internal_ResultT]] = deque()
        try:
            while len(futures) < self.internal_max_workers:
                try:
                    value = next(iterator)
                except StopIteration:
                    break
                futures.append(self.submit(function, value, context=context, stage=stage))
            while futures:
                yield self.internal_result(futures.popleft(), stage)
                try:
                    value = next(iterator)
                except StopIteration:
                    continue
                futures.append(self.submit(function, value, context=context, stage=stage))
        finally:
            for future in futures:
                future.cancel()

    def map_completed(
        self,
        function: Callable[[internal_InputT], internal_ResultT],
        values: Iterable[internal_InputT],
        *,
        context: TaskScope | None = None,
        stage: WorkStage = WorkStage.GENERAL,
    ) -> Generator[CompletedResult[internal_ResultT], None, None]:
        iterator = iter(values)
        pending: dict[Future[internal_ResultT], int] = {}
        next_index = 0
        try:
            while len(pending) < self.internal_max_workers:
                try:
                    value = next(iterator)
                except StopIteration:
                    break
                pending[self.submit(function, value, context=context, stage=stage)] = next_index
                next_index += 1
            while pending:
                done = {future for future in pending if future.done()}
                while not done:
                    if self.in_worker and self.internal_help_once(stage):
                        done = {future for future in pending if future.done()}
                        continue
                    done, internal_not_done = wait(
                        tuple(pending),
                        timeout=0.001 if self.in_worker else None,
                        return_when=FIRST_COMPLETED,
                    )
                future = next(iter(done))
                index = pending.pop(future)
                yield CompletedResult(index, self.internal_result(future, stage))
                try:
                    value = next(iterator)
                except StopIteration:
                    continue
                pending[self.submit(function, value, context=context, stage=stage)] = next_index
                next_index += 1
        finally:
            for future in pending:
                future.cancel()

    def metrics(self) -> RuntimeMetrics:
        with self.internal_lock:
            return RuntimeMetrics(
                submitted=self.internal_submitted,
                completed=self.internal_completed,
                cancelled=self.internal_cancelled_count,
                peak_workers=self.internal_peak,
                queue_seconds=self.internal_queue_seconds,
                execution_seconds=self.internal_execution_seconds,
                parent_workers=self.internal_max_workers,
                active_workers=self.internal_active,
                queued_tasks=sum(len(queue) for queue in self.internal_pending.values()),
                raster_capacity_bytes=self.internal_raster_budget.capacity,
                raster_available_bytes=self.internal_raster_budget.available,
                page_capacity=self.internal_stage_limits[WorkStage.PAGE],
                page_active=self.internal_stage_active[WorkStage.PAGE],
                ocr_capacity=self.internal_stage_limits[WorkStage.OCR],
                ocr_active=self.internal_stage_active[WorkStage.OCR],
            )

    def shutdown(self, *, wait: bool = True) -> None:
        with self.internal_lock:
            executor = self.internal_executor
            self.internal_executor = None
            pending = tuple(work for queue in self.internal_pending.values() for work in queue)
            self.internal_pending.clear()
            self.internal_round_robin.clear()
        for work in pending:
            work.future.cancel()
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=False)


RUNTIME = ExecutionRuntime.default()


def configure_runtime(config: RuntimeConfig) -> None:
    RUNTIME.configure(config)


def shutdown_runtime(*, wait: bool = True) -> None:
    RUNTIME.shutdown(wait=wait)


def runtime_metrics() -> RuntimeMetrics:
    return RUNTIME.metrics()


__all__ = (
    "CompletedResult",
    "ExecutionRuntime",
    "RUNTIME",
    "RuntimeConfig",
    "RuntimeMetrics",
    "TaskScope",
    "WorkStage",
    "configure_runtime",
    "runtime_metrics",
    "shutdown_runtime",
)
