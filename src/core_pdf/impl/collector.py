# SPDX-License-Identifier: AGPL-3.0-only

"""Pausing Python's cyclic garbage collector while a page is captured.

A page's capture keeps every glyph, style and drawing it records alive until it ends, so
the collector, which runs after every few thousand new container objects, traverses and
promotes them again and again without freeing any. The capture breaks its own reference
cycles when it ends, so what it leaves is freed by reference counting, and the
collector loses nothing by waiting.

The collector's switch belongs to the whole process. Pauses on any number of threads
share one count: the first turns the collector off, only if it was on, and the last
turns it back on, so a pause never enables a collector its caller had disabled. A thread
that toggles the collector itself while a pause is open can have its change undone when
the last pause ends.
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import wraps
from threading import Lock

_lock = Lock()
_state = {"depth": 0, "resume": False}


@contextmanager
def collector_paused() -> Generator[None]:
    with _lock:
        if _state["depth"] == 0:
            _state["resume"] = gc.isenabled()
            gc.disable()
        _state["depth"] += 1
    try:
        yield
    finally:
        with _lock:
            _state["depth"] -= 1
            if _state["depth"] == 0 and _state["resume"]:
                gc.enable()


def pauses_collector[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    @wraps(function)
    def paused(*args: P.args, **kwargs: P.kwargs) -> R:
        with collector_paused():
            return function(*args, **kwargs)

    return paused
