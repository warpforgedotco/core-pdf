# SPDX-License-Identifier: AGPL-3.0-only
"""Main-thread signal setup for the OCR engine, without importing the engine.

``tesserocr`` pulls in ``cysignals``, which installs signal handlers and so can
only be imported from the main thread. Binding ``tesserocr`` at module scope
guaranteed that, but charged every native-text extraction ~25 ms and dragged in
PIL and the rasterizer for work it never does.

Importing only ``cysignals`` here keeps the same guarantee for a few
milliseconds: once its handlers are installed on the main thread, ``tesserocr``
itself imports fine from an OCR worker, so it can stay on the OCR path.
"""

from __future__ import annotations

import threading
from contextlib import suppress
from importlib import import_module

internal_OCR_SIGNALS_READY = False

internal_MAIN_THREAD_MESSAGE = (
    "core_pdf must initialize OCR on the main thread; import PdfDocument or call "
    "prewarm_runtime() during application startup"
)


def internal_prepare_ocr_signals() -> None:
    """Install cysignals' handlers from the main thread, once per process."""
    global internal_OCR_SIGNALS_READY
    if internal_OCR_SIGNALS_READY:
        return
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(internal_MAIN_THREAD_MESSAGE)
    # Not every tesserocr build depends on cysignals. Where it is absent there
    # is no signal handler to install and therefore no main-thread constraint.
    with suppress(ImportError):
        import_module("cysignals.signals")
    internal_OCR_SIGNALS_READY = True


__all__ = (
    "internal_MAIN_THREAD_MESSAGE",
    "internal_prepare_ocr_signals",
)
