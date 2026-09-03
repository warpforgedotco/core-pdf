# SPDX-License-Identifier: AGPL-3.0-only
"""Performance bounds for content-operator dispatch in capture and text-only modes."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any, cast

import pytest

from core_pdf.impl.spec.s_07_content.state import TextState
from core_pdf.impl.spec.s_07_syntax.lexer import PdfLexer
from tests.helpers.resolvers import IdentityResolver

# Text-heavy: BT/Tf/Td/Tj/TJ/ET plus the graphics operators around them.
TEXT_BLOCK = (
    b"q 1 0 0 1 72 720 cm 0.2 0.3 0.4 rg BT /F1 12 Tf 14 TL 0 0 Td "
    b"(Hello world) Tj [ (kerned) -250 (text) ] TJ 1 0 0 1 0 -14 Tm "
    b"0.5 Tc 0.25 Tw (more text) Tj ET Q\n" * 40
)

# Graphics-heavy content exercises path construction and painting handlers.
PATH_BLOCK = (
    b"q 2 w 1 J 1 j 10 M [] 0 d 0.9 0.1 0.1 RG "
    b"100 100 m 200 100 l 200 200 l 100 200 l h S "
    b"10 10 120 90 re f 1 0 0 1 5 5 cm Q\n" * 40
)


def internal_state() -> TextState:
    document = cast(
        Any,
        SimpleNamespace(
            resolver=IdentityResolver(),
            internal_cache_lock=threading.RLock(),
            legacy_pdfminer_text_operators=False,
        ),
    )
    return TextState(document, {})


def internal_dispatch(state: TextState, data: bytes) -> None:
    from core_pdf.impl.spec.s_07_content.operations import dispatch_operations

    cast(Any, dispatch_operations)(
        PdfLexer(data),
        state.op_handlers,
        cast(Any, state),
        0,
        operands=state.operands,
    )


@pytest.mark.benchmark_high_impact
def test_dispatch_text_block_capture_benchmark(benchmark) -> None:
    state = internal_state()
    benchmark(internal_dispatch, state, TEXT_BLOCK)


@pytest.mark.benchmark_high_impact
def test_dispatch_path_block_capture_benchmark(benchmark) -> None:
    state = internal_state()
    benchmark(internal_dispatch, state, PATH_BLOCK)


@pytest.mark.benchmark_high_impact
def test_dispatch_path_block_text_only_benchmark(benchmark) -> None:
    """Text-only mode: the skip stage consumes the graphics operators."""
    state = internal_state()
    state.capture_graphics = False
    state.capture_glyphs = False
    state.capture_clipping = False
    benchmark(internal_dispatch, state, PATH_BLOCK)
