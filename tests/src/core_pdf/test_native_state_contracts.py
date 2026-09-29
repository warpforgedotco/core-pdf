# SPDX-License-Identifier: AGPL-3.0-only

"""ContentScanner's native colour, line-state and text-position operators.

The scanner applies these operators to the capture state itself; the spec
handlers stay the reference. Every capture here runs twice, once as shipped
and once with native_state_operators reporting nothing, so that every operator
goes through its handler, and the two must agree bit for bit: the captured
program, the state at every handler the scanner still calls (so between the
native operators, not just after them), and the state at the end.
"""

import random
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from core_pdf import PdfDocument
from core_pdf.impl import capture_recording
from core_pdf.impl.capture_recording import (
    NATIVE_PATH_HANDLERS,
    NATIVE_STATE_HANDLERS,
    TextState,
)
from core_pdf_spec.s_07_content.model import GraphicsState
from tests.src.core_pdf.pdf_bytes import one_page_pdf

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
ADDRESS = re.compile(r"0x[0-9a-f]+")

CORPUS = (
    ("pypdf/resources/issue-301.pdf", 1),
    ("PyMuPDF/tests/resources/test_2885.pdf", 2),
    ("SCORE-Bench/src/fhhd0346-p009.pdf", 1),
    ("llama_index/docs/examples/data/10k/lyft_2021.pdf", 3),
    ("PyMuPDF/tests/resources/test_3362.pdf", 1),
    ("pdfminer.six/samples/nonfree/i1040nr.pdf", 1),
    ("llama_index/docs/examples/query_engine/pdf_tables/billionaires_page.pdf", 1),
    ("PyMuPDF/tests/resources/test_3806.pdf", 1),
    ("PyMuPDF/tests/resources/test_2904.pdf", 1),
    ("pdfminer.six/samples/nonfree/kampo.pdf", 2),
    ("pdfminer.six/samples/nonfree/175.pdf", 2),
    ("pdfminer.six/samples/contrib/issue_566_test_2.pdf", 1),
    ("SCORE-Bench/src/global-AIDS-strategy-p74-75-p001.pdf", 1),
    ("SCORE-Bench/src/EPA_DCWaterQuality_Tables_Equations-p007.pdf", 1),
    ("SCORE-Bench/src/s12940-025-01154-x-p001.pdf", 1),
    ("pdf20examples/pdf20-utf8-test.pdf", 1),
    ("pypdf/sample-files/015-arabic/habibi-rotated.pdf", 1),
    ("PyMuPDF/tests/resources/test_2634.pdf", 1),
    ("PyMuPDF/tests/resources/test_2954.pdf", 1),
    ("PyMuPDF/tests/resources/test-4055.pdf", 1),
    ("pdfplumber/tests/pdfs/issue-71-duplicate-chars-2.pdf", 1),
    ("PyMuPDF/tests/resources/test_4928.pdf", 1),
    ("PyMuPDF/tests/resources/test_3725.pdf", 1),
)

GRAPHICS_FIELDS = tuple(
    sorted({name for kind in GraphicsState.__mro__ for name in getattr(kind, "__slots__", ())})
)
NATIVE_NAMES = frozenset(NATIVE_STATE_HANDLERS) | frozenset(NATIVE_PATH_HANDLERS)


DESCRIBED: dict[int, tuple[object, object]] = {}


def exact(value: object) -> object:
    """A key equal only for values of the same type and the same bits."""
    kind = type(value)
    if isinstance(value, float) and kind is float:
        return value.hex()
    if kind is int or kind is bool or kind is str or value is None:
        return (kind.__name__, value)
    if isinstance(value, (tuple, list)):
        return (kind.__name__, tuple(exact(item) for item in value))
    # Resources, decoders, colour spaces: the objects themselves are never the
    # scanner's, and their reprs are costly, so each is described once.
    entry = DESCRIBED.get(id(value))
    if entry is None or entry[0] is not value:
        entry = DESCRIBED[id(value)] = (value, (kind.__name__, ADDRESS.sub("0x", repr(value))))
    return entry[1]


def graphics_key(graphics: object) -> object:
    return tuple((name, exact(getattr(graphics, name))) for name in GRAPHICS_FIELDS)


def state_key(state: Any) -> object:
    layout = state.text_layout
    return (
        graphics_key(state.graphics),
        # Deeper saves are compared when a Q brings them back.
        len(state.stack),
        graphics_key(state.stack[-1]) if state.stack else None,
        exact(state.text_matrix),
        exact(state.line_matrix),
        exact(state.current_point),
        exact(state.subpath_start),
        state.shared_glyph_paint is None,
        None if layout is None else layout.style is None,
        state.in_text_object,
        state.pending_line_break,
        state.sequence,
        len(state.runs),
        len(state.text_boundaries),
        state.type3_uncolored,
        len(state.capture_graphics_stack),
        exact(state.clip_bbox),
        exact(state.group_alpha),
    )


def captured(data: bytes, pages: int, *, native: bool, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The program each page captures, and the state at each handler call."""
    trace: list[object] = []
    frames: list[bool] = []
    table = TextState.operation_table
    choose = capture_recording.native_state_operators

    def recorded_table(self: Any) -> dict[str, Any]:
        handlers = dict(table(self))
        for name, handler in handlers.items():
            if name not in NATIVE_NAMES:
                handlers[name] = observed(self, name, handler)
        return handlers

    def observed(state: Any, name: str, handler: Any) -> Any:
        def call(operands: Any, depth: int) -> Any:
            trace.append((name, state_key(state)))
            return handler(operands, depth)

        return call

    complete = TextState.captured_program

    def completed(self: Any, *args: Any, **kwargs: Any) -> Any:
        trace.append(("captured_program", state_key(self)))
        return complete(self, *args, **kwargs)

    def chosen(handlers: Any, state: object) -> frozenset[str]:
        operators = choose(handlers, state)
        frames.append(bool(operators))
        return operators if native else frozenset()

    DESCRIBED.clear()
    with monkeypatch.context() as patch:
        patch.setattr(TextState, "operation_table", recorded_table)
        patch.setattr(TextState, "captured_program", completed)
        patch.setattr(capture_recording, "native_state_operators", chosen)
        programs = []
        with PdfDocument(data) as document:
            for page in document.pages[:pages]:
                body = page.get_page_program().body
                programs.append(
                    (
                        ADDRESS.sub("0x", repr(body.runs)),
                        ADDRESS.sub("0x", repr(body.glyphs)),
                        ADDRESS.sub("0x", repr(body.drawings)),
                        body.lines.array.tobytes(),
                        ADDRESS.sub("0x", repr(body.text_boundaries)),
                    )
                )
    return programs, trace, frames


def assert_same_capture(data: bytes, pages: int, monkeypatch: pytest.MonkeyPatch) -> bool:
    native, native_trace, frames = captured(data, pages, native=True, monkeypatch=monkeypatch)
    handled, handled_trace, _ = captured(data, pages, native=False, monkeypatch=monkeypatch)
    assert len(native_trace) == len(handled_trace)
    for index, (left, right) in enumerate(zip(native_trace, handled_trace, strict=True)):
        assert left == right, f"state differs before handler call {index} ({left[0]})"
    assert native == handled
    return any(frames)


@pytest.mark.parametrize(("name", "pages"), CORPUS, ids=[name for name, _ in CORPUS])
def test_corpus_captures_as_the_handlers_do(
    name: str, pages: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = FIXTURES / name
    if not path.exists():
        pytest.skip("reference corpus not checked out")
    assert assert_same_capture(path.read_bytes(), pages, monkeypatch)


SCANNED_NUMBERS = (
    b"0.1",
    b"0.2",
    b"0.3",
    b"0",
    b"1",
    b"-1",
    b"-0",
    b"+3",
    b"0.5",
    b"-0.0",
    b"+.5",
    b"-.25",
    b"1.",
    b"2",
    b"12",
    b"3.25",
    b"-2.5",
    b"0.3333333",
    b"1000",
    b"123456789012345",
)
# Numbers the scanner hands to the Python lexer: too long, or not PDF syntax.
NUMBERS = (
    *SCANNED_NUMBERS,
    b"99999999999999999",
    b"0.1234567890123456",
    b"1e5",
    b"1.5.5",
)
OTHER_OPERANDS = (
    b"/N",
    b"/DeviceRGB",
    b"(s)",
    b"<41>",
    b"[]",
    b"[3]",
    b"[1 2]",
    b"[0.5 -0 2.]",
    b"[1 (a)]",
    b"[/N]",
    b"[99999999999999999]",
    b"[" + b"9" * 400 + b"]",
    b"[" + b"9" * 400 + b".0]",
    b"[1 [2]]",
    b"true",
    b"null",
)
ARITY = {
    "G": 1,
    "g": 1,
    "RG": 3,
    "rg": 3,
    "K": 4,
    "k": 4,
    "w": 1,
    "J": 1,
    "j": 1,
    "M": 1,
    "Td": 2,
    "TD": 2,
    "Tm": 6,
    "T*": 0,
    "TL": 1,
    "q": 0,
    "Q": 0,
    "cm": 6,
    "m": 2,
    "l": 2,
    "c": 6,
    "re": 4,
    "h": 0,
    "S": 0,
    "f": 0,
    "n": 0,
    "W": 0,
    "Tc": 1,
    "Tw": 1,
    "Tz": 1,
    "Ts": 1,
    "i": 1,
    "d0": 2,
    "d1": 6,
    "BT": 0,
    "ET": 0,
}
TEXT = (b"(ab) Tj", b"[(a) -120 (b)] TJ", b"(c) '", b'1 2 (d) "', b"/F1 9 Tf", b"/F1 12 Tf")


def random_content(rng: random.Random) -> bytes:
    parts = [b"BT /F1 10 Tf"] if rng.random() < 0.8 else []
    for _ in range(rng.randint(1, 40)):
        roll = rng.random()
        if roll < 0.15:
            parts.append(rng.choice(TEXT))
            continue
        if roll < 0.25:
            dash = rng.choice(OTHER_OPERANDS[4:14] + (b"5",))
            phase = rng.choice(SCANNED_NUMBERS + (b"",))
            parts.append(dash + b" " + phase + b" d")
            continue
        name = rng.choice(tuple(ARITY))
        if rng.random() < 0.7:
            operands = [rng.choice(SCANNED_NUMBERS) for _ in range(ARITY[name])]
        else:
            operands = [
                rng.choice(OTHER_OPERANDS) if rng.random() < 0.3 else rng.choice(NUMBERS)
                for _ in range(rng.randint(0, 7))
            ]
        parts.append(b" ".join([*operands, name.encode()]))
    return b" ".join(parts)


@pytest.mark.parametrize("seed", range(8))
def test_random_operands_capture_as_the_handlers_do(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = random.Random(seed)
    for _ in range(60):
        content = random_content(rng)
        try:
            assert assert_same_capture(one_page_pdf(content), 1, monkeypatch)
        except AssertionError as error:
            raise AssertionError(f"{content!r}: {error}") from error


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(
            b"1 0 0 RG 0.5 g 0 0 0 1 k 0.2 G 1 1 0 0 K 0 1 0 rg 0 0 m 5 5 l S", id="colours"
        ),
        pytest.param(b"2 0 0 RG -1 2 0.5 rg 0 0 m 5 5 l B", id="colour-clamps"),
        pytest.param(b"-0.0 g -0 G 1.0 1. 1 RG 0 0 m 5 5 l B", id="negative-zero-colour"),
        pytest.param(b"3 w 1 J 2 j 4 M [3 1] 0.5 d 0 0 m 50 0 l S", id="line-state"),
        pytest.param(b"-2 w 0.5 M -0.0 w 0 M [] 0 d [] -0.0 d 0 0 m 5 0 l S", id="line-clamps"),
        pytest.param(b"1.5 J 2.0 j [1 (a)] 0 d [2] 3 d 0 0 m 5 0 l S", id="declined-line-state"),
        pytest.param(
            b"BT /F1 12 Tf 12 TL 10 20 Td (a) Tj T* (b) Tj 0 -14 TD (c) Tj T* (d) Tj ET",
            id="text-lines",
        ),
        pytest.param(
            b"BT /F1 12 Tf 1 0 0 1 20 30 Tm (a) Tj 2 0 0.5 1 5 5 Tm 3 4 Td (b) Tj T* (c) Tj ET",
            id="text-matrix",
        ),
        pytest.param(
            # (0.1 + 0.2) + 0.3 is not 0.1 + (0.2 + 0.3): the handler's order is kept.
            b"BT /F1 12 Tf 0.1 0.1 0.2 0.2 0.3 0.3 Tm 1 1 Td (a) Tj 0.7 -1.1 TD T* (b) Tj ET",
            id="text-rounding",
        ),
        pytest.param(
            b"BT /F1 12 Tf -0.0 -0 Td (a) Tj -1 0 0 -1 -0.0 0 Tm 0 0 Td (b) Tj ET", id="text-zero"
        ),
        pytest.param(
            b"BT /F1 12 Tf 1 2 3 Td 1 Td 1 2 3 4 5 Tm 1 2 3 4 5 6 7 Tm (x) Tj ET", id="text-arity"
        ),
        pytest.param(b"0 0 6 0 0 d1 1 0 0 rg 0.5 G 0 0 m 5 5 l B", id="type3-uncolored"),
        pytest.param(b"q 1 0 0 rg 3 w Q 0 0 m 5 5 l B q 2 0 0 2 5 5 cm 10 w Q", id="saved-state"),
        pytest.param(
            # 0.1 * 0.1 + 0.1 * 0.1 + 0.1 rounds differently grouped the other way.
            b"0.1 0 0.1 1 0.1 0 cm 1 0 0 1 0.1 0.1 cm 0.7 -0.8 0.9 1.1 0.3 0.1 cm 0 0 m 5 5 l S "
            b"1 0 0 1 0 0 cm 1 0 -0.0 1 0 0 cm 2 0 0 2 1 1 cm 0 0 m 5 5 c S",
            id="concatenation",
        ),
        pytest.param(
            b"Q q q 0.5 g 1 0 0 1 3 3 cm Q 0 0 m 5 5 l S Q Q 0 0 m 5 5 l S 1 2 cm q 1 Q",
            id="unbalanced-saves",
        ),
    ],
)
def test_native_state_captures_as_the_handlers_do(
    content: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert assert_same_capture(one_page_pdf(content), 1, monkeypatch)


def test_every_native_operator_is_applied_without_its_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = (
        b"q 2 0 0 2 5 5 cm 0.5 G 0.5 g 1 0 0 RG 0 1 0 rg 0 0 0 1 K 0 0 0 1 k 2 w [1 2] 0 d "
        b"1 J 1 j 3 M BT /F1 9 Tf 12 TL 1 2 Td 1 -12 TD 1 0 0 1 5 5 Tm T* (a) Tj ET Q"
    )
    codes = {
        getattr(function, "__code__", None): name
        for name, function in NATIVE_STATE_HANDLERS.items()
    }
    dispatch = capture_recording.CaptureStreamExecutor.dispatch_frame.__code__
    called: set[str] = set()

    def profile(frame: Any, event: str, arg: object) -> None:
        # Handlers the scanner calls are called from dispatch_frame's frame;
        # capture also calls some itself, around forms and pages.
        if event == "call" and frame.f_code in codes and frame.f_back.f_code is dispatch:
            called.add(codes[frame.f_code])

    sys.setprofile(profile)
    try:
        with PdfDocument(one_page_pdf(content)) as document:
            document.pages[0].get_page_program()
    finally:
        sys.setprofile(None)
    assert called == set()
    monkeypatch.setattr(capture_recording, "native_state_operators", lambda *_: frozenset())
    sys.setprofile(profile)
    try:
        with PdfDocument(one_page_pdf(content)) as document:
            document.pages[0].get_page_program()
    finally:
        sys.setprofile(None)
    assert called == set(NATIVE_STATE_HANDLERS)


def test_an_overridden_handler_keeps_its_handler() -> None:
    class Overriding(TextState):
        def op_w(self, operands: Any, depth: int) -> None:
            pass

    with PdfDocument(one_page_pdf(b"1 w")) as document:
        state = Overriding(document)
        operators = capture_recording.native_state_operators(state.operation_table(), state)
    assert "w" not in operators
    assert "RG" in operators
