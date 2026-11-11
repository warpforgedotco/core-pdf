# SPDX-License-Identifier: AGPL-3.0-only

"""Check that ContentScanner's native state operators capture exactly as the spec handlers do.

The scanner applies colour, line-state, text-position, q/Q/cm and path operators to the
capture state itself when core_pdf.impl.capture_recording.native_state_operators finds the
spec handlers unoverridden; the handlers stay the reference. Every capture here runs twice,
once as shipped and once with native_state_operators reporting nothing, so that every
operator goes through its handler, and the two must agree bit for bit: the captured program,
the state at every handler the scanner still calls (so between the native operators, not
just after them), and the state at the end. The captures are the first pages of upstream test
files, content written around the operators' edge cases (clamps, negative zeros, rounding
order, declined operands, unbalanced saves), and random operator sequences; then the native
operators must be applied without their handlers, and an overridden handler keeps its own.

uv run --all-packages python scripts/check_native_state_operators.py [random contents]
"""

from __future__ import annotations

import contextlib
import logging
import random
import re
import sys
import warnings
from collections.abc import Generator
from pathlib import Path
from typing import Any

from core_pdf import PdfDocument
from core_pdf.impl import capture_recording
from core_pdf.impl.capture_recording import (
    NATIVE_PATH_HANDLERS,
    NATIVE_STATE_HANDLERS,
    TextState,
)
from core_pdf_spec.s_07_content.model import GraphicsState

UPSTREAM = Path("tests/upstream")
# Files the native operators were first checked on, with the pages to capture.
NAMED = (
    ("pypdf/resources/issue-301.pdf", 1),
    ("llama_index/docs/examples/data/10k/lyft_2021.pdf", 3),
    ("pdfminer.six/samples/nonfree/i1040nr.pdf", 1),
    ("llama_index/docs/examples/query_engine/pdf_tables/billionaires_page.pdf", 1),
    ("pdfminer.six/samples/nonfree/kampo.pdf", 2),
    ("pdfminer.six/samples/nonfree/175.pdf", 2),
    ("pdfminer.six/samples/contrib/issue_566_test_2.pdf", 1),
    ("pypdf/sample-files/015-arabic/habibi-rotated.pdf", 1),
    ("pdfplumber/tests/pdfs/issue-71-duplicate-chars-2.pdf", 1),
)
# And the first page of each of these.
FIRST_PAGES = ("pdfplumber/tests/pdfs/*.pdf", "pdfminer.six/samples/*.pdf")
ADDRESS = re.compile(r"0x[0-9a-f]+")
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
    # Resources, decoders, colour spaces: the objects themselves are never the scanner's,
    # and their reprs are costly, so each is described once.
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


@contextlib.contextmanager
def patched(owner: object, name: str, value: object) -> Generator[None]:
    original = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, original)


def captured(
    data: bytes, pages: int, *, native: bool
) -> tuple[Any, list[tuple[str, object]], list[bool]]:
    """The program each page captures, the state at each handler call, and whether any
    content stream had native operators."""
    trace: list[tuple[str, object]] = []
    frames: list[bool] = []
    table = TextState.operation_table
    choose = capture_recording.native_state_operators
    complete = TextState.captured_program

    def observed(state: Any, name: str, handler: Any) -> Any:
        def call(operands: Any, depth: int) -> Any:
            trace.append((name, state_key(state)))
            return handler(operands, depth)

        return call

    def recorded_table(self: Any) -> dict[str, Any]:
        handlers = dict(table(self))
        for name, handler in handlers.items():
            if name not in NATIVE_NAMES:
                handlers[name] = observed(self, name, handler)
        return handlers

    def completed(self: Any, *args: Any, **kwargs: Any) -> Any:
        trace.append(("captured_program", state_key(self)))
        return complete(self, *args, **kwargs)

    def chosen(handlers: Any, state: object) -> frozenset[str]:
        operators = choose(handlers, state)
        frames.append(bool(operators))
        return operators if native else frozenset()

    DESCRIBED.clear()
    programs = []
    with (
        patched(TextState, "operation_table", recorded_table),
        patched(TextState, "captured_program", completed),
        patched(capture_recording, "native_state_operators", chosen),
        PdfDocument(data) as document,
    ):
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


def difference(data: bytes, pages: int) -> str | None:
    """Where the native and handled captures part, or None when they agree."""
    native, native_trace, frames = captured(data, pages, native=True)
    handled, handled_trace, _ = captured(data, pages, native=False)
    if len(native_trace) != len(handled_trace):
        return f"{len(native_trace)} handler calls natively, {len(handled_trace)} handled"
    for index, (left, right) in enumerate(zip(native_trace, handled_trace, strict=True)):
        if left != right:
            return f"state differs before handler call {index} ({left[0]})"
    if native != handled:
        return "captured programs differ"
    if not any(frames):
        return "no content stream took the native operators"
    return None


def serialize_pdf(objects: dict[int, bytes]) -> bytes:
    data = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for number in range(1, len(objects) + 1):
        offsets[number] = len(data)
        data += b"%d 0 obj\n" % number + objects[number] + b"\nendobj\n"
    xref = len(data)
    size = len(objects) + 1
    data += b"xref\n0 %d\n0000000000 65535 f \n" % size
    data += b"".join(b"%010d 00000 n \n" % offsets[number] for number in range(1, size))
    data += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, xref)
    return bytes(data)


def one_page_pdf(content: bytes) -> bytes:
    return serialize_pdf(
        {
            1: b"<< /Type /Catalog /Pages 2 0 R >>",
            2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
            4: b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
            5: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        }
    )


SCANNED_NUMBERS = (
    b"0.1", b"0.2", b"0.3", b"0", b"1", b"-1", b"-0", b"+3", b"0.5", b"-0.0", b"+.5",
    b"-.25", b"1.", b"2", b"12", b"3.25", b"-2.5", b"0.3333333", b"1000",
    b"123456789012345",
)  # fmt: skip
# Numbers the scanner hands to the Python lexer: too long, or not PDF syntax.
NUMBERS = (*SCANNED_NUMBERS, b"99999999999999999", b"0.1234567890123456", b"1e5", b"1.5.5")
OTHER_OPERANDS = (
    b"/N", b"/DeviceRGB", b"(s)", b"<41>", b"[]", b"[3]", b"[1 2]", b"[0.5 -0 2.]",
    b"[1 (a)]", b"[/N]", b"[99999999999999999]", b"[" + b"9" * 400 + b"]",
    b"[" + b"9" * 400 + b".0]", b"[1 [2]]", b"true", b"null",
)  # fmt: skip
ARITY = {
    "G": 1, "g": 1, "RG": 3, "rg": 3, "K": 4, "k": 4, "w": 1, "J": 1, "j": 1, "M": 1,
    "Td": 2, "TD": 2, "Tm": 6, "T*": 0, "TL": 1, "q": 0, "Q": 0, "cm": 6, "m": 2, "l": 2,
    "c": 6, "re": 4, "h": 0, "S": 0, "f": 0, "n": 0, "W": 0, "Tc": 1, "Tw": 1, "Tz": 1,
    "Ts": 1, "i": 1, "d0": 2, "d1": 6, "BT": 0, "ET": 0,
}  # fmt: skip
TEXT = (b"(ab) Tj", b"[(a) -120 (b)] TJ", b"(c) '", b'1 2 (d) "', b"/F1 9 Tf", b"/F1 12 Tf")
EDGE_CASES = {
    "colours": b"1 0 0 RG 0.5 g 0 0 0 1 k 0.2 G 1 1 0 0 K 0 1 0 rg 0 0 m 5 5 l S",
    "colour-clamps": b"2 0 0 RG -1 2 0.5 rg 0 0 m 5 5 l B",
    "negative-zero-colour": b"-0.0 g -0 G 1.0 1. 1 RG 0 0 m 5 5 l B",
    "line-state": b"3 w 1 J 2 j 4 M [3 1] 0.5 d 0 0 m 50 0 l S",
    "line-clamps": b"-2 w 0.5 M -0.0 w 0 M [] 0 d [] -0.0 d 0 0 m 5 0 l S",
    "declined-line-state": b"1.5 J 2.0 j [1 (a)] 0 d [2] 3 d 0 0 m 5 0 l S",
    "text-lines": b"BT /F1 12 Tf 12 TL 10 20 Td (a) Tj T* (b) Tj 0 -14 TD (c) Tj T* (d) Tj ET",
    "text-matrix": (
        b"BT /F1 12 Tf 1 0 0 1 20 30 Tm (a) Tj 2 0 0.5 1 5 5 Tm 3 4 Td (b) Tj T* (c) Tj ET"
    ),
    # (0.1 + 0.2) + 0.3 is not 0.1 + (0.2 + 0.3): the handler's order is kept.
    "text-rounding": (
        b"BT /F1 12 Tf 0.1 0.1 0.2 0.2 0.3 0.3 Tm 1 1 Td (a) Tj 0.7 -1.1 TD T* (b) Tj ET"
    ),
    "text-zero": b"BT /F1 12 Tf -0.0 -0 Td (a) Tj -1 0 0 -1 -0.0 0 Tm 0 0 Td (b) Tj ET",
    "text-arity": b"BT /F1 12 Tf 1 2 3 Td 1 Td 1 2 3 4 5 Tm 1 2 3 4 5 6 7 Tm (x) Tj ET",
    "type3-uncolored": b"0 0 6 0 0 d1 1 0 0 rg 0.5 G 0 0 m 5 5 l B",
    "saved-state": b"q 1 0 0 rg 3 w Q 0 0 m 5 5 l B q 2 0 0 2 5 5 cm 10 w Q",
    # 0.1 * 0.1 + 0.1 * 0.1 + 0.1 rounds differently grouped the other way.
    "concatenation": (
        b"0.1 0 0.1 1 0.1 0 cm 1 0 0 1 0.1 0.1 cm 0.7 -0.8 0.9 1.1 0.3 0.1 cm 0 0 m 5 5 l S "
        b"1 0 0 1 0 0 cm 1 0 -0.0 1 0 0 cm 2 0 0 2 1 1 cm 0 0 m 5 5 c S"
    ),
    "unbalanced-saves": (
        b"Q q q 0.5 g 1 0 0 1 3 3 cm Q 0 0 m 5 5 l S Q Q 0 0 m 5 5 l S 1 2 cm q 1 Q"
    ),
}


def random_content(rng: random.Random) -> bytes:
    parts = [b"BT /F1 10 Tf"] if rng.random() < 0.8 else []
    for _ in range(rng.randint(1, 40)):
        roll = rng.random()
        if roll < 0.15:
            parts.append(rng.choice(TEXT))
            continue
        if roll < 0.25:
            dash = rng.choice((*OTHER_OPERANDS[4:14], b"5"))
            phase = rng.choice((*SCANNED_NUMBERS, b""))
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


def handlers_called(content: bytes, *, native: bool) -> set[str]:
    """The native operators' handlers the scanner calls while capturing content."""
    codes = {
        getattr(function, "__code__", None): name
        for name, function in NATIVE_STATE_HANDLERS.items()
    }
    dispatch = capture_recording.CaptureStreamExecutor.dispatch_frame.__code__
    called: set[str] = set()

    def profile(frame: Any, event: str, _arg: object) -> None:
        # Handlers the scanner calls are called from dispatch_frame's frame; capture also
        # calls some itself, around forms and pages.
        if event == "call" and frame.f_code in codes and frame.f_back.f_code is dispatch:
            called.add(codes[frame.f_code])

    with contextlib.ExitStack() as stack:
        if not native:
            stack.enter_context(
                patched(capture_recording, "native_state_operators", lambda *_: frozenset())
            )
        sys.setprofile(profile)
        try:
            with PdfDocument(one_page_pdf(content)) as document:
                document.pages[0].get_page_program()
        finally:
            sys.setprofile(None)
    return called


def main(argv: list[str]) -> int:
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    count = int(argv[0]) if argv else 480
    failures: list[str] = []
    files = dict(NAMED)
    for pattern in FIRST_PAGES:
        for path in sorted(UPSTREAM.glob(pattern)):
            files.setdefault(str(path.relative_to(UPSTREAM)), 1)
    checked = 0
    for name, pages in files.items():
        path = UPSTREAM / name
        if not path.exists():
            continue
        try:
            found = difference(path.read_bytes(), pages)
        except Exception as error:  # noqa: BLE001
            if name not in dict(NAMED):
                # A test file core cannot capture is no test of the operators.
                continue
            found = f"raised {type(error).__name__}: {error}"
        checked += 1
        if found is not None and found != "no content stream took the native operators":
            failures.append(f"{name}: {found}")
    for label, content in EDGE_CASES.items():
        found = difference(one_page_pdf(content), 1)
        if found is not None:
            failures.append(f"{label}: {found}")
    rng = random.Random(0)
    for _ in range(count):
        content = random_content(rng)
        found = difference(one_page_pdf(content), 1)
        if found is not None and found != "no content stream took the native operators":
            failures.append(f"{content!r:.160}: {found}")
    every = (
        b"q 2 0 0 2 5 5 cm 0.5 G 0.5 g 1 0 0 RG 0 1 0 rg 0 0 0 1 K 0 0 0 1 k 2 w [1 2] 0 d "
        b"1 J 1 j 3 M BT /F1 9 Tf 12 TL 1 2 Td 1 -12 TD 1 0 0 1 5 5 Tm T* (a) Tj ET Q"
    )
    if handlers_called(every, native=True):
        failures.append("native operators still called their handlers")
    if handlers_called(every, native=False) != set(NATIVE_STATE_HANDLERS):
        failures.append("without native operators, not every handler was called")

    class Overriding(TextState):
        def op_w(self, operands: Any, depth: int) -> None:
            pass

    with PdfDocument(one_page_pdf(b"1 w")) as document:
        state = Overriding(document)
        operators = capture_recording.native_state_operators(state.operation_table(), state)
    if "w" in operators or "RG" not in operators:
        failures.append("an overridden handler did not keep its handler")
    print(
        f"{checked} files, {len(EDGE_CASES)} edge cases and {count} random contents:",
        f"{len(failures)} differ",
    )
    for failure in failures[:10]:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
