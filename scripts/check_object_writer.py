# SPDX-License-Identifier: AGPL-3.0-only

"""Check that core's compiled serializer writes exactly what the spec's serializer writes,
and that its compiled reference walk finds what the Python walk found.

core_pdf.impl.object_serialize writes PDF objects with core-pdf-cythonized's ObjectWriter,
which must give the bytes core_pdf_spec.s_07_syntax.serialize gives, in both layouts. This
serializes every object of the upstream test PDFs both ways, then deep and cyclic values,
then values generated around what the writer handles itself and what it hands back (names
with every byte, literal and hex strings with every byte, ints of every size, int and str
subclasses, reals, bytes, nested dictionaries, lists and tuples, other mappings and
sequences, streams, unwritable values), and fails on the first output or error that differs.
Every object and generated value is also walked for its references both ways, which must
give the same references in the same order, as a save writes objects in that order.

uv run --all-packages python scripts/check_object_writer.py [generated values]
"""

from __future__ import annotations

import enum
import glob
import random
import sys
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core_pdf import PdfDocument
from core_pdf.impl import object_serialize
from core_pdf.impl.document_editing import references_in
from core_pdf.impl.recovery_lexer import BareKeyword
from core_pdf_spec.s_07_syntax import serialize as spec
from core_pdf_spec.s_07_syntax.serialize import ObjectLayout
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.types import PdfName, PdfReference, PdfString
from scripts.check_strict_object_scanner import CORPUS


class Level(enum.IntEnum):
    LOW = 3


def python_references_in(value: Any) -> list[PdfReference]:
    """references_in as it was written in Python, the reference for the compiled walk."""
    found: list[PdfReference] = []
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, PdfReference):
            found.append(item)
        elif isinstance(item, PdfStream):
            pending.extend(item.dictionary.values())
        elif isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
    return found


def same_references(value: object) -> bool:
    """Whether the compiled walk finds the very references, in the order, Python does."""
    expected = python_references_in(value)
    actual = references_in(value)
    return len(expected) == len(actual) and all(
        left is right for left, right in zip(expected, actual, strict=True)
    )


def outcome(serialize: Any, value: object, layout: ObjectLayout) -> tuple[Any, ...]:
    try:
        return ("ok", serialize(value, layout))
    except Exception as error:  # noqa: BLE001
        return ("raised", type(error).__name__, str(error))


def generated(rng: random.Random, depth: int = 0) -> object:
    roll = rng.random()
    if depth > 3 or roll < 0.6:
        return rng.choice(
            (
                lambda: None,
                lambda: rng.random() < 0.5,
                lambda: rng.choice((0, -1, 7, 2**62, -(2**63), 2**64 + 5, -(10**30))),
                lambda: Level.LOW,
                lambda: rng.choice((0.0, -0.0, 1.5, -2.25, 1e-7, 123456789.125, 3.0)),
                lambda: PdfName.of(bytes(rng.randrange(256) for _ in range(rng.randint(0, 6)))),
                lambda: rng.choice(("Type", "A B", "é", BareKeyword("OBJR"), "")),
                lambda: PdfString(
                    bytes(rng.randrange(256) for _ in range(rng.randint(0, 12))),
                    is_literal=rng.choice((True, False, None)),
                ),
                lambda: bytes(rng.randrange(256) for _ in range(rng.randint(0, 5))),
                lambda: PdfReference(rng.randint(1, 10**6), rng.choice((0, 1, 65535))),
                lambda: float("nan"),
                lambda: object(),
                lambda: "Ā",
            )
        )()
    if roll < 0.75:
        items = [generated(rng, depth + 1) for _ in range(rng.randint(0, 4))]
        return tuple(items) if rng.random() < 0.3 else items
    if roll < 0.92:
        keys = [
            rng.choice((PdfName.of(b"K%d" % index), f"S{index}", b"B", PdfName.of(b"A#")))
            for index in range(rng.randint(0, 4))
        ]
        entries = {key: generated(rng, depth + 1) for key in keys}
        return OrderedDict(entries) if rng.random() < 0.2 else entries
    data = bytes(rng.randrange(256) for _ in range(rng.randint(0, 20)))
    return PdfStream({"Filter": PdfName.of(b"X"), "Length": 999}, data)


def main(argv: list[str]) -> int:
    count = int(argv[0]) if argv else 30000
    differences: list[str] = []
    objects = 0
    paths = sorted({path for pattern in CORPUS for path in glob.glob(pattern, recursive=True)})
    reference = PdfReference(5, 0)
    pairs = (
        (spec.serialize_object, object_serialize.serialize_object),
        (
            partial(spec.serialize_indirect_object, reference),
            partial(object_serialize.serialize_indirect_object, reference),
        ),
    )
    for path in paths:
        try:
            document = PdfDocument.open(path)
        except Exception:  # noqa: BLE001
            continue
        for key, entry in document.xref.items():
            if not entry.in_use:
                continue
            number = PdfReference(key >> 16, entry.generation)
            try:
                value = document.resolver.resolve(number)
            except Exception:  # noqa: BLE001
                continue
            objects += 1
            if not same_references(value):
                differences.append(f"{path} {number} references")
            differences.extend(
                f"{path} {number} {layout.name}"
                for layout in ObjectLayout
                if outcome(spec.serialize_object, value, layout)
                != outcome(object_serialize.serialize_object, value, layout)
            )
    # Containers nested past the writer's depth, deep enough to recurse past Python's limit,
    # and cyclic: the writer hands them to the spec's serializer, which raises.
    deep: object = 1
    for level in range(160):
        deep = [deep] if level % 2 else {"K": deep}
    cyclic: list[object] = []
    cyclic.append({"Self": cyclic})
    very_deep: object = None
    for _ in range(sys.getrecursionlimit() * 2):
        very_deep = [very_deep]
    for value in (deep, cyclic, very_deep):
        for layout in ObjectLayout:
            differences.extend(
                f"{layout.name} nested {type(value).__name__}"
                for expected, actual in pairs
                if outcome(expected, value, layout) != outcome(actual, value, layout)
            )
    rng = random.Random(0)
    for _ in range(count):
        value = generated(rng)
        if not same_references(value):
            differences.append(f"references {value!r:.160}")
        layout = rng.choice(tuple(ObjectLayout))
        differences.extend(
            f"{layout.name} {value!r:.160}"
            for expected, actual in pairs
            if outcome(expected, value, layout) != outcome(actual, value, layout)
        )
    summary = f"{objects} objects from {len(paths)} files, in both layouts, and {count} generated"
    print(f"{summary}: {len(differences)} differ")
    for difference in differences[:5]:
        print(difference)
    return 1 if differences else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
