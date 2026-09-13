# SPDX-License-Identifier: AGPL-3.0-only
"""Object streams obey their header identities and decoded byte ranges (§7.5.7)."""

from __future__ import annotations

import pytest

from core_pdf_spec.exceptions import PdfParseError
from core_pdf_spec.s_07_syntax.objects import PdfObjectStream
from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.types import PdfReference, PdfString


def object_stream(header: bytes, body: bytes, count: int) -> PdfStream:
    return PdfStream({"Type": "ObjStm", "N": count, "First": len(header)}, header + body)


def test_object_stream_uses_one_coherent_decoded_snapshot() -> None:
    calls: list[object] = []

    def decode(data: object, dictionary: object, *, parent_dictionary: object = None) -> bytes:
        calls.append(data)
        if len(calls) != 1:
            raise AssertionError("the constructor decoded the stream again")
        return b"4 0 2 3 [1](two)"

    stream = PdfStream({"Type": "ObjStm", "N": 2, "First": 8}, b"encoded", decoder=decode)
    objects = PdfObjectStream(stream)
    try:
        first = objects.get_at_index(0, expected_reference=PdfReference(4))
        assert first == [1]
        assert objects.get(4) is first
        second = objects.get_at_index(1, expected_reference=PdfReference(2))
        assert isinstance(second, PdfString)
        assert second.data == b"two"
        assert objects.object_bytes(4) == b"[1]"
        assert objects.object_bytes(2) == b"(two)"
        assert calls == [b"encoded"]
    finally:
        objects.close()
    assert first == [1]
    assert second.data == b"two"
    objects.close()
    assert objects.object_bytes(4) is None


@pytest.mark.parametrize(
    ("header", "body", "count"),
    [
        (b"1 1 ", b" 42", 1),
        (b"1 3 2 0 ", b"42 true", 2),
        (b"1 0 2 0 ", b"42", 2),
        (b"1 0 1 3 ", b"42 true", 2),
        (b"0 0 ", b"42", 1),
        (b"1 -1 ", b"42", 1),
        (b"1 0 2 2 ", b"42", 2),
    ],
)
def test_object_stream_rejects_invalid_header_ranges(
    header: bytes, body: bytes, count: int
) -> None:
    with pytest.raises(PdfParseError, match="object stream header"):
        PdfObjectStream(object_stream(header, body, count))


@pytest.mark.parametrize(
    ("header", "body", "count"),
    [
        (b"1 0 2 4 ", b"[42 ]", 2),
        (b"1 0 2 4 ", b"(abc)", 2),
        (b"1 0 2 8 ", b"<< /A 1 >>", 2),
        (b"1 0 ", b"42 99", 1),
        (b"1 0 ", b"2 0 R", 1),
        (b"1 0 ", b"<< /Length 1 >> stream\nx\nendstream", 1),
        (b"1 0 ", b"[<< /Length 1 >> stream\nx\nendstream]", 1),
        (b"1 0 ", b"<< /Child << /Length 1 >> stream\nx\nendstream >>", 1),
    ],
)
def test_compressed_object_cannot_consume_its_neighbor_or_extra_value(
    header: bytes, body: bytes, count: int
) -> None:
    objects = PdfObjectStream(object_stream(header, body, count))
    try:
        with pytest.raises(PdfParseError):
            objects.get(1)
        # A failed parse must not populate the object cache.
        with pytest.raises(PdfParseError):
            objects.get(1)
    finally:
        objects.close()


def test_adjacent_objects_and_references_inside_containers_remain_valid() -> None:
    objects = PdfObjectStream(object_stream(b"8 0 3 7 ", b"[9 0 R]<< /A 9 0 R >> %end", 2))
    try:
        assert objects.get(8) == [PdfReference(9)]
        assert objects.get(3) == {"A": PdfReference(9)}
    finally:
        objects.close()


@pytest.mark.parametrize(
    ("index", "reference"),
    [
        (-1, PdfReference(1)),
        (2, PdfReference(1)),
        (True, PdfReference(1)),
        (0, PdfReference(2)),
        (1, PdfReference(1)),
        (0, PdfReference(1, 1)),
    ],
)
def test_compressed_ordinal_requires_the_expected_header_identity(
    index: int, reference: PdfReference
) -> None:
    objects = PdfObjectStream(object_stream(b"1 0 2 2 ", b"1 2", 2))
    try:
        # Validate coordinates even when the demanded object was already cached.
        assert objects.get(1) == 1
        with pytest.raises(PdfParseError, match="compressed object reference"):
            objects.get_at_index(index, expected_reference=reference)
        missing = object()
        assert objects.get(PdfReference(1, 1), missing) is missing
        assert objects.object_bytes(PdfReference(1, 1)) is None
        assert objects.get(99, missing) is missing
    finally:
        objects.close()


def test_decoder_failures_propagate_without_retries() -> None:
    def decode(data: object, dictionary: object, *, parent_dictionary: object = None) -> bytes:
        raise RuntimeError("decoder failed")

    with pytest.raises(RuntimeError, match="decoder failed"):
        PdfObjectStream(PdfStream({"Type": "ObjStm", "N": 1, "First": 4}, decoder=decode))
