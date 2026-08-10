"""Unit tests for the shared compat kernel (``compat._common``)."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from core_pdf.api.v0.compat._common import (
    cluster_by,
    coerce_bbox,
    flip_box,
    open_source,
    synthesize_characters,
    write_bytes,
)
from core_pdf.api.v0.compat.state import OpenedState, StructuredState, SyntheticState
from core_pdf.impl.engine.structured import Document, Page

FIXTURE = Path("vendor/pdfminer.six/samples/simple1.pdf")
ENCRYPTED_FIXTURE = Path("vendor/pdfminer.six/samples/encryption/rc4-40.pdf")

pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="vendor fixtures not present")
requires_encrypted = pytest.mark.skipif(
    not ENCRYPTED_FIXTURE.exists(), reason="encrypted fixture not present"
)


def test_open_source_accepts_path_bytes_views_and_readers() -> None:
    raw = FIXTURE.read_bytes()
    for source in (FIXTURE, str(FIXTURE), raw, bytearray(raw), memoryview(raw), BytesIO(raw)):
        with open_source(source) as document:
            assert len(document.pages) == 1


@requires_encrypted
def test_open_source_forwards_password() -> None:
    with open_source(ENCRYPTED_FIXTURE, password="foo") as document:
        assert document.pages


@requires_encrypted
def test_pdfminer_facade_forwards_password() -> None:
    from core_pdf.api.v0.compat.pdfminer import extract_text

    assert extract_text(ENCRYPTED_FIXTURE, password="foo").strip()


def test_write_bytes_supports_paths_and_streams(tmp_path: Path) -> None:
    target = tmp_path / "data.bin"
    write_bytes(target, b"payload")
    assert target.read_bytes() == b"payload"
    stream = BytesIO()
    write_bytes(stream, b"payload")
    assert stream.getvalue() == b"payload"


def test_flip_box_and_coerce_bbox_round_trip() -> None:
    assert flip_box((10, 20, 30, 40), 100) == (10.0, 60.0, 30.0, 80.0)
    assert coerce_bbox([1, 2, 3, 4]) == (1.0, 2.0, 3.0, 4.0)
    with pytest.raises(ValueError, match="rectangle"):
        coerce_bbox("nope")


def test_synthesize_characters_divides_the_box_evenly() -> None:
    boxes = list(synthesize_characters("ab", (0.0, 0.0, 10.0, 5.0)))
    assert boxes == [("a", (0.0, 0.0, 5.0, 5.0)), ("b", (5.0, 0.0, 10.0, 5.0))]
    assert list(synthesize_characters("", (0.0, 0.0, 10.0, 5.0))) == []


def test_cluster_by_groups_numeric_keys_within_tolerance() -> None:
    groups = cluster_by([1.0, 1.5, 5.0, 5.2], lambda value: value, 1.0)
    assert groups == [[1.0, 1.5], [5.0, 5.2]]
    named = cluster_by([{"k": "a"}, {"k": "b"}, {"k": "a"}], "k")
    assert [[item["k"] for item in group] for group in named] == [["a", "a"], ["b"]]


def test_state_factories_return_explicit_variants() -> None:
    synthetic = StructuredState.synthetic(Document(pages=(Page(page_number=1),)))
    assert isinstance(synthetic, SyntheticState)
    assert synthetic.images == ()
    with pytest.raises(ValueError, match="source PDF"):
        _ = synthetic.source_pdf
    with StructuredState.open(FIXTURE) as opened:
        assert isinstance(opened, OpenedState)
        assert opened.source_pdf is opened.pdf


def test_directly_constructed_base_state_still_dispatches() -> None:
    state = StructuredState(None, Document(pages=(Page(page_number=1),)))
    assert state.images == ()
    with pytest.raises(ValueError, match="engine editor"):
        state.engine_edit()
    inserted = state.insert_page(Page(page_number=2))
    assert isinstance(inserted, SyntheticState)
    assert len(inserted.pages) == 2
