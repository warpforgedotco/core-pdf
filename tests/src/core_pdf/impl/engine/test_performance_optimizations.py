# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import math

from core_pdf.impl.engine.execution import RuntimeConfig, SharedMemoryPdfBuffer
from core_pdf.impl.engine.layout.models import TextRun
from core_pdf.impl.engine.layout.spatial import SpatialIndex
from core_pdf.impl.engine.layout.word_frequencies import word_rank
from core_pdf.impl.engine.parse.capture import internal_observations_from_runs
from core_pdf.impl.engine.spec.s_07_syntax.lexer_helpers import parse_float_token, parse_int_token
from core_pdf.impl.engine.spec.s_08_graphics.matrix import IDENTITY_MATRIX, Matrix
from core_pdf.impl.engine.spec.s_09_fonts.cmap_resources import resolve_cmap_decoder


def test_shared_memory_pdf_buffer() -> None:
    data = b"%PDF-1.7 sample data for shared memory"
    with SharedMemoryPdfBuffer(data) as shm_buf:
        buf = shm_buf.get_buffer()
        extracted = bytes(buf)
        del buf
        assert extracted == data
        assert shm_buf.name
        assert shm_buf.size == len(data)
    assert shm_buf.closed


def test_runtime_config_process_workers() -> None:
    cfg = RuntimeConfig(parent_workers=2, ocr_workers=2, process_workers=4)
    assert cfg.process_workers == 4


def test_internal_observations_from_runs_single_pass() -> None:
    runs = (
        TextRun(
            text="Hello",
            x0=10.0,
            y0=20.0,
            x1=50.0,
            y1=30.0,
            tx=10.0,
            ty=20.0,
            font_size=12.0,
            space_width=3.0,
            order=0,
            stream_order=0,
            xobject_depth=0,
            seqno=1,
            confidence=0.95,
        ),
        TextRun(
            text="World",
            x0=55.0,
            y0=20.0,
            x1=90.0,
            y1=30.0,
            tx=55.0,
            ty=20.0,
            font_size=12.0,
            space_width=3.0,
            order=1,
            stream_order=1,
            xobject_depth=0,
            seqno=2,
            confidence=None,
        ),
    )
    batch = internal_observations_from_runs(runs)
    assert len(batch) == 2
    assert batch.text == ("Hello", "World")
    assert batch.bbox.shape == (2, 4)
    assert batch.bbox[0, 0] == 10.0
    assert batch.bbox[1, 0] == 55.0
    assert batch.confidence[0] == 0.95
    assert math.isnan(batch.confidence[1])


def test_spatial_index_vectorized_intersections() -> None:
    boxes = [
        (0.0, 0.0, 10.0, 10.0),
        (20.0, 20.0, 30.0, 30.0),
        (5.0, 5.0, 15.0, 15.0),
    ]
    idx = SpatialIndex.from_boxes(boxes)
    hits = idx.intersecting_hits((0.0, 0.0, 8.0, 8.0))
    items = tuple(hit.item for hit in hits)
    assert 0 in items
    assert 1 not in items


def test_cmap_caching() -> None:
    decoder1 = resolve_cmap_decoder("Identity-H")
    decoder2 = resolve_cmap_decoder("Identity-H")
    assert decoder1 is decoder2


def test_word_rank_caching() -> None:
    rank1 = word_rank("The")
    rank2 = word_rank("the")
    assert rank1 == rank2
    assert rank1 is not None
    assert rank1 > 0


def test_lexer_number_parsing_memoryview() -> None:
    val_int = memoryview(b"12345")
    val_float = memoryview(b"123.45")
    assert parse_int_token(val_int) == 12345
    assert parse_float_token(val_float) == 123.45


def test_matrix_multiply_identity_short_circuit() -> None:
    m = Matrix(2.0, 0.0, 0.0, 2.0, 10.0, 10.0)
    assert IDENTITY_MATRIX.multiply(m) is m
    assert m.multiply(IDENTITY_MATRIX) is m
