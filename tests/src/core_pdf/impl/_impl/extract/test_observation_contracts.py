# SPDX-License-Identifier: AGPL-3.0-only
"""Selections must validate their contracts before taking no-op or empty paths."""

from dataclasses import replace
from typing import Any, cast

import numpy
import pytest

from core_pdf.impl._impl.extract.contracts import ObservationBatch


def observations() -> ObservationBatch:
    return ObservationBatch.from_columns(("A", "B"), ((0, 0, 1, 1), (2, 0, 3, 1)), source=0)


@pytest.mark.parametrize(
    "mask",
    [
        [True, True, False],
        [True, False, True],
        [False],
        [],
        [[True], [False]],
        [[False], [False]],
        True,
    ],
)
def test_mask_shape_is_checked_before_selection_fast_paths(mask: Any) -> None:
    batch = observations()
    array = numpy.asarray(mask, dtype=numpy.bool_)
    with pytest.raises(ValueError, match="mask.*shape"):
        batch.select(array)
    for primary in (batch, ObservationBatch.empty()):
        with pytest.raises(ValueError, match="mask.*shape"):
            ObservationBatch.concatenate_selected(primary, batch, array)


@pytest.mark.parametrize("mask", [[0, 0], [1, 1], [0.0, 1.0]])
def test_selection_rejects_non_boolean_masks(mask: Any) -> None:
    batch = observations()
    array = numpy.asarray(mask)
    with pytest.raises(TypeError, match="boolean"):
        batch.select(array)
    with pytest.raises(TypeError, match="boolean"):
        ObservationBatch.concatenate_selected(ObservationBatch.empty(), batch, array)


@pytest.mark.parametrize("indexes", [[0.1, 1.9], [0.0, 1.0], [False, True], ["0", "1"]])
def test_take_rejects_coerced_indexes(indexes: Any) -> None:
    with pytest.raises(TypeError, match="integers"):
        observations().take(indexes)


@pytest.mark.parametrize("indexes", [[[0], [1]], 0])
def test_take_rejects_non_vector_indexes(indexes: Any) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        observations().take(indexes)


@pytest.mark.parametrize("indexes", [[2], [-3], numpy.array([2**64 - 1], dtype=numpy.uint64)])
def test_take_checks_bounds_without_integer_overflow(indexes: Any) -> None:
    with pytest.raises(IndexError, match="out of range"):
        observations().take(indexes)


def test_valid_selections_preserve_order_references_and_no_op_identity() -> None:
    batch = observations()
    assert batch.select(numpy.array([True, True])) is batch
    assert batch.take([0, 1]) is batch
    assert batch.take([-1, 0, -1]).text == ("B", "A", "B")
    assert batch.select(numpy.array([False, True])).bbox.tolist() == [[2, 0, 3, 1]]
    assert len(batch.take([])) == len(batch.select(numpy.array([False, False]))) == 0
    assert not batch.bbox.flags.writeable
    assert numpy.isnan(batch.confidence).all()
    empty = ObservationBatch.empty()
    assert ObservationBatch.concatenate_selected(empty, batch, numpy.array([True, True])) is batch
    assert ObservationBatch.concatenate_selected(batch, empty, numpy.array([], dtype=bool)) is batch
    combined = ObservationBatch.concatenate_selected(batch, batch, numpy.array([False, True]))
    assert combined.text == ("A", "B", "B")
    assert combined.bbox.tolist() == [[0, 0, 1, 1], [2, 0, 3, 1], [2, 0, 3, 1]]


@pytest.mark.parametrize(
    "name",
    ["source", "confidence", "sequence", "visible", "rotation", "font_size", "line_break_before"],
)
def test_scalar_columns_require_vectors(name: str) -> None:
    batch = observations()
    column = getattr(batch, name).reshape((2, 1))
    with pytest.raises(ValueError, match="shape"):
        replace(batch, **{name: column})


@pytest.mark.parametrize("name", ["bbox", "polygon", "confidence", "source", "sequence", "visible"])
def test_direct_construction_checks_declared_column_dtypes(name: str) -> None:
    batch = observations()
    column = getattr(batch, name).astype(numpy.float64)
    with pytest.raises(TypeError, match="dtype"):
        replace(batch, **{name: column})
    assert column.flags.writeable


@pytest.mark.parametrize(
    "columns",
    [
        {"bbox": [(0, 0, 1, 1)]},
        {"polygon": [(0,) * 8]},
        {"confidence": [0.5]},
        {"sequence": [0]},
        {"visible": [True]},
        {"rotation": [0]},
        {"font_size": [12]},
        {"line_break_before": [False]},
        {"references": [None]},
        {"bbox": numpy.empty((0, 3))},
        {"bbox": numpy.empty((0, 5))},
        {"bbox": numpy.empty((0, 2, 2))},
        {"polygon": numpy.empty((0, 4))},
        {"visible": numpy.empty((0, 1), dtype=bool)},
        {"confidence": numpy.empty((0, 2))},
    ],
)
def test_empty_text_still_validates_supplied_columns(columns: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ObservationBatch.from_columns((), source=0, **{"bbox": (), **columns})


def test_column_factory_handles_empty_generators_and_checks_row_shape() -> None:
    assert len(ObservationBatch.from_columns(iter(()), iter(()), source=0)) == 0
    with pytest.raises(ValueError, match="bbox.*shape"):
        ObservationBatch.from_columns(("A", "B"), cast(Any, ((0, 0),) * 4), source=0)


def test_column_factory_does_not_freeze_or_alias_caller_arrays() -> None:
    boxes = numpy.array([[0, 0, 1, 1]], dtype=numpy.float32)
    confidence = numpy.array([0.5], dtype=numpy.float32)
    batch = ObservationBatch.from_columns(("A",), boxes, source=0, confidence=confidence)
    assert boxes.flags.writeable
    assert confidence.flags.writeable
    boxes[0, 0] = 99
    confidence[0] = 1.0
    assert batch.bbox.tolist() == [[0, 0, 1, 1]]
    assert batch.confidence.tolist() == [0.5]
