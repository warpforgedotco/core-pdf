# SPDX-License-Identifier: AGPL-3.0-only
"""Affine sampling must bound source gathers by the output tile size."""

from typing import Any

import numpy
import pytest

from core_pdf.impl._impl.render import image_affine_target


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("components", [1, 3, 4])
@pytest.mark.parametrize("all_valid", [False, True])
def test_affine_sample_tiles_bound_allocations_and_preserve_pixels(
    monkeypatch: pytest.MonkeyPatch, transposed: bool, components: int, all_valid: bool
) -> None:
    budget = 4096
    allocations: list[int] = []

    class RecordingArray(numpy.ndarray):
        def __getitem__(self, key: Any) -> Any:
            result = super().__getitem__(key)
            if isinstance(result, numpy.ndarray) and not numpy.shares_memory(result, self):
                allocations.append(result.nbytes)
            return result

        def take(self, *args: Any, **kwargs: Any) -> Any:
            result = super().take(*args, **kwargs)
            allocations.append(result.nbytes)
            return result

    monkeypatch.setattr(image_affine_target, "AFFINE_BLIT_SCRATCH_BYTES", budget)
    source = numpy.arange(64 * 10000 * components, dtype=numpy.uint8).reshape(64, 10000, components)
    samples = source.view(RecordingArray)
    target = numpy.full((64, 16, 4), 7, dtype=numpy.uint8)
    source_y = numpy.arange(16) * 4 if transposed else numpy.arange(64)
    source_x = numpy.arange(64) * 149 if transposed else numpy.arange(16) * 613
    valid_rows = numpy.ones(64, dtype=numpy.bool_)
    valid_columns = numpy.ones(16, dtype=numpy.bool_)
    if not all_valid:
        valid_rows[::7] = False
        valid_columns[::5] = False

    sampler = image_affine_target.internal_ImageAffineTargetMixin()
    sampler.blit_opaque_sampled_tiles(
        samples,
        target,
        source_y,
        source_x,
        valid_rows,
        valid_columns,
        components,
        transposed=transposed,
    )

    assert allocations
    assert max(allocations) <= budget
    for row in range(64):
        for column in range(16):
            pixel = target[row, column]
            if not valid_rows[row] or not valid_columns[column]:
                assert numpy.all(pixel == 7)
                continue
            source_row = int(source_y[column if transposed else row])
            source_column = int(source_x[row if transposed else column])
            expected = source[source_row, source_column, : 1 if components == 1 else 3]
            assert numpy.all(pixel[:3] == expected)
            assert pixel[3] == 255
