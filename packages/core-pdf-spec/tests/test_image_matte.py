# SPDX-License-Identifier: AGPL-3.0-only

import numpy
import pytest

from core_pdf_spec.s_11_transparency.images import unblend_matte_components


def test_matte_inverse_precedes_color_space_clipping() -> None:
    # ISO 32000-1/2 11.6.5.2 permits any colour when opacity is zero.
    samples = numpy.asarray([[0.5, 0.75], [0.25, 1.0], [0.0, 0.0]])
    alpha = numpy.asarray([0.5, 0.25, 0.0])
    actual = unblend_matte_components(samples, alpha, (0.5, 0.5))
    numpy.testing.assert_allclose(actual, [[0.5, 1.0], [-0.5, 2.5], [0.5, 0.5]])


@pytest.mark.parametrize("alpha", [[-0.1], [1.1], [float("nan")], [0.5, 1.0]])
def test_matte_rejects_invalid_opacity(alpha: list[float]) -> None:
    with pytest.raises(ValueError, match="matte"):
        unblend_matte_components(numpy.asarray([[0.5]]), numpy.asarray(alpha), (0.0,))


def test_matte_rejects_mismatched_color_components() -> None:
    with pytest.raises(ValueError, match="matte"):
        unblend_matte_components(numpy.asarray([[0.5, 0.5]]), numpy.asarray([0.5]), (0.0,))
