# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import numpy

def outline_edges(
    xs: numpy.ndarray[Any, Any],
    ys: numpy.ndarray[Any, Any],
    spans: Any,
) -> tuple[numpy.ndarray[Any, Any] | None, list[tuple[int, int, bool]], bool]: ...
