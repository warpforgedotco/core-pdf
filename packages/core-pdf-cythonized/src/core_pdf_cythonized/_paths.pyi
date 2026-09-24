# SPDX-License-Identifier: AGPL-3.0-only

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy

Rectangle = tuple[float, float, float, float]

def flatten_path_commands(
    commands: Iterable[Any],
    matrix: Sequence[float] | None,
    hypot: Callable[[float, float], float],
) -> tuple[
    numpy.ndarray[Any, Any],
    numpy.ndarray[Any, Any],
    list[tuple[int, int, bool]],
    Rectangle | None,
    bool,
    numpy.ndarray[Any, Any],
]: ...
