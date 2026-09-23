# SPDX-License-Identifier: AGPL-3.0-only

from typing import Any

import numpy

def composite_elementary_normal(
    destination: numpy.ndarray[Any, Any],
    rendered: numpy.ndarray[Any, Any],
    source_alpha: numpy.ndarray[Any, Any],
) -> numpy.ndarray[Any, Any]: ...
