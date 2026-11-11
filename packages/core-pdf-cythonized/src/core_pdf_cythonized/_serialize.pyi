# SPDX-License-Identifier: AGPL-3.0-only

from collections.abc import Callable
from typing import Any

class ObjectWriter:
    def __init__(
        self,
        spaced: bool,
        name_type: type,
        string_type: type,
        reference_type: type,
        stream_type: type,
        fallback: Callable[[Any], bytes],
    ) -> None: ...
    def serialize(self, value: object) -> bytes: ...

def collect_references(value: object, reference_type: type, stream_type: type) -> list[Any]: ...
