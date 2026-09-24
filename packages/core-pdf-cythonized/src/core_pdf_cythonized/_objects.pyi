# SPDX-License-Identifier: AGPL-3.0-only

from collections.abc import Callable
from typing import Any

class ObjectScanner:
    def __init__(
        self,
        data: Any,
        whitespace_table: bytes,
        separator_table: bytes,
        name_escapes: bool,
        split_whitespace_compatible: bool,
        names: dict[bytes, Any],
        name_of: Callable[[bytes], Any],
        string_type: Callable[..., Any],
        reference_type: Callable[[int, int], Any],
    ) -> None: ...
    def release(self) -> None: ...
    def parse_dictionary(
        self,
        pos: int,
        decipher: Callable[[int, int, bytes, None], bytes | memoryview] | None = None,
        object_number: int = 0,
        generation: int = 0,
    ) -> tuple[dict[Any, Any], int] | None: ...
    def parse_array(
        self,
        pos: int,
        decipher: Callable[[int, int, bytes, None], bytes | memoryview] | None = None,
        object_number: int = 0,
        generation: int = 0,
    ) -> tuple[list[Any], int] | None: ...
