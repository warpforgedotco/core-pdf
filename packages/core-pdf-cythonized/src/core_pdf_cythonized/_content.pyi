# SPDX-License-Identifier: AGPL-3.0-only

from collections.abc import Callable, Set
from typing import Any

class ContentScanner:
    pos: int
    @property
    def operands(self) -> list[Any]: ...
    def __init__(
        self,
        data: Any,
        keywords: Set[bytes],
        object_keywords: Set[str],
        make_name: Callable[[bytes], Any],
    ) -> None: ...
    def set_path_state(self, state: Any) -> None: ...
    def next_operation(self) -> tuple[str, tuple[Any, ...]] | int: ...
