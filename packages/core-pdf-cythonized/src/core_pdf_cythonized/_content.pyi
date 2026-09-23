# SPDX-License-Identifier: AGPL-3.0-only

from collections.abc import Set
from typing import Any

SCAN_OPERATOR: int
SCAN_NAME: int

class ContentScanner:
    pos: int
    def __init__(self, data: Any, keywords: Set[bytes]) -> None: ...
    def next_operation(self, operands: list[Any], names: dict[bytes, Any]) -> tuple[bytes, int]: ...
