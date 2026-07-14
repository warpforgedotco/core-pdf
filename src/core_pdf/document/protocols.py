# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

from typing import Any, Protocol


class DocumentMixinProtocol(Protocol):
    resolver: Any
    acroform_cache: dict[str, Any] | None
    fields_cache: list[Any] | None
    oc_layers: dict[str, bool] | None
    named_destinations_cache: dict[str, Any] | None

    def catalog(self) -> dict[str, Any]: ...

    def page_index_for(self, page_obj: Any) -> int | None: ...
