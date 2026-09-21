# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any


def install_lazy_module_exports(
    module_globals: dict[str, Any],
    exports: Mapping[str, tuple[str, str]],
) -> None:
    module_name = module_globals.get("__name__", "?")

    def __getattr__(name: str) -> Any:
        try:
            import_path, attribute_name = exports[name]
        except KeyError:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}") from None
        return getattr(import_module(import_path), attribute_name)

    def __dir__() -> list[str]:
        return sorted((*module_globals, *exports))

    module_globals["__getattr__"] = __getattr__
    module_globals["__dir__"] = __dir__
