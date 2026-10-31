"""Resolve upstream package names to core_pdf_compat facades at import time.

Set CORE_PDF_COMPAT_ALIASES to comma-separated ``upstream=facade`` pairs, then load
this module as a pytest plugin. Upstream checkouts stay unmodified. A name with no
facade raises ModuleNotFoundError, so a gap fails instead of reaching a real library.
"""

import importlib
import importlib.util
import os
import sys
from importlib.abc import InspectLoader, Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import CodeType, ModuleType


def parse_aliases(value: str) -> dict[str, str]:
    pairs = (pair.split("=", 1) for pair in value.split(",") if pair)
    return {upstream: f"core_pdf_compat.{facade}" for upstream, facade in pairs}


class FacadeAliasFinder(MetaPathFinder, Loader):
    def __init__(self, aliases: dict[str, str]) -> None:
        self.aliases = aliases

    def resolve(self, name: str) -> str | None:
        for upstream, facade in self.aliases.items():
            if name == upstream:
                return facade
            if name.startswith(f"{upstream}."):
                return f"{facade}{name.removeprefix(upstream)}"
        return None

    def find_spec(self, fullname, path=None, target=None) -> ModuleSpec | None:
        facade = self.resolve(fullname)
        if facade is None:
            return None
        if any(upstream.startswith(f"{fullname}.") for upstream in self.aliases):
            return None
        try:
            importlib.import_module(facade)
        except ModuleNotFoundError as error:
            if error.name and error.name.startswith("core_pdf_compat"):
                raise ModuleNotFoundError(f"no facade for {fullname}", name=fullname) from None
            raise
        return ModuleSpec(fullname, self, origin=facade)

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        assert spec.origin is not None
        return importlib.import_module(spec.origin)

    def exec_module(self, module: ModuleType) -> None:
        return None

    def get_code(self, fullname: str) -> CodeType | None:
        # `python -m upstream.module` runs the facade module's own code.
        facade = self.resolve(fullname)
        assert facade is not None
        spec = importlib.util.find_spec(facade)
        assert spec is not None
        assert isinstance(spec.loader, InspectLoader)
        return spec.loader.get_code(facade)


sys.meta_path.insert(0, FacadeAliasFinder(parse_aliases(os.environ["CORE_PDF_COMPAT_ALIASES"])))
