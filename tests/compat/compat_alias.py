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


class UnsupportedBySuite(Exception):
    """A test reached an API listed as out of scope; compat_scope skips it."""


class StubMeta(type):
    def __call__(cls, *args: object, **kwargs: object) -> object:
        raise UnsupportedBySuite(cls.__name__)

    def __getattr__(cls, name: str) -> object:
        if name.startswith("__"):
            raise AttributeError(name)
        raise UnsupportedBySuite(f"{cls.__name__}.{name}")

    # Upstream test files index and compare these while collecting.
    def __getitem__(cls, key: object) -> object:
        return cls

    def __sub__(cls, other: object) -> object:
        return cls

    def __lt__(cls, other: object) -> bool:
        return False

    __gt__ = __le__ = __ge__ = __lt__


class StubModule(ModuleType):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.__path__ = []

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise UnsupportedBySuite(self.__name__)

    def __getattr__(self, name: str) -> object:
        if name.startswith("__"):
            raise AttributeError(name)
        return StubMeta(name, (), {})


class StubFinder(MetaPathFinder, Loader):
    def __init__(self, names: list[str]) -> None:
        self.names = names

    def find_spec(self, fullname, path=None, target=None) -> ModuleSpec | None:
        if any(fullname == name or fullname.startswith(f"{name}.") for name in self.names):
            return ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        return StubModule(spec.name)

    def exec_module(self, module: ModuleType) -> None:
        return None


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
_stubs = [name for name in os.environ.get("CORE_PDF_COMPAT_STUBS", "").split(",") if name]
if _stubs:
    sys.meta_path.insert(0, StubFinder(_stubs))
