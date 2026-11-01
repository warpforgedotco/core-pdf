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


COLLECTION = {"active": False}


class StubMeta(type):
    def __call__(cls, *args: object, **kwargs: object) -> object:
        if COLLECTION["active"]:
            return cls
        raise UnsupportedBySuite(cls.__name__)

    def __getattr__(cls, name: str) -> object:
        if name.startswith("__"):
            raise AttributeError(name)
        if COLLECTION["active"]:
            return cls
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
        if COLLECTION["active"]:
            return self
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


PARTIAL = {name for name in os.environ.get("CORE_PDF_COMPAT_PARTIAL", "").split(",") if name}
STUBS = [name for name in os.environ.get("CORE_PDF_COMPAT_STUBS", "").split(",") if name]


class FacadeAliasFinder(MetaPathFinder, Loader):
    def __init__(self, aliases: dict[str, str]) -> None:
        self.aliases = aliases
        self.facade_specs: dict[str, ModuleSpec | None] = {}

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
        module = importlib.import_module(spec.origin)
        self.facade_specs[spec.name] = module.__spec__
        if spec.name in PARTIAL:
            # Names the facade does not have yet become stand-ins, so importing them is fine
            # and a test is skipped only if it reaches one.
            vars(module)["__getattr__"] = StubModule(spec.name).__getattr__
        # A stubbed submodule can only be imported from a package.
        is_stub_parent = any(stub.startswith(f"{spec.name}.") for stub in STUBS)
        if (spec.name in PARTIAL or is_stub_parent) and not hasattr(module, "__path__"):
            module.__path__ = []
        return module

    def exec_module(self, module: ModuleType) -> None:
        # The import system gave the facade module the upstream name's spec. Put its own back,
        # or its relative imports warn that __package__ and __spec__.parent disagree.
        spec = module.__spec__
        assert spec is not None
        module.__spec__ = self.facade_specs[spec.name]

    def get_code(self, fullname: str) -> CodeType | None:
        # `python -m upstream.module` runs the facade module's own code.
        facade = self.resolve(fullname)
        assert facade is not None
        spec = importlib.util.find_spec(facade)
        assert spec is not None
        assert isinstance(spec.loader, InspectLoader)
        return spec.loader.get_code(facade)


sys.meta_path.insert(0, FacadeAliasFinder(parse_aliases(os.environ["CORE_PDF_COMPAT_ALIASES"])))
if STUBS:
    sys.meta_path.insert(0, StubFinder(STUBS))
