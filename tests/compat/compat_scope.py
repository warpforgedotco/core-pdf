"""Deselect the upstream tests that out_of_scope.toml rules out, one test case at a time.

CORE_PDF_COMPAT_OUT_OF_SCOPE lists names that drop a test mentioning them anywhere in its
source. CORE_PDF_COMPAT_STRATEGIES lists strategies that drop only the cases running them:
see uses_strategy. CORE_PDF_COMPAT_STRATEGY_AGNOSTIC lists the tests exempt from that.
"""

import ast
import importlib
import inspect
import os
import re
import sys
import textwrap
from typing import Any

import pytest


def env_list(name: str) -> list[str]:
    return [value for value in os.environ.get(name, "").split(",") if value]


def uses_strategy(item: pytest.Item, strategies: set[str]) -> bool:
    # A case uses a strategy when one is among its parameter values, or when the code in
    # the test's body names it. Decorators are left out, so the other cases of a
    # parametrized test stay in, and so are comments and any prose in docstrings.
    if not strategies:
        return False
    spec = getattr(item, "callspec", None)
    if spec is not None and any(
        isinstance(value, str) and value in strategies for value in spec.params.values()
    ):
        return True
    function = getattr(item, "function", None)
    if function is None:
        return False
    definition = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
    if not isinstance(definition, ast.FunctionDef):
        return False
    for statement in definition.body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Attribute) and node.attr.lower() in strategies:
                return True
            if isinstance(node, ast.Constant) and node.value in strategies:
                return True
    return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    names = env_list("CORE_PDF_COMPAT_OUT_OF_SCOPE")
    strategies = set(env_list("CORE_PDF_COMPAT_STRATEGIES"))
    agnostic = set(env_list("CORE_PDF_COMPAT_STRATEGY_AGNOSTIC"))
    if not names and not strategies:
        return
    pattern = re.compile(rf"\b({'|'.join(map(re.escape, names))})\b") if names else None
    keep = set(env_list("CORE_PDF_COMPAT_KEEP"))
    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        function = getattr(item, "function", None)
        source = inspect.getsource(function) if function is not None else ""
        kept = getattr(item, "originalname", item.name) in keep
        mentioned = pattern is not None and pattern.search(source) is not None and not kept
        exempt = function is not None and function.__name__ in agnostic
        dropped = mentioned or (not exempt and uses_strategy(item, strategies))
        (deselected if dropped else selected).append(item)
    config.hook.pytest_deselected(items=deselected)
    items[:] = selected


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    report = yield
    error = call.excinfo.value if call.excinfo is not None else None
    name = type(error).__name__
    if name == "UnsupportedBySuite" or (
        isinstance(error, NotImplementedError) and "no PDF writer" in str(error)
    ):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0, f"out of scope: {error}")
    return report


def contains_stub(value: object) -> bool:
    if isinstance(value, (list, tuple, set)):
        return any(contains_stub(item) for item in value)
    if isinstance(value, dict):
        return any(contains_stub(item) for item in value.values())
    return type(value).__name__ == "StubMeta"


@pytest.hookimpl(tryfirst=True)
def pytest_collectstart() -> None:
    set_collecting(active=True)


def pytest_collection_finish() -> None:
    set_collecting(active=False)


def set_collecting(*, active: bool) -> None:
    module = sys.modules.get("compat_alias")
    if module is not None:
        module.COLLECTION["active"] = active


def pytest_runtest_setup(item: pytest.Item) -> None:
    set_collecting(active=False)
    spec = getattr(item, "callspec", None)
    if spec is not None and contains_stub(spec.params):
        pytest.skip("out of scope: parametrized with an unimplemented API")


def real_unstructured_elements(elements: list[Any]) -> list[object]:
    # Upstream tests check real element classes and feed elements to real chunking, so
    # the facade's elements are rebuilt from their class name, id, text and metadata.
    # Imported by name, because only the suite's own environment has unstructured.
    real: Any = importlib.import_module("unstructured.documents.elements")
    coordinates_module: Any = importlib.import_module("unstructured.documents.coordinates")

    fields = set(real.ElementMetadata.__annotations__)
    rebuilt: list[object] = []
    for element in elements:
        values = {key: value for key, value in element.metadata.items() if key in fields}
        coordinates = values.pop("coordinates", None)
        if coordinates is not None:
            system = coordinates_module.PixelSpace(
                width=coordinates.width, height=coordinates.height
            )
            values["coordinates"] = real.CoordinatesMetadata(coordinates.points, system)
        cls = getattr(real, type(element).__name__, real.Text)
        item = cls(
            text=element.text,
            element_id=element.id or None,
            metadata=real.ElementMetadata(**values),
        )
        rebuilt.append(item)
    return rebuilt


def pytest_configure() -> None:
    if "unstructured.partition.pdf" not in os.environ.get("CORE_PDF_COMPAT_ALIASES", ""):
        return
    import core_pdf_compat.unstructured as facade  # noqa: PLC0415

    module: Any = facade
    partition = module.partition_pdf

    def partition_pdf(*args: Any, **kwargs: Any) -> list[object]:
        return real_unstructured_elements(partition(*args, **kwargs))

    module.partition_pdf = partition_pdf
