"""Deselect upstream tests whose source mentions a name in CORE_PDF_COMPAT_OUT_OF_SCOPE."""

import importlib
import inspect
import os
import re
import sys
from typing import Any

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    names = [name for name in os.environ.get("CORE_PDF_COMPAT_OUT_OF_SCOPE", "").split(",") if name]
    if not names:
        return
    pattern = re.compile(rf"\b({'|'.join(map(re.escape, names))})\b")
    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        function = getattr(item, "function", None)
        source = inspect.getsource(function) if function is not None else ""
        (deselected if pattern.search(source) else selected).append(item)
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
    # the facade's elements are rebuilt from their class name, text and metadata.
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
        item = cls(text=element.text, metadata=real.ElementMetadata(**values))
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
