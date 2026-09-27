# SPDX-License-Identifier: AGPL-3.0-only


from __future__ import annotations

import importlib
import inspect
import pickle
import pkgutil
from copy import replace
from types import FunctionType, ModuleType
from typing import Any, ClassVar, Protocol

import pytest

from core_records import RecordType


class ValueClass(Protocol):
    __fields__: ClassVar[tuple[str, ...]]

    def __replace__(self, /, **changes: Any) -> ValueClass: ...


PACKAGE_ROOTS = (
    "core_pdf",
    "core_pdf_spec",
    "core_pdf_ocr",
    "core_pdf_compat",
    "core_pdf_validate",
    "core_postscript",
    "core_jbig2",
    "core_pdf_crypto",
    "core_adobe_fonts",
)

SCALARS: dict[str, object] = {
    "int": 1,
    "float": 1.0,
    "str": "",
    "bytes": b"",
    "bool": False,
    "object": None,
}

EMPTY_CONTAINERS: dict[str, object] = {
    "tuple": (),
    "Sequence": (),
    "Iterable": (),
    "frozenset": frozenset(),
    "list": [],
    "set": set(),
    "dict": {},
    "Mapping": {},
    "MutableMapping": {},
    "PdfDict": {},
}


def modules() -> list[ModuleType]:
    modules: list[ModuleType] = []
    for root in PACKAGE_ROOTS:
        try:
            package = importlib.import_module(root)
        except Exception:  # noqa: BLE001
            continue
        modules.append(package)
        for info in pkgutil.walk_packages(package.__path__, f"{root}."):
            if "._vendor" in info.name:
                continue
            try:
                modules.append(importlib.import_module(info.name))
            except Exception:  # noqa: BLE001
                continue
    return modules


def value_classes() -> dict[str, type[ValueClass]]:
    classes: dict[str, type[ValueClass]] = {}
    for module in modules():
        for value in vars(module).values():
            if not isinstance(value, type) or value.__module__ != module.__name__:
                continue
            if "__fields__" in value.__dict__:
                key = f"{value.__module__}.{value.__qualname__}"
                classes[key] = value
    return classes


def split_arguments(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current = ""
    for character in text:
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
        if character == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += character
    if current.strip():
        parts.append(current.strip())
    return parts


def synthesize(annotation: object) -> object:
    text = (annotation if isinstance(annotation, str) else str(annotation)).strip()
    if any(part.strip() == "None" for part in text.split("|")):
        return None
    if "ndarray" in text or "NDArray" in text:
        import numpy

        return numpy.zeros((2, 2, 1), dtype=numpy.uint8)
    base, _, arguments = text.partition("[")
    base = base.strip()
    if base in SCALARS:
        return SCALARS[base]
    if base == "tuple" and arguments:
        items = split_arguments(arguments.rstrip("]"))
        if items and items[-1] != "...":
            return tuple(synthesize(item) for item in items)
    return EMPTY_CONTAINERS.get(base)


def build(cls: type[ValueClass]) -> ValueClass:
    signature = inspect.signature(cls.__init__)
    annotations = cls.__init__.__annotations__
    args: list[object] = []
    kwargs: dict[str, object] = {}
    for name, parameter in list(signature.parameters.items())[1:]:
        if parameter.default is not inspect.Parameter.empty:
            continue
        value = synthesize(annotations.get(name, "object"))
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            kwargs[name] = value
        else:
            args.append(value)
    return cls(*args, **kwargs)


def buildable() -> dict[str, type[ValueClass]]:
    buildable: dict[str, type[ValueClass]] = {}
    for name, cls in value_classes().items():
        try:
            build(cls)
        except Exception:  # noqa: BLE001
            continue
        buildable[name] = cls
    return buildable


VALUE_CLASSES = value_classes()
BUILDABLE = buildable()
GENERATED = {name: cls for name, cls in VALUE_CLASSES.items() if isinstance(cls, RecordType)}


def is_frozen(cls: type[ValueClass]) -> bool:
    return any("__setattr__" in base.__dict__ for base in cls.__mro__[:-1])


def defines_eq(cls: type[ValueClass]) -> bool:
    return any("__eq__" in base.__dict__ for base in cls.__mro__[:-1])


def equal(left: object, right: object) -> bool | None:
    if left is right:
        return True
    try:
        verdict = left == right
    except ValueError:
        return None
    return verdict if isinstance(verdict, bool) else None


def test_value_classes_are_discovered() -> None:
    assert len(VALUE_CLASSES) > 150


def test_almost_every_value_class_is_constructible_from_placeholders() -> None:
    unbuildable = sorted(set(VALUE_CLASSES) - set(BUILDABLE))
    assert len(unbuildable) <= 8, unbuildable


@pytest.mark.parametrize("name", sorted(VALUE_CLASSES))
def test_fields_are_readable_and_match_slots(name: str) -> None:
    cls = VALUE_CLASSES[name]
    fields = cls.__fields__
    assert isinstance(fields, tuple)
    assert all(isinstance(field, str) for field in fields)
    assert len(set(fields)) == len(fields)
    own_slots = cls.__dict__.get("__slots__")
    if own_slots is not None:
        assert set(own_slots) <= set(fields)


@pytest.mark.parametrize("name", sorted(BUILDABLE))
def test_repr_names_the_class(name: str) -> None:
    cls = BUILDABLE[name]
    if not any("__repr__" in base.__dict__ for base in cls.__mro__[:-1]):
        return
    instance = build(cls)
    rendered = repr(instance)
    assert rendered.startswith(f"{type(instance).__qualname__}(")
    assert rendered.endswith(")")


@pytest.mark.parametrize("name", sorted(BUILDABLE))
def test_equality_compares_by_value_and_rejects_other_types(name: str) -> None:
    cls = BUILDABLE[name]
    instance = build(cls)
    assert instance == instance  # noqa: PLR0124
    if not defines_eq(cls):
        return
    twin = build(cls)
    fields_agree = all(
        equal(getattr(instance, field, None), getattr(twin, field, None)) is not False
        for field in cls.__fields__
    )
    if fields_agree:
        assert equal(instance, twin) is not False
    assert instance.__eq__(object()) is NotImplemented
    assert instance != object()


@pytest.mark.parametrize("name", sorted(BUILDABLE))
def test_hashability_follows_mutability(name: str) -> None:
    cls = BUILDABLE[name]
    instance = build(cls)
    if not defines_eq(cls):
        return
    if not is_frozen(cls):
        assert cls.__hash__ is None
        with pytest.raises(TypeError):
            hash(instance)
        return
    try:
        digest = hash(instance)
    except TypeError:
        return
    assert digest == hash(build(cls))


@pytest.mark.parametrize("name", sorted(BUILDABLE))
def test_frozen_classes_reject_assignment_and_deletion(name: str) -> None:
    cls = BUILDABLE[name]
    if not is_frozen(cls):
        return
    instance = build(cls)
    field = cls.__fields__[0]
    with pytest.raises(AttributeError, match="cannot assign"):
        setattr(instance, field, None)
    with pytest.raises(AttributeError, match="cannot delete"):
        delattr(instance, field)


@pytest.mark.parametrize("name", sorted(BUILDABLE))
def test_replace_round_trips_and_rejects_unknown_fields(name: str) -> None:
    cls = BUILDABLE[name]
    if "__replace__" not in cls.__dict__:
        return
    instance = build(cls)
    copied = replace(instance)
    assert type(copied) is cls
    for field in cls.__fields__:
        assert hasattr(copied, field) == hasattr(instance, field)
        if hasattr(instance, field):
            assert equal(getattr(copied, field), getattr(instance, field)) is not False
    with pytest.raises(TypeError, match="unexpected keyword"):
        replace(instance, definitely_not_a_field=None)


@pytest.mark.parametrize("name", sorted(BUILDABLE))
def test_frozen_slotted_classes_survive_pickling(name: str) -> None:
    cls = BUILDABLE[name]
    if not is_frozen(cls) or "__getstate__" not in dir(cls):
        return
    if any(base.__dict__.get("__slots__") is None for base in cls.__mro__[:-1]):
        return
    instance = build(cls)
    try:
        restored = pickle.loads(pickle.dumps(instance))
    except pickle.PicklingError, AttributeError, TypeError:
        return
    assert type(restored) is cls
    for field in cls.__fields__:
        assert hasattr(restored, field) == hasattr(instance, field)


def test_most_value_classes_are_generated() -> None:
    assert len(GENERATED) > 200


@pytest.mark.parametrize("dunder", ["__init__", "__eq__", "__hash__", "__repr__", "__replace__"])
def test_generated_methods_are_specialised_per_class(dunder: str) -> None:
    codes = [
        method.__code__
        for cls in GENERATED.values()
        if isinstance(method := cls.__dict__.get(dunder), FunctionType)
        and method.__code__.co_filename == "<record>"
    ]
    assert codes
    assert len({id(code) for code in codes}) == len(codes)


@pytest.mark.parametrize("name", sorted(GENERATED))
def test_generated_records_own_a_slot_for_every_new_field(name: str) -> None:
    cls = GENERATED[name]
    slots = {
        slot
        for base in cls.__mro__
        for slot in (
            (base.__dict__.get("__slots__", ()),)
            if isinstance(base.__dict__.get("__slots__", ()), str)
            else base.__dict__.get("__slots__", ())
        )
    }
    assert set(cls.__fields__) <= slots
