# SPDX-License-Identifier: AGPL-3.0-only


from __future__ import annotations

import builtins
import re
from collections.abc import Callable
from functools import partial
from types import CodeType, FunctionType, MemberDescriptorType
from typing import Any, ClassVar, NoReturn, Self, dataclass_transform
from weakref import WeakSet

__all__ = (
    "FrozenFields",
    "GeneratedRecord",
    "PickleFields",
    "Record",
    "RecordType",
    "ReplaceFields",
    "ReprFields",
    "frozen_setattr",
)

frozen_setattr = object.__setattr__


class FrozenFields:
    __slots__ = ()

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")


class PickleFields:
    __slots__ = ()

    __fields__: ClassVar[tuple[str, ...]]

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)


class ReprFields:
    __slots__ = ()

    __fields__: ClassVar[tuple[str, ...]]
    __repr_fields__: ClassVar[tuple[str, ...] | None] = None

    def __repr__(self) -> str:
        names = self.__repr_fields__ if self.__repr_fields__ is not None else self.__fields__
        fields = ", ".join([f"{name}={getattr(self, name)!r}" for name in names])
        return f"{self.__class__.__qualname__}({fields})"


class ReplaceFields:
    __slots__ = ()

    __fields__: ClassVar[tuple[str, ...]]

    def __replace__(self, /, **changes: Any) -> Self:
        values = [changes.pop(name, getattr(self, name)) for name in self.__fields__]
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(*values)


class Record(FrozenFields, PickleFields, ReprFields, ReplaceFields):
    __slots__ = ()


MISSING: Any = object()
CLASS_VARIABLE = re.compile(r"^\s*(?:typing\.)?ClassVar\b")

type FieldSpecs = dict[str, tuple[Any, str]]

GENERATED: WeakSet[Callable[..., Any]] = WeakSet()
SHARED: tuple[Callable[..., Any], ...] = (
    ReprFields.__repr__,
    ReplaceFields.__replace__,
    PickleFields.__getstate__,
)


def own_annotations(namespace: dict[str, Any]) -> dict[str, str]:
    annotations = namespace.get("__annotations__")
    if annotations is None:
        import annotationlib

        annotate = annotationlib.get_annotate_from_class_namespace(namespace)
        if annotate is None:
            return {}
        annotations = annotationlib.call_annotate_function(annotate, annotationlib.Format.STRING)
    return {
        name: annotation
        for name, annotation in annotations.items()
        if not CLASS_VARIABLE.match(annotation)
    }


def inherited_specs(bases: tuple[type, ...]) -> FieldSpecs:
    specs: FieldSpecs = {}
    for base in reversed(bases):
        for ancestor in reversed(base.__mro__):
            specs.update(ancestor.__dict__.get("__field_specs__", {}))
    return specs


def slotted_names(bases: tuple[type, ...]) -> set[str]:
    names: set[str] = set()
    for base in bases:
        for ancestor in base.__mro__:
            slots = ancestor.__dict__.get("__slots__", ())
            names.update((slots,) if isinstance(slots, str) else slots)
    return names


def inherits_frozen_guard(bases: tuple[type, ...]) -> bool:
    return any(
        "__setattr__" in ancestor.__dict__
        for base in bases
        for ancestor in base.__mro__
        if ancestor is not object
    )


TOKEN = re.compile(r"_f(\d+)_")
BUILTINS: dict[str, Any] = {"__builtins__": builtins}
TEMPLATES: dict[tuple[str, int], Template] = {}


def init_template(names: list[str]) -> list[str]:
    body = [f"    __set_{index}(self, {name})" for index, name in enumerate(names)]
    return [f"def __init__({', '.join(['self', *names])}):", *(body or ["    pass"])]


def mutable_init_template(names: list[str]) -> list[str]:
    body = [f"    self.{name} = {name}" for name in names]
    return [f"def __init__({', '.join(['self', *names])}):", *(body or ["    pass"])]


def post_init_template(names: list[str]) -> list[str]:
    return [*init_template(names), "    self.__post_init__()"]


def mutable_post_init_template(names: list[str]) -> list[str]:
    return [*mutable_init_template(names), "    self.__post_init__()"]


def eq_template(names: list[str]) -> list[str]:
    compare = " and ".join(f"self.{name} == other.{name}" for name in names) or "True"
    return [
        "def __eq__(self, other):",
        "    if self is other:",
        "        return True",
        "    if other.__class__ is not self.__class__:",
        "        return NotImplemented",
        f"    return {compare}",
    ]


def hash_template(names: list[str]) -> list[str]:
    return [
        "def __hash__(self):",
        f"    return hash(({''.join(f'self.{name}, ' for name in names)}))",
    ]


def repr_template(names: list[str]) -> list[str]:
    parts = ", ".join(f"{name}={{self.{name}!r}}" for name in names)
    return ["def __repr__(self):", f'    return f"{{self.__class__.__qualname__}}({parts})"']


def replace_template(names: list[str]) -> list[str]:
    return [
        "def __replace__(self, /, **changes):",
        *(f"    {name} = changes.pop({name!r}, self.{name})" for name in names),
        "    if changes:",
        '        raise TypeError(f"__replace__() got unexpected keyword arguments'
        ' {sorted(changes)!r}")',
        f"    return self.__class__({', '.join(names)})",
    ]


def getstate_template(names: list[str]) -> list[str]:
    return [
        "def __getstate__(self):",
        f"    return [{', '.join(f'self.{name}' for name in names)}]",
    ]


SOURCES: dict[str, Callable[[list[str]], list[str]]] = {
    "init": init_template,
    "mutable_init": mutable_init_template,
    "post_init": post_init_template,
    "mutable_post_init": mutable_post_init_template,
    "__eq__": eq_template,
    "__hash__": hash_template,
    "__repr__": repr_template,
    "__replace__": replace_template,
    "__getstate__": getstate_template,
}


type Patches = tuple[tuple[int, int | str], ...]


class Template:
    __slots__ = ("code", "consts", "names", "varnames")

    def __init__(self, kind: str, count: int) -> None:
        source = "\n".join(SOURCES[kind]([f"_f{index}_" for index in range(count)]))
        (self.code,) = (
            item
            for item in compile(source, "<record>", "exec").co_consts
            if isinstance(item, CodeType)
        )
        self.names = token_patches(self.code.co_names)
        self.varnames = token_patches(self.code.co_varnames) if kind.endswith("init") else ()
        self.consts = token_patches(self.code.co_consts)

    def specialise(self, qualname: str, fields: tuple[str, ...]) -> CodeType:
        code = self.code
        return code.replace(
            co_names=patched(code.co_names, self.names, fields),
            co_varnames=patched(code.co_varnames, self.varnames, fields),
            co_consts=patched(code.co_consts, self.consts, fields),
            co_qualname=qualname,
        )


def token_patches(items: tuple[Any, ...]) -> Patches:
    patches: list[tuple[int, int | str]] = []
    for position, item in enumerate(items):
        if not isinstance(item, str) or len(parts := TOKEN.split(item)) == 1:
            continue
        if len(parts) == 3 and parts[0] == parts[2] == "":
            patches.append((position, int(parts[1])))
        else:
            patches.append(
                (
                    position,
                    "".join(
                        f"{{{part}}}" if odd % 2 else part.replace("{", "{{").replace("}", "}}")
                        for odd, part in enumerate(parts)
                    ),
                )
            )
    return tuple(patches)


def patched[T](items: tuple[T, ...], patches: Patches, fields: tuple[str, ...]) -> tuple[T, ...]:
    if not patches:
        return items
    result: list[Any] = list(items)
    for position, patch in patches:
        result[position] = fields[patch] if isinstance(patch, int) else patch.format(*fields)
    return tuple(result)


def specialise(
    cls: type,
    kind: str,
    fields: tuple[str, ...],
    scope: dict[str, Any] = BUILTINS,
    defaults: tuple[Any, ...] | None = None,
) -> FunctionType:
    template = TEMPLATES.get((kind, len(fields)))
    if template is None:
        template = TEMPLATES[kind, len(fields)] = Template(kind, len(fields))
    name = template.code.co_name
    code = template.specialise(f"{cls.__qualname__}.{name}", fields)
    function = FunctionType(code, scope, name, defaults)
    function.__module__ = cls.__module__
    GENERATED.add(function)
    return function


def slot_setter(cls: type, name: str) -> Callable[[object, object], None]:
    for ancestor in cls.__mro__:
        if name in ancestor.__dict__:
            descriptor = ancestor.__dict__[name]
            if type(descriptor) is MemberDescriptorType:
                setter: Callable[[object, object], None] = descriptor.__set__
                return setter
            break
    return partial(field_setter, name)


def field_setter(name: str, instance: object, value: object) -> None:
    frozen_setattr(instance, name, value)


def takes_fields_in_order(init: object, fields: tuple[str, ...]) -> bool:
    code = getattr(init, "__code__", None)
    return isinstance(code, CodeType) and code.co_varnames[1 : code.co_argcount] == fields


def specialises(cls: type, name: str, explicit: set[str]) -> bool:
    if name in explicit:
        return False
    inherited = getattr(cls, name, None)
    return inherited in GENERATED or inherited in SHARED


def build_init(cls: type, specs: FieldSpecs, frozen: bool) -> FunctionType:
    fields = tuple(specs)
    defaults: list[Any] = []
    for name, (default, _) in specs.items():
        if default is not MISSING:
            defaults.append(default)
        elif defaults:
            message = f"non-default field {name!r} follows a default field in {cls.__qualname__}"
            raise TypeError(message)
    kind = "post_init" if hasattr(cls, "__post_init__") else "init"
    if frozen:
        scope = dict(BUILTINS)
        for index, name in enumerate(fields):
            scope[f"__set_{index}"] = slot_setter(cls, name)
        function = specialise(cls, kind, fields, scope, tuple(defaults) or None)
    else:
        function = specialise(cls, f"mutable_{kind}", fields, BUILTINS, tuple(defaults) or None)
    function.__annotations__ = {name: annotation for name, (_, annotation) in specs.items()}
    function.__annotations__["return"] = "None"
    return function


@dataclass_transform(frozen_default=True, eq_default=True)
class RecordType(type):
    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        *,
        init: bool = True,
        frozen: bool = True,
        eq: bool = True,
        hash: bool | None = None,
        **kwargs: Any,
    ) -> RecordType:
        own = own_annotations(namespace)
        if not own and not any(isinstance(base, RecordType) for base in bases):
            return super().__new__(mcs, name, bases, namespace, **kwargs)
        specs = inherited_specs(bases)
        for field, annotation in own.items():
            inherited = specs.get(field, (MISSING, annotation))[0]
            specs[field] = (namespace.pop(field, inherited), annotation)
        fields = tuple(specs)
        if tuple(namespace.get("__fields__", fields)) != fields:
            raise TypeError(f"{name}.__fields__ disagrees with its annotated fields {fields}")
        guarded = inherits_frozen_guard(bases)
        if frozen and not guarded:
            namespace.setdefault("__setattr__", FrozenFields.__setattr__)
            namespace.setdefault("__delattr__", FrozenFields.__delattr__)
        elif not frozen and guarded:
            raise TypeError(f"mutable record {name} cannot inherit a frozen one")
        if "__slots__" not in namespace:
            owned = slotted_names(bases)
            namespace["__slots__"] = tuple(field for field in own if field not in owned)
        explicit = set(namespace)
        generate_eq = eq and "__eq__" not in explicit
        generate_hash = "__hash__" not in explicit and (
            bool(hash) or (hash is None and frozen and generate_eq)
        )
        if "__hash__" not in explicit and not generate_hash and (hash is False or generate_eq):
            namespace["__hash__"] = None
        namespace["__fields__"] = fields
        namespace["__field_specs__"] = specs
        namespace.setdefault("__match_args__", fields)
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        generated: dict[str, FunctionType] = {}
        if init and "__init__" not in explicit:
            generated["__init__"] = build_init(cls, specs, frozen)
        if generate_eq:
            generated["__eq__"] = specialise(cls, "__eq__", fields)
        if generate_hash:
            generated["__hash__"] = specialise(cls, "__hash__", fields)
        if specialises(cls, "__repr__", explicit):
            names = getattr(cls, "__repr_fields__", None)
            generated["__repr__"] = specialise(cls, "__repr__", fields if names is None else names)
        if specialises(cls, "__replace__", explicit) and takes_fields_in_order(
            generated.get("__init__", getattr(cls, "__init__", None)), fields
        ):
            generated["__replace__"] = specialise(cls, "__replace__", fields)
        if specialises(cls, "__getstate__", explicit):
            generated["__getstate__"] = specialise(cls, "__getstate__", fields)
        for attribute, function in generated.items():
            setattr(cls, attribute, function)
        return cls


class GeneratedRecord(Record, metaclass=RecordType):
    __slots__ = ()
