import copy
import inspect
import pickle
from typing import Any, ClassVar

import pytest

from core_records import (
    FrozenFields,
    GeneratedRecord,
    PickleFields,
    Record,
    RecordType,
    ReplaceFields,
    ReprFields,
    frozen_setattr,
)


class HandPoint(Record):
    __slots__ = ("x", "y", "label")
    __fields__: ClassVar[tuple[str, ...]] = ("x", "y", "label")
    __match_args__ = ("x", "y", "label")

    x: int
    y: int
    label: str

    def __init__(self, x: int, y: int, label: str = "") -> None:
        frozen_setattr(self, "x", x)
        frozen_setattr(self, "y", y)
        frozen_setattr(self, "label", label)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.label == other.label

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.label))


class Point(GeneratedRecord):
    x: int
    y: int
    label: str = ""
    scale: ClassVar[int] = 3


class Named(Point):
    name: str = "n"


class Quiet(Point):
    __repr_fields__: ClassVar[tuple[str, ...]] = ("label",)


class Mutable(ReprFields, ReplaceFields, metaclass=RecordType, frozen=False):
    value: int
    items: list[int]


class Unhashable(GeneratedRecord, hash=False):
    value: int


class Identity(GeneratedRecord, eq=False):
    value: int


class Custom(GeneratedRecord):
    value: int

    def __init__(self, value: str) -> None:
        frozen_setattr(self, "value", int(value))

    def __repr__(self) -> str:
        return f"Custom<{self.value}>"


class Empty(GeneratedRecord):
    value: object


def test_generated_record_is_a_record() -> None:
    assert Point.__mro__[1:6] == (GeneratedRecord, Record, FrozenFields, PickleFields, ReprFields)
    assert type(Point) is RecordType
    assert type(GeneratedRecord) is RecordType


def test_the_generated_surface_matches_the_hand_written_one() -> None:
    assert Point.__fields__ == HandPoint.__fields__
    assert Point.__match_args__ == HandPoint.__match_args__
    assert Point.__slots__ == HandPoint.__slots__
    generated = inspect.signature(Point.__init__)
    written = inspect.signature(HandPoint.__init__)
    assert [(p.name, p.kind, p.default) for p in generated.parameters.values()] == [
        (p.name, p.kind, p.default) for p in written.parameters.values()
    ]
    assert Point.__init__.__annotations__ == {
        "x": "int",
        "y": "int",
        "label": "str",
        "return": "None",
    }


def test_generated_methods_are_named_for_their_class() -> None:
    for name in ("__init__", "__eq__", "__hash__", "__repr__", "__replace__", "__getstate__"):
        method = Point.__dict__[name]
        assert method.__qualname__ == f"Point.{name}"
        assert method.__module__ == __name__


def test_class_variables_are_not_fields() -> None:
    assert "scale" not in Point.__fields__
    assert Point.scale == 3


def test_no_instance_dict() -> None:
    assert not hasattr(Point(1, 2), "__dict__")
    assert not hasattr(Named(1, 2), "__dict__")
    assert not hasattr(Mutable(1, []), "__dict__")


def test_init_positional_keyword_and_defaults() -> None:
    point = Point(1, y=2)
    assert (point.x, point.y, point.label) == (1, 2, "")
    with pytest.raises(TypeError):
        Point(1)  # ty: ignore[missing-argument]
    with pytest.raises(TypeError):
        Point(1, 2, "a", "b")  # ty: ignore[too-many-positional-arguments]


def test_equality_matches_the_hand_written_shape() -> None:
    assert Point(1, 2) == Point(1, 2)
    assert Point(1, 2) != Point(1, 3)
    point = Point(1, 2)
    assert point.__eq__(point) is True
    assert point.__eq__(HandPoint(1, 2)) is NotImplemented
    assert point.__eq__(Named(1, 2)) is NotImplemented
    assert Named(1, 2).__eq__(point) is NotImplemented
    assert point != Named(1, 2)
    assert point.__eq__(object()) is NotImplemented


def test_equality_short_circuits_in_field_order() -> None:
    seen: list[str] = []

    class Probe:
        def __init__(self, name: str, verdict: bool) -> None:
            self.name = name
            self.verdict = verdict

        def __eq__(self, other: object) -> bool:
            seen.append(self.name)
            return self.verdict

        __hash__ = None

    probes: list[Any] = [Probe("x", False), Probe("y", True), Probe("x2", False), Probe("y2", True)]
    left = Point(probes[0], probes[1])
    right = Point(probes[2], probes[3])
    assert (left == right) is False
    assert seen == ["x"]


def test_hash_matches_the_field_tuple() -> None:
    assert hash(Point(1, 2, "a")) == hash((1, 2, "a")) == hash(HandPoint(1, 2, "a"))
    assert hash(Named(1, 2)) == hash((1, 2, "", "n"))


def test_assignment_and_deletion_are_refused() -> None:
    point = Point(1, 2)
    with pytest.raises(AttributeError, match="cannot assign to field 'x'"):
        setattr(point, "x", 3)  # noqa: B010
    with pytest.raises(AttributeError, match="cannot delete field 'y'"):
        delattr(point, "y")  # noqa: B043


def test_repr_names_every_field_by_qualname() -> None:
    assert repr(Point(1, -2)) == repr(HandPoint(1, -2)).replace("HandPoint", "Point")
    assert repr(Point(1, -2)) == "Point(x=1, y=-2, label='')"
    assert repr(Named(1, 2)) == "Named(x=1, y=2, label='', name='n')"


def test_repr_fields_choose_what_repr_shows() -> None:
    assert repr(Quiet(1, 2, "p")) == "Quiet(label='p')"


def test_pickle_state_and_round_trip() -> None:
    assert Point(1, 2, "p").__getstate__() == HandPoint(1, 2, "p").__getstate__()
    for value in (Point(1, 2, "p"), Named(1, 2), Quiet(3, 4)):
        assert value.__getstate__() == [getattr(value, name) for name in value.__fields__]
        restored = pickle.loads(pickle.dumps(value))
        assert type(restored) is type(value)
        assert restored == value


def test_replace_passes_every_field_positionally() -> None:
    assert copy.replace(Point(1, 2), y=5) == Point(1, 5)
    replaced = copy.replace(Named(1, 2, "p"), name="q")
    assert replaced == Named(1, 2, "p", "q")
    assert copy.replace(Custom("4")) == Custom("4")


def test_replace_refuses_unknown_fields() -> None:
    with pytest.raises(TypeError, match=r"unexpected keyword arguments \['z'\]"):
        copy.replace(Point(1, 2), z=3)


def test_subclass_fields_extend_the_parent() -> None:
    assert Named.__fields__ == ("x", "y", "label", "name")
    assert Named.__slots__ == ("name",)
    assert Named.__match_args__ == ("x", "y", "label", "name")
    assert Named(1, 2).label == ""


def test_match_args_support_positional_patterns() -> None:
    match Point(1, 2, "p"):
        case Point(x, y, label):
            assert (x, y, label) == (1, 2, "p")
        case _:
            pytest.fail("no match")


def test_explicit_dunders_win() -> None:
    assert Custom("7").value == 7
    assert repr(Custom("7")) == "Custom<7>"
    assert Custom("7") == Custom("7")


def test_eq_false_keeps_identity_semantics() -> None:
    assert Identity(1) != Identity(1)
    assert "__eq__" not in Identity.__dict__
    assert Identity.__hash__ is object.__hash__


def test_hash_false_leaves_equal_records_unhashable() -> None:
    assert Unhashable(1) == Unhashable(1)
    assert Unhashable.__hash__ is None
    with pytest.raises(TypeError):
        hash(Unhashable(1))


def test_mutable_records_assign_and_are_unhashable() -> None:
    record = Mutable(1, [])
    record.value = 2
    assert record == Mutable(2, [])
    assert Mutable.__hash__ is None
    assert "__setattr__" not in Mutable.__dict__
    assert repr(record) == "Mutable(value=2, items=[])"
    assert copy.replace(record, value=3) == Mutable(3, [])
    with pytest.raises(TypeError):
        hash(record)


def test_a_mutable_record_cannot_extend_a_frozen_one() -> None:
    with pytest.raises(TypeError, match="cannot inherit a frozen"):

        class Thawed(Point, frozen=False):  # ty: ignore[invalid-frozen-dataclass-subclass]
            extra: int = 0


def test_a_required_field_cannot_follow_a_default() -> None:
    with pytest.raises(TypeError, match="non-default field 'z'"):

        class Broken(Point):
            z: int  # ty: ignore[dataclass-field-order]


def test_string_annotations_are_read_without_evaluation() -> None:
    namespace: dict[str, Any] = {"GeneratedRecord": GeneratedRecord}
    source = (
        "from __future__ import annotations\n"
        "from typing import ClassVar\n"
        "class Deferred(GeneratedRecord):\n"
        "    first: Undefined\n"
        "    second: ClassVar[Undefined]\n"
        "    third: int = 1\n"
    )
    exec(source, namespace)
    deferred = namespace["Deferred"]
    assert deferred.__fields__ == ("first", "third")
    assert deferred.__init__.__annotations__["first"] == "Undefined"


def test_every_record_gets_its_own_code_objects() -> None:
    for name in ("__init__", "__eq__", "__hash__", "__repr__", "__replace__", "__getstate__"):
        assert Point.__dict__[name].__code__ is not Named.__dict__[name].__code__


def test_generated_equality_and_hash_compile_to_the_hand_written_bytecode() -> None:
    for name in ("__eq__", "__hash__"):
        generated = Point.__dict__[name].__code__
        written = HandPoint.__dict__[name].__code__
        assert generated.co_code == written.co_code
        assert generated.co_names == written.co_names
        assert generated.co_varnames == written.co_varnames


def test_frozen_init_writes_through_slot_descriptors() -> None:
    init = Point.__dict__["__init__"]
    assert "__setattr__" not in init.__code__.co_names
    setters = [init.__globals__[name] for name in init.__code__.co_names]
    assert [setter.__self__ for setter in setters] == [
        Point.__dict__["x"],
        Point.__dict__["y"],
        Point.__dict__["label"],
    ]


def test_mutable_init_assigns_attributes() -> None:
    init = Mutable.__dict__["__init__"]
    assert init.__code__.co_names == ("value", "items")


def test_empty_bodies_generate_nothing() -> None:
    assert "__init__" not in GeneratedRecord.__dict__
    assert "__eq__" not in GeneratedRecord.__dict__
    assert GeneratedRecord.__slots__ == ()
    assert Empty(None).value is None


def test_a_written_field_list_must_agree_with_the_annotations() -> None:
    with pytest.raises(TypeError, match="disagrees"):

        class Reordered(GeneratedRecord):
            __fields__: ClassVar[tuple[str, ...]] = ("b", "a")
            a: int
            b: int


def test_init_false_keeps_the_inherited_initialiser() -> None:
    class Parsed(Custom, init=False):
        __repr_fields__: ClassVar[tuple[str, ...]] = ()

    assert "__init__" not in Parsed.__dict__
    assert Parsed("5").value == 5
    assert repr(Parsed("5")) == "Custom<5>"


def test_replace_stays_shared_when_the_initialiser_reorders_fields() -> None:
    class KeywordOnly(GeneratedRecord):
        first: int
        second: int

        def __init__(self, first: int, *, second: int) -> None:
            frozen_setattr(self, "first", first)
            frozen_setattr(self, "second", second)

    assert "__replace__" not in KeywordOnly.__dict__
    assert KeywordOnly.__replace__ is ReplaceFields.__replace__
