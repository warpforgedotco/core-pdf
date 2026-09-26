"""Each mixin does what the hand-written method it replaced did, from __fields__."""

import copy
import pickle
from typing import ClassVar

import pytest

from core_records import (
    FrozenFields,
    PickleFields,
    Record,
    ReplaceFields,
    ReprFields,
    frozen_setattr,
)


class Point(Record):
    __slots__ = ("x", "y")
    __fields__: ClassVar[tuple[str, ...]] = ("x", "y")

    x: int
    y: int

    def __init__(self, x: int, y: int) -> None:
        frozen_setattr(self, "x", x)
        frozen_setattr(self, "y", y)

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and (self.x, self.y) == (other.x, other.y)

    def __hash__(self) -> int:
        return hash((self.x, self.y))


class Labelled(Point):
    __slots__ = ("label",)
    __fields__: ClassVar[tuple[str, ...]] = ("x", "y", "label")
    __repr_fields__: ClassVar[tuple[str, ...]] = ("label",)

    label: str

    def __init__(self, x: int, y: int, label: str = "") -> None:
        super().__init__(x, y)
        frozen_setattr(self, "label", label)


def test_record_is_the_four_mixins_in_order() -> None:
    assert Record.__mro__[1:5] == (FrozenFields, PickleFields, ReprFields, ReplaceFields)


def test_the_mixins_add_no_instance_dict() -> None:
    assert not hasattr(Point(1, 2), "__dict__")


def test_assignment_and_deletion_are_refused() -> None:
    point = Point(1, 2)
    with pytest.raises(AttributeError, match="cannot assign to field 'x'"):
        setattr(point, "x", 3)  # noqa: B010 -- a literal assignment is a type error
    with pytest.raises(AttributeError, match="cannot delete field 'y'"):
        delattr(point, "y")  # noqa: B043 -- as is a literal deletion


def test_repr_names_every_field_by_qualname() -> None:
    assert repr(Point(1, -2)) == "Point(x=1, y=-2)"


def test_repr_fields_choose_what_repr_shows() -> None:
    assert repr(Labelled(1, 2, "p")) == "Labelled(label='p')"


def test_pickle_round_trips_through_a_frozen_class() -> None:
    for value in (Point(1, 2), Labelled(1, 2, "p")):
        restored = pickle.loads(pickle.dumps(value))
        assert type(restored) is type(value)
        assert [getattr(restored, name) for name in value.__fields__] == [
            getattr(value, name) for name in value.__fields__
        ]


def test_replace_passes_every_field_positionally() -> None:
    assert copy.replace(Point(1, 2), y=5) == Point(1, 5)
    replaced = copy.replace(Labelled(1, 2, "p"), label="q")
    assert (type(replaced), replaced.x, replaced.y, replaced.label) == (Labelled, 1, 2, "q")


def test_replace_refuses_unknown_fields() -> None:
    with pytest.raises(TypeError, match=r"unexpected keyword arguments \['z'\]"):
        copy.replace(Point(1, 2), z=3)
