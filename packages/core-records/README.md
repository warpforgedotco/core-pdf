# core-records

The boilerplate of a hand-written value class, once for the whole workspace:
`FrozenFields` (refuses assignment and deletion), `PickleFields` (pickle state
for a slotted class whose `__setattr__` refuses writes), `ReprFields`
(`Qualname(field=value, ...)`), `ReplaceFields` (`copy.replace`), and `Record`, all
four. A class declares `__fields__` and writes its own `__init__`, `__eq__` and
`__hash__`; the mixins read `__fields__`.

`GeneratedRecord` is a `Record` whose metaclass, `RecordType`, writes that
boilerplate from the class body instead. The annotated names, excluding
`ClassVar`, are the fields in order, and a value assigned to one in the body is its
default. For each class it builds `__slots__`, `__fields__`, `__match_args__`, an
`__init__` that stores through the slot descriptors (and then calls `__post_init__`
when the class has one), `__eq__` (identity shortcut,
exact class check, field-by-field `and`), `__hash__` over the field tuple, and its
own `__repr__`, `__replace__` and `__getstate__` in place of the shared mixin
methods. Every generated method is a separate code object per class, so the
interpreter specialises each one for its own receiver type, and it is built on
first use, so importing a module of records compiles nothing. A method written in the
class body wins. Class keywords select the variations: `frozen=False` for a mutable
record (no guards, unhashable when compared by value), `init=False` to inherit the
parent's `__init__`, `eq=False` to keep identity equality, and `hash=False` for an
equality-compared record that stays unhashable.
A subclass that declares no fields of its own is an ordinary subclass: it keeps
the `__init__`, equality, hash and mutability it inherits and only gets its own
`__repr__`, `__replace__` and `__getstate__`; give it `init=False` (and
`frozen=False` under a mutable record) so type checkers see the same.
`RecordType` is marked with `typing.dataclass_transform`, so type checkers see the
generated `__init__` and the frozen fields.

```python
class Point(GeneratedRecord):
    x: float
    y: float
    label: str = ""
```

This distribution is the lowest workspace floor package of `core-pdf`: it imports no
other workspace package, and every other one may import it.

Run its tests without the rest of the workspace installed:

```sh
uv sync --locked --package core-records --group test
uv run --locked --no-sync pytest packages/core-records/tests
```
