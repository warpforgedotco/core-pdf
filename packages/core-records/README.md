# core-records

The boilerplate of a hand-written value class, once for the whole workspace:
`FrozenFields` (refuses assignment and deletion), `PickleFields` (pickle state
for a slotted class whose `__setattr__` refuses writes), `ReprFields`
(`Qualname(field=value, ...)`), `ReplaceFields` (`copy.replace`), and `Record`, all
four. A class declares `__fields__` and writes its own `__init__`, `__eq__` and
`__hash__`; the mixins read `__fields__`.

This distribution is the lowest workspace floor package of `core-pdf`: it imports no
other workspace package, and every other one may import it.

Run its tests without the rest of the workspace installed:

```sh
uv sync --locked --package core-records --group test
uv run --locked --no-sync pytest packages/core-records/tests
```
