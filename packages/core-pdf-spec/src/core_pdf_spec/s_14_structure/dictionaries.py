from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, ClassVar, NoReturn, Self, cast

from core_pdf_spec.s_07_syntax.stream import PdfStream
from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.types import MISSING

frozen_setattr = object.__setattr__


def parse_role_map(
    entries: Iterable[tuple[str, object]], resolve_name: Callable[[object], str | None]
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name, value in entries:
        mapped = resolve_name(value)
        if mapped is None:
            raise ValueError("invalid role map entry")
        mapping[name] = mapped
    return mapping


class StructureAttribute:
    __slots__ = ("value", "revision", "explicit_revision")

    value: PdfDict | PdfStream
    revision: int
    explicit_revision: bool

    __fields__: ClassVar[tuple[str, ...]] = ("value", "revision", "explicit_revision")
    __match_args__ = ("value", "revision", "explicit_revision")

    def __init__(self, value: PdfDict | PdfStream, revision: int, explicit_revision: bool) -> None:
        frozen_setattr(self, "value", value)
        frozen_setattr(self, "revision", revision)
        frozen_setattr(self, "explicit_revision", explicit_revision)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"value={self.value!r}, "
            f"revision={self.revision!r}, "
            f"explicit_revision={self.explicit_revision!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.value == other.value
            and self.revision == other.revision
            and self.explicit_revision == other.explicit_revision
        )

    def __hash__(self) -> int:
        return hash((self.value, self.revision, self.explicit_revision))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        value = changes.pop("value", self.value)
        revision = changes.pop("revision", self.revision)
        explicit_revision = changes.pop("explicit_revision", self.explicit_revision)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(value, revision, explicit_revision)


def attribute_entries(
    entries: Sequence[object],
    *,
    resolve: Callable[[object], object],
    resolve_revision: Callable[[object], int | None],
) -> Iterator[StructureAttribute]:
    index = 0
    pending: object = MISSING
    while index < len(entries):
        attribute = resolve(entries[index]) if pending is MISSING else pending
        pending = MISSING
        if not isinstance(attribute, (dict, PdfStream)):
            raise ValueError("invalid structure attribute entry")
        index += 1
        revision = 0
        explicit_revision = False
        if index < len(entries):
            candidate = resolve(entries[index])
            if isinstance(candidate, (dict, PdfStream)):
                pending = candidate
            else:
                parsed = resolve_revision(candidate)
                if parsed is None:
                    raise ValueError("invalid structure attribute revision")
                revision = parsed
                explicit_revision = True
                index += 1
        yield StructureAttribute(cast(PdfDict | PdfStream, attribute), revision, explicit_revision)


def marked_content_id(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("invalid structure content mcid")
    return value


__all__ = (
    "parse_role_map",
    "StructureAttribute",
    "attribute_entries",
    "marked_content_id",
)
