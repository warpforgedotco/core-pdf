from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, NoReturn, Protocol, Self

from core_pdf_spec.s_07_syntax.types import PdfDict
from core_pdf_spec.s_08_graphics.matrix import Matrix

frozen_setattr = object.__setattr__


class DecodedFontGlyph:
    __slots__ = ("code_bytes", "char_code", "cid", "gid", "unicode", "width_code")

    code_bytes: bytes
    char_code: int
    cid: int
    gid: int | None
    unicode: str
    width_code: int

    __fields__: ClassVar[tuple[str, ...]] = (
        "code_bytes",
        "char_code",
        "cid",
        "gid",
        "unicode",
        "width_code",
    )
    __match_args__ = ("code_bytes", "char_code", "cid", "gid", "unicode", "width_code")

    def __init__(
        self,
        code_bytes: bytes,
        char_code: int,
        cid: int,
        gid: int | None,
        unicode: str,
        width_code: int,
    ) -> None:
        frozen_setattr(self, "code_bytes", code_bytes)
        frozen_setattr(self, "char_code", char_code)
        frozen_setattr(self, "cid", cid)
        frozen_setattr(self, "gid", gid)
        frozen_setattr(self, "unicode", unicode)
        frozen_setattr(self, "width_code", width_code)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"code_bytes={self.code_bytes!r}, "
            f"char_code={self.char_code!r}, "
            f"cid={self.cid!r}, "
            f"gid={self.gid!r}, "
            f"unicode={self.unicode!r}, "
            f"width_code={self.width_code!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.code_bytes == other.code_bytes
            and self.char_code == other.char_code
            and self.cid == other.cid
            and self.gid == other.gid
            and self.unicode == other.unicode
            and self.width_code == other.width_code
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.code_bytes,
                self.char_code,
                self.cid,
                self.gid,
                self.unicode,
                self.width_code,
            )
        )

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
        code_bytes = changes.pop("code_bytes", self.code_bytes)
        char_code = changes.pop("char_code", self.char_code)
        cid = changes.pop("cid", self.cid)
        gid = changes.pop("gid", self.gid)
        unicode = changes.pop("unicode", self.unicode)
        width_code = changes.pop("width_code", self.width_code)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(code_bytes, char_code, cid, gid, unicode, width_code)


class FontService(Protocol):
    @property
    def is_type3(self) -> bool: ...
    @property
    def is_vertical(self) -> bool: ...
    @property
    def font(self) -> dict[str, Any]: ...

    @property
    def font_matrix(self) -> Matrix: ...

    def decode_glyphs(self, data: bytes) -> tuple[DecodedFontGlyph, ...]: ...
    def glyph_name(self, code: int) -> str: ...
    def glyph_advance_vector(
        self,
        code: int,
        *,
        font_size: float,
        char_space: float,
        word_space: float,
        horizontal_scale: float,
        encoded_space: bool,
    ) -> tuple[float, float]: ...
    def text_advance_vector(
        self,
        data: bytes | bytearray | memoryview,
        *,
        font_size: float,
        char_space: float,
        word_space: float,
        horizontal_scale: float,
        glyphs: tuple[DecodedFontGlyph, ...] | None = None,
    ) -> tuple[float, float]: ...


FontProvider = Callable[[PdfDict, PdfDict], FontService]


__all__ = ["DecodedFontGlyph", "FontService", "FontProvider"]
