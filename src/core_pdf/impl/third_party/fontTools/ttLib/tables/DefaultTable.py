from __future__ import annotations

from typing import TYPE_CHECKING

from core_pdf.impl.third_party.fontTools.misc.textTools import Tag
from core_pdf.impl.third_party.fontTools.ttLib import getClassTag

if TYPE_CHECKING:
    from typing import Any

    from core_pdf.impl.third_party.fontTools.misc.xmlWriter import XMLWriter
    from core_pdf.impl.third_party.fontTools.ttLib import TTFont


class DefaultTable:
    dependencies: list[str] = []

    def __init__(self, tag: str | bytes | None = None) -> None:
        if tag is None:
            tag = getClassTag(self.__class__)
        self.tableTag = Tag(tag)

    def decompile(self, data: bytes, ttFont: TTFont) -> None:
        self.data = data

    def compile(self, ttFont: TTFont) -> bytes:
        return self.data

    def toXML(
        self, writer: XMLWriter, ttFont: TTFont, **kwargs: dict[str, Any]
    ) -> None:
        if hasattr(self, "ERROR"):
            writer.comment("An error occurred during the decompilation of this table")
            writer.newline()
            writer.comment(self.ERROR)
            writer.newline()
        writer.begintag("hexdata")
        writer.newline()
        writer.dumphex(self.compile(ttFont))
        writer.endtag("hexdata")
        writer.newline()

    def fromXML(
        self, name: str, attrs: dict[str, str], content: str, ttFont: TTFont
    ) -> None:
        from core_pdf.impl.third_party.fontTools import ttLib
        from core_pdf.impl.third_party.fontTools.misc.textTools import readHex

        if name != "hexdata":
            raise ttLib.TTLibError("can't handle '%s' element" % name)
        self.decompile(readHex(content), ttFont)

    def __repr__(self) -> str:
        return "<'%s' table at %x>" % (self.tableTag, id(self))

    def __eq__(self, other: Any) -> bool:
        if type(self) != type(other):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __ne__(self, other: Any) -> bool:
        result = self.__eq__(other)
        return result if result is NotImplemented else not result
