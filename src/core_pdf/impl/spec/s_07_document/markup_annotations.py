"""ISO 32000-2 12.5.6.2: markup annotations' replies and pop-up windows."""

from __future__ import annotations

from collections.abc import Callable

from core_pdf.impl.spec.citations import cites
from core_pdf.impl.spec.s_07_syntax.types import PdfDict

# RT's values (Table 172); R is the default.
REPLY_TYPES = frozenset({"R", "Group"})


@cites(
    "ISO 32000-2",
    "12.5.6.2",
    introduced="1.5",
    rule="IRT names the annotation this one replies to (RT R) or is grouped with (RT Group)",
)
def reply_relation(
    annot: PdfDict,
    resolve: Callable[[object], object],
    resolve_name: Callable[[object], str | None],
) -> tuple[PdfDict, str] | None:
    """The annotation an annotation is in reply to (IRT), and how (RT, PDF 1.6): "R", a reply
    to it, or "Group", one annotation grouped with it, which a reader shows and manages as
    one with the annotation IRT names. RT is R when absent or not one of the two. None when
    IRT names no annotation dictionary."""
    target = resolve(annot.get("IRT"))
    if not isinstance(target, dict) or target is annot:
        return None
    reply_type = resolve_name(annot.get("RT"))
    return target, reply_type if reply_type in REPLY_TYPES else "R"


@cites(
    "ISO 32000-2",
    "12.5.6.14",
    introduced="1.3",
    rule="a markup annotation's Popup names its pop-up; the pop-up's Parent names it back",
)
def popup_link(
    annot: PdfDict, subtype: str | None, resolve: Callable[[object], object]
) -> PdfDict | None:
    """The annotation linked as a pop-up window: for a Popup annotation its Parent, the
    markup annotation whose text it shows; for any other its Popup entry (Table 172), the
    pop-up annotation for entering and editing its text. None when the entry names no
    annotation dictionary."""
    linked = resolve(annot.get("Parent" if subtype == "Popup" else "Popup"))
    if not isinstance(linked, dict) or linked is annot:
        return None
    return linked


__all__ = ("REPLY_TYPES", "popup_link", "reply_relation")
