# Ported from pypdf, whose notices follow. ViewerPreferences is Copyright (c) 2023, Pubpub-ZZ.
#
# Copyright (c) 2006, Mathieu Fenniak
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# * The name of the author may not be used to endorse or promote products
# derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""pypdf's structural ``generic`` types, built on the object types in ``generic``.

``generic`` imports this module on its last line, so everything here sees a complete
``generic``. None of these types write bytes: anything with its own written form gives it
to ``generic.to_core`` through ``core_form``, and core_pdf_spec's serializer does the rest.
"""

from __future__ import annotations

import bisect
import logging
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime
from functools import cached_property
from typing import Any, cast

from core_pdf_compat.pypdf.constants import (
    CheckboxRadioButtonAttributes,
    FieldDictionaryAttributes,
    OutlineFontFlag,
    TypFitArguments,
)
from core_pdf_compat.pypdf.constants import (
    TypArguments as TA,  # noqa: N817
)
from core_pdf_compat.pypdf.errors import (
    STREAM_TRUNCATED_PREMATURELY,
    PdfReadError,
    PdfStreamError,
    PyPdfError,
)
from core_pdf_compat.pypdf.generic import (
    ArrayObject,
    BooleanObject,
    ByteStringObject,
    DecodedStreamObject,
    DictionaryObject,
    EncodedStreamObject,
    FloatObject,
    NameObject,
    NullObject,
    NumberObject,
    StreamObject,
    TextStringObject,
    create_string_object,
    is_null_or_none,
)
from core_pdf_spec.s_07_syntax_primitives.text_string import PDFDOC_ENCODING_TABLE

logger = logging.getLogger(__name__)

# Bytes pypdf treats as undefined in PDFDocEncoding, which core-pdf's table passes through.
PDFDOC_UNDEFINED = frozenset({0x00, 0x16, 0x7F, 0x9F, 0xAD})
PDFDOC_REVERSE = {
    character: code
    for code, character in enumerate(PDFDOC_ENCODING_TABLE)
    if code not in PDFDOC_UNDEFINED
}
HEX_DIGITS = b"0123456789abcdefABCDEF"
WHITESPACE = b" \n\r\t\x00\x0c"
STRING_ESCAPES = {
    b"n": ord(b"\n"),
    b"r": ord(b"\r"),
    b"t": ord(b"\t"),
    b"b": ord(b"\b"),
    b"f": ord(b"\f"),
    **{symbol: ord(symbol) for symbol in (bytes((c,)) for c in b"()/\\ %<>[]#_&$")},
}
BACKSLASH = 0x5C


def encode_pdfdocencoding(unicode_string: str) -> bytes:
    try:
        return bytes(PDFDOC_REVERSE[character] for character in unicode_string)
    except KeyError:
        raise UnicodeEncodeError(
            "pdfdocencoding", unicode_string, -1, -1, "does not exist in translation table"
        ) from None


def decode_pdfdocencoding(byte_array: bytes) -> str:
    for byte in byte_array:
        if byte in PDFDOC_UNDEFINED:
            raise UnicodeDecodeError(
                "pdfdocencoding",
                bytes((byte,)),
                -1,
                -1,
                "does not exist in translation table",
            )
    return "".join(PDFDOC_ENCODING_TABLE[byte] for byte in byte_array)


def hex_to_rgb(value: str) -> tuple[float, float, float]:
    digits = value.lstrip("#")
    return (
        int(digits[0:2], 16) / 255.0,
        int(digits[2:4], 16) / 255.0,
        int(digits[4:6], 16) / 255.0,
    )


def _next_token(stream: Any) -> bytes:
    token = stream.read(1)
    while token and token in WHITESPACE:
        token = stream.read(1)
    return token


def read_hex_string_from_stream(
    stream: Any, forced_encoding: str | list[str] | dict[int, str] | None = None
) -> TextStringObject | ByteStringObject:
    """Read a hex string, assuming the stream is on its opening delimiter."""
    stream.read(1)
    output = bytearray()
    pending = b""
    while True:
        token = _next_token(stream)
        if not token:
            raise PdfStreamError(STREAM_TRUNCATED_PREMATURELY)
        if token == b">":
            break
        if token not in HEX_DIGITS:
            raise PdfStreamError(f"Invalid hexadecimal character {token!r} in hex string")
        pending += token
        if len(pending) == 2:
            output.append(int(pending, 16))
            pending = b""
    if pending:
        output.append(int(pending + b"0", 16))
    return create_string_object(bytes(output), forced_encoding)


def _read_octal_escape(stream: Any, first: bytes) -> int:
    start = stream.tell() - 1
    digits = first
    for _ in range(2):
        following = stream.read(1)
        if not b"0" <= following <= b"7":
            if following:
                stream.seek(-1, 1)
            break
        digits += following
    code = int(digits, 8)
    if code > 255:
        stream.seek(start)
        return BACKSLASH
    return code


def _read_escape(stream: Any, output: bytearray) -> None:
    """Handle the bytes after a backslash in a literal string."""
    token = stream.read(1)
    if not token:
        raise PdfStreamError(STREAM_TRUNCATED_PREMATURELY)
    if token in STRING_ESCAPES:
        output.append(STRING_ESCAPES[token])
    elif b"0" <= token <= b"7":
        output.append(_read_octal_escape(stream, token))
    elif token in b"\n\r":
        # An escaped line break adds nothing. A two-byte break is one line break.
        following = stream.read(1)
        if following and following not in b"\n\r":
            stream.seek(-1, 1)
    else:
        logger.warning("Unexpected escaped string: %s", token.decode("utf-8", "ignore"))
        output += b"\\" + token


def read_string_from_stream(
    stream: Any, forced_encoding: str | list[str] | dict[int, str] | None = None
) -> TextStringObject | ByteStringObject:
    """Read a literal string, assuming the stream is on its opening delimiter."""
    stream.read(1)
    depth = 1
    output = bytearray()
    while True:
        token = stream.read(1)
        if not token:
            raise PdfStreamError(STREAM_TRUNCATED_PREMATURELY)
        if token == b"\\":
            _read_escape(stream, output)
            continue
        if token == b"(":
            depth += 1
        elif token == b")":
            depth -= 1
            if depth == 0:
                break
        output.append(token[0])
    return create_string_object(bytes(output), forced_encoding)


class Fit:
    """How a destination shows its page (ISO 32000-2 12.3.2.2)."""

    def __init__(self, fit_type: str, fit_args: Iterable[Any] = ()) -> None:
        self.fit_type = NameObject(fit_type)
        self.fit_args: list[Any] = [
            NullObject() if is_null_or_none(arg) else FloatObject(arg) for arg in fit_args
        ]

    @classmethod
    def xyz(
        cls, left: float | None = None, top: float | None = None, zoom: float | None = None
    ) -> Fit:
        return cls("/XYZ", (left, top, zoom))

    @classmethod
    def fit(cls) -> Fit:
        return cls("/Fit")

    @classmethod
    def fit_horizontally(cls, top: float | None = None) -> Fit:
        return cls("/FitH", (top,))

    @classmethod
    def fit_vertically(cls, left: float | None = None) -> Fit:
        return cls("/FitV", (left,))

    @classmethod
    def fit_rectangle(
        cls,
        left: float | None = None,
        bottom: float | None = None,
        right: float | None = None,
        top: float | None = None,
    ) -> Fit:
        return cls("/FitR", (left, bottom, right, top))

    @classmethod
    def fit_box(cls) -> Fit:
        return cls("/FitB")

    @classmethod
    def fit_box_horizontally(cls, top: float | None = None) -> Fit:
        return cls("/FitBH", (top,))

    @classmethod
    def fit_box_vertically(cls, left: float | None = None) -> Fit:
        return cls("/FitBV", (left,))

    def __str__(self) -> str:
        if not self.fit_args:
            return f"Fit({self.fit_type})"
        return f"Fit({self.fit_type}, {self.fit_args})"


PAGE_FIT = Fit.fit()


def _raw(node: Any, key: str, default: Any = None) -> Any:
    """A dictionary entry as stored, so a reference stays a reference."""
    return dict.get(node, key, default)


def _reset_tree_links(child: Any) -> None:
    for key in ("/Parent", "/Next", "/Prev"):
        child.pop(key, None)


class TreeObject(DictionaryObject):
    """A node of a doubly linked tree, as outlines are (ISO 32000-2 12.3.3)."""

    def __init__(self, dct: DictionaryObject | None = None) -> None:
        DictionaryObject.__init__(self)
        if dct:
            for key in dict.keys(dct):
                dict.__setitem__(self, key, dict.__getitem__(dct, key))

    def has_children(self) -> bool:
        return "/First" in self

    def __iter__(self) -> Any:
        return self.children()

    def children(self) -> Iterator[Any]:
        if not self.has_children():
            return
        last = self["/Last"]
        child = self["/First"].get_object()
        visited: set[int] = set()
        while True:
            if id(child) in visited:
                logger.warning("Detected cycle in outline structure for %s", child)
                return
            visited.add(id(child))
            yield child
            following = _raw(child, "/Next")
            if child == last or is_null_or_none(following):
                return
            child = following.get_object()

    def add_child(self, child: Any, pdf: Any) -> None:
        self.insert_child(child, None, pdf)

    def inc_parent_counter_default(self, parent: Any, n: int) -> None:
        visited: set[int] = set()
        while not is_null_or_none(parent):
            node = parent.get_object()
            if "/Count" not in node or id(node) in visited:
                return
            visited.add(id(node))
            node[NameObject("/Count")] = NumberObject(max(0, node["/Count"] + n))
            parent = _raw(node, "/Parent")

    def inc_parent_counter_outline(self, parent: Any, n: int) -> None:
        visited: set[int] = set()
        while not is_null_or_none(parent):
            node = parent.get_object()
            if id(node) in visited:
                return
            visited.add(id(node))
            is_open = node.get_resolved("/%is_open%", True) == True  # noqa: E712
            count = abs(node.get_resolved("/Count", 0)) + n
            node[NameObject("/Count")] = NumberObject(count if is_open else -count)
            if not is_open:
                return
            parent = _raw(node, "/Parent")

    def insert_child(
        self,
        child: Any,
        before: Any,
        pdf: Any,
        inc_parent_counter: Callable[..., Any] | None = None,
    ) -> Any:
        del pdf
        increment = inc_parent_counter or self.inc_parent_counter_default
        child_obj = child.get_object()
        reference = child.indirect_reference
        child_obj[NameObject("/Parent")] = self.indirect_reference
        if "/First" not in self:
            self[NameObject("/First")] = reference
            self[NameObject("/Count")] = NumberObject(0)
            self[NameObject("/Last")] = reference
            child_obj.pop("/Next", None)
            child_obj.pop("/Prev", None)
        else:
            self._link_child(child_obj, reference, before)
        increment(self, child_obj.get_resolved("/Count", 1))
        return reference

    def _link_child(self, child_obj: Any, reference: Any, before: Any) -> None:
        after = self["/Last"]
        # pypdf starts this walk at /Last, so a `before` other than the last child appends.
        while after.indirect_reference != before:
            if "/Next" not in after:
                after[NameObject("/Next")] = reference
                child_obj[NameObject("/Prev")] = after.indirect_reference
                child_obj.pop("/Next", None)
                self[NameObject("/Last")] = reference
                return
            after = after["/Next"]
        previous = _none_if_null(_raw(after, "/Prev"))
        if previous is None:
            # pypdf leaves /First on the old first child, which orphans this one.
            child_obj.pop("/Prev", None)
            self[NameObject("/First")] = reference
        else:
            previous.get_object()[NameObject("/Next")] = reference
            child_obj[NameObject("/Prev")] = previous
        child_obj[NameObject("/Next")] = after.indirect_reference
        after[NameObject("/Prev")] = reference

    def _unlink_child(self, previous: Any, previous_ref: Any, current: Any) -> None:
        following = _none_if_null(_raw(current, "/Next"))
        count = NumberObject(self["/Count"] - 1)
        if previous is None and following is None:
            count = NumberObject(0)
            del self[NameObject("/First")]
            self.pop("/Last", None)
        elif previous is None:
            del following.get_object()[NameObject("/Prev")]
            self[NameObject("/First")] = following
        elif following is not None:
            following.get_object()[NameObject("/Prev")] = previous_ref
            previous[NameObject("/Next")] = following
        else:
            del previous[NameObject("/Next")]
            self[NameObject("/Last")] = previous_ref
        self[NameObject("/Count")] = count

    def remove_child(self, child: Any) -> None:
        child_obj = child.get_object()
        if "/Parent" not in child_obj:
            raise ValueError("Removed child does not appear to be a tree item")
        if child_obj["/Parent"] != self:
            raise ValueError("Removed child is not a member of this tree")

        previous = previous_ref = None
        current_ref = _raw(self, "/First")
        visited: set[int] = set()
        while current_ref is not None and id(current := current_ref.get_object()) not in visited:
            visited.add(id(current))
            if current == child_obj:
                self._unlink_child(previous, previous_ref, current)
                _reset_tree_links(child_obj)
                return
            previous, previous_ref = current, current_ref
            current_ref = _none_if_null(_raw(current, "/Next"))
        raise ValueError("Removal couldn't find item in tree")

    def remove_from_tree(self) -> None:
        if "/Parent" not in self:
            raise ValueError("Removed child does not appear to be a tree item")
        cast("TreeObject", self["/Parent"]).remove_child(self)

    def empty_tree(self) -> None:
        # A list first: pypdf resets links as it walks them, so it stops after one child.
        for child in list(self):
            _reset_tree_links(child.get_object())
        for key in ("/Count", "/First", "/Last"):
            self.pop(key, None)


class Field(TreeObject):
    """A form field dictionary, as ``get_fields`` returns it."""

    def __init__(self, data: DictionaryObject) -> None:
        DictionaryObject.__init__(self)
        self.indirect_reference = data.indirect_reference
        attributes = (
            FieldDictionaryAttributes.attributes() + CheckboxRadioButtonAttributes.attributes()
        )
        for attribute in attributes:
            if attribute in data:
                self[NameObject(attribute)] = data[attribute]
        value = self.get_resolved("/V")
        if isinstance(value, EncodedStreamObject):
            self[NameObject("/V")] = TextStringObject(value.get_data().decode())

    @property
    def field_type(self) -> NameObject | None:
        return self.get_resolved(FieldDictionaryAttributes.FT)

    @property
    def parent(self) -> DictionaryObject | None:
        return self.get_resolved(FieldDictionaryAttributes.Parent)

    @property
    def kids(self) -> ArrayObject | None:
        return self.get_resolved(FieldDictionaryAttributes.Kids)

    @property
    def name(self) -> str | None:
        return self.get_resolved(FieldDictionaryAttributes.T)

    @property
    def alternate_name(self) -> str | None:
        return self.get_resolved(FieldDictionaryAttributes.TU)

    @property
    def mapping_name(self) -> str | None:
        return self.get_resolved(FieldDictionaryAttributes.TM)

    @property
    def flags(self) -> int | None:
        return self.get_resolved(FieldDictionaryAttributes.Ff)

    @property
    def value(self) -> Any:
        return self.get_resolved(FieldDictionaryAttributes.V)

    @property
    def default_value(self) -> Any:
        return self.get_resolved(FieldDictionaryAttributes.DV)

    @property
    def additional_actions(self) -> DictionaryObject | None:
        return self.get_resolved(FieldDictionaryAttributes.AA)


class Destination(TreeObject):
    """A view of a page. Its written form is a GoTo action (ISO 32000-2 12.3.2)."""

    # The dictionary this was read from, when it came from a file.
    node: DictionaryObject | None = None

    def __init__(self, title: str | bytes, page: Any, fit: Fit) -> None:
        self._filtered_children: list[Any] = []
        DictionaryObject.__init__(self)
        self[NameObject("/Title")] = TextStringObject(title)
        self[NameObject("/Page")] = page
        self[NameObject("/Type")] = fit.fit_type
        for key, value in self._fit_entries(fit.fit_type, fit.fit_args).items():
            self[NameObject(key)] = value

    @staticmethod
    def _fit_entries(kind: str, args: list[Any]) -> dict[str, Any]:
        if kind == TypFitArguments.XYZ:
            left, top, zoom = (*args, *(NumberObject(0) for _ in range(3)))[:3]
            return {TA.LEFT: left, TA.TOP: top, "/Zoom": zoom}
        if not args or kind in {TypFitArguments.FIT, TypFitArguments.FIT_B}:
            return {}
        keys = _FIT_KEYS.get(kind)
        if keys is None:
            raise PdfReadError(f"Unknown Destination Type: {kind!r}")
        if len(args) != len(keys):
            return dict.fromkeys(keys, NullObject())
        return dict(zip(keys, args, strict=True))

    @property
    def dest_array(self) -> ArrayObject:
        output = ArrayObject([self.raw_get("/Page"), self["/Type"]])
        output.extend(
            self[key] for key in ("/Left", "/Bottom", "/Right", "/Top", "/Zoom") if key in self
        )
        return output

    def core_form(self) -> DictionaryObject:
        return DictionaryObject(
            {NameObject("/D"): self.dest_array, NameObject("/S"): NameObject("/GoTo")}
        )

    @property
    def title(self) -> str | None:
        return self.get_resolved("/Title")

    @property
    def page(self) -> Any:
        return _raw(self, "/Page")

    @property
    def typ(self) -> str | None:
        return self.get_resolved("/Type")

    @property
    def zoom(self) -> Any:
        return self.get_resolved("/Zoom")

    @property
    def left(self) -> Any:
        return self.get_resolved("/Left")

    @property
    def right(self) -> Any:
        return self.get_resolved("/Right")

    @property
    def top(self) -> Any:
        return self.get_resolved("/Top")

    @property
    def bottom(self) -> Any:
        return self.get_resolved("/Bottom")

    @property
    def color(self) -> ArrayObject:
        default = ArrayObject([FloatObject(0), FloatObject(0), FloatObject(0)])
        return self.get_resolved("/C", default)

    @property
    def font_format(self) -> OutlineFontFlag:
        return OutlineFontFlag(self.get_resolved("/F", 0))

    @property
    def outline_count(self) -> int | None:
        return self.get_resolved("/Count")


_FIT_KEYS: dict[str, tuple[str, ...]] = {
    TypFitArguments.FIT_R: (TA.LEFT, TA.BOTTOM, TA.RIGHT, TA.TOP),
    TypFitArguments.FIT_H: (TA.TOP,),
    TypFitArguments.FIT_BH: (TA.TOP,),
    TypFitArguments.FIT_V: (TA.LEFT,),
    TypFitArguments.FIT_BV: (TA.LEFT,),
}


class OutlineItem(Destination):
    """An entry of the document outline. Its written form is an outline item dictionary."""

    def core_form(self) -> DictionaryObject:
        form = DictionaryObject()
        for key in ("/Title", "/Parent", "/First", "/Last", "/Next", "/Prev"):
            if key in self:
                form[NameObject(key)] = self.raw_get(key)
        form[NameObject("/Dest")] = self.dest_array
        return form


def _flag(key: str, default: BooleanObject | None) -> property:
    def setter(self: DictionaryObject, value: bool) -> None:
        self[NameObject(key)] = BooleanObject(value is True)

    return property(lambda self: self.get_resolved(key, default), setter)


def _name(key: str, allowed: tuple[str, ...] = (), default: NameObject | None = None) -> property:
    def setter(self: DictionaryObject, value: str) -> None:
        if value[0] != "/":
            raise ValueError(f"{value} does not start with '/'")
        if allowed and value not in allowed:
            raise ValueError(f"{value} is an unacceptable value")
        self[NameObject(key)] = NameObject(value)

    return property(lambda self: self.get_resolved(key, default), setter)


def _array(key: str, *, empty_default: bool = False) -> property:
    def getter(self: DictionaryObject) -> ArrayObject | None:
        return self.get_resolved(key, ArrayObject() if empty_default else None)

    def setter(self: DictionaryObject, value: ArrayObject | None) -> None:
        if value is None:
            self.pop(key, None)
        elif not isinstance(value, ArrayObject):
            raise ValueError("ArrayObject is expected")
        else:
            self[NameObject(key)] = value

    return property(getter, setter)


def _number(key: str) -> property:
    def setter(self: DictionaryObject, value: int) -> None:
        self[NameObject(key)] = NumberObject(value)

    return property(lambda self: self.get_resolved(key), setter)


NOT_SET = BooleanObject(False)


class ViewerPreferences(DictionaryObject):
    """The catalog's /ViewerPreferences dictionary (ISO 32000-2 12.2)."""

    def __init__(self, obj: DictionaryObject | None = None) -> None:
        super().__init__()
        if not is_null_or_none(obj):
            source = cast(DictionaryObject, obj)
            self.update(source.items())
            self.indirect_reference = source.indirect_reference

    @property
    def PRINT_SCALING(self) -> NameObject:  # noqa: N802
        return NameObject("/PrintScaling")

    hide_toolbar = _flag("/HideToolbar", NOT_SET)
    hide_menubar = _flag("/HideMenubar", NOT_SET)
    hide_windowui = _flag("/HideWindowUI", NOT_SET)
    fit_window = _flag("/FitWindow", NOT_SET)
    center_window = _flag("/CenterWindow", NOT_SET)
    display_doctitle = _flag("/DisplayDocTitle", NOT_SET)
    non_fullscreen_pagemode = _name(
        "/NonFullScreenPageMode",
        ("/UseNone", "/UseOutlines", "/UseThumbs", "/UseOC"),
        NameObject("/UseNone"),
    )
    direction = _name("/Direction", ("/L2R", "/R2L"), NameObject("/L2R"))
    view_area = _name("/ViewArea")
    view_clip = _name("/ViewClip")
    print_area = _name("/PrintArea")
    print_clip = _name("/PrintClip")
    print_scaling = _name("/PrintScaling")
    duplex = _name("/Duplex", ("/Simplex", "/DuplexFlipShortEdge", "/DuplexFlipLongEdge"))
    pick_tray_by_pdfsize = _flag("/PickTrayByPDFSize", None)
    print_pagerange = _array("/PrintPageRange")
    num_copies = _number("/NumCopies")
    enforce = _array("/Enforce", empty_default=True)


def _parse_pdf_date(text: str | None) -> datetime | None:
    """pypdf._utils.parse_iso8824_date, which is private to pypdf."""
    if not text:
        return None
    value = f"D:{text}" if text[0].isdigit() else text
    if value.endswith(("Z", "z")):
        value += "0000"
    value = value.replace("z", "+").replace("Z", "+").replace("'", "")
    sign = max(value.find("+"), value.find("-"))
    if sign > 0 and sign != len(value) - 5:
        value += "00"
    for layout in (
        "D:%Y",
        "D:%Y%m",
        "D:%Y%m%d",
        "D:%Y%m%d%H",
        "D:%Y%m%d%H%M",
        "D:%Y%m%d%H%M%S",
        "D:%Y%m%d%H%M%S%z",
    ):
        try:
            parsed = datetime.strptime(value, layout)  # noqa: DTZ007
        except ValueError:
            continue
        return parsed.replace(tzinfo=UTC) if value.endswith("+0000") else parsed
    raise ValueError(f"Can not convert date: {text}")


def _format_pdf_date(moment: datetime) -> str:
    """pypdf._utils.format_iso8824_date, which is private to pypdf."""
    text = moment.strftime("D:%Y%m%d%H%M%S")
    offset = moment.utcoffset()
    if offset is None:
        return text
    seconds = int(offset.total_seconds())
    hours, remainder = divmod(abs(seconds), 3600)
    return f"{text}{'+' if seconds >= 0 else '-'}{hours:02d}'{remainder // 60:02d}'"


def _none_if_null(value: Any) -> Any:
    return None if is_null_or_none(value) else value


class EmbeddedFile:
    """A file embedded in the document (ISO 32000-2 7.11)."""

    def __init__(
        self, name: str, pdf_object: DictionaryObject, parent: ArrayObject | None = None
    ) -> None:
        self._name = name
        self.pdf_object = pdf_object
        self._parent = parent

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def _create_new(cls, writer: Any, name: str, content: str | bytes) -> EmbeddedFile:
        data = content.encode("latin-1") if isinstance(content, str) else content
        file_entry = DecodedStreamObject()
        file_entry.set_data(data)
        file_entry[NameObject("/Type")] = NameObject("/EmbeddedFile")
        ef_entry = DictionaryObject({NameObject("/F"): writer._add_object(file_entry)})
        filespec = DictionaryObject()
        reference = writer._add_object(filespec)
        name_object = cast(TextStringObject, create_string_object(name))
        filespec[NameObject("/Type")] = NameObject("/Filespec")
        filespec[NameObject("/F")] = name_object
        filespec[NameObject("/EF")] = ef_entry
        names = cls._get_names_array(writer)
        index = cls._get_insertion_index(names, name_object)
        names.insert(index, reference)
        names.insert(index, name_object)
        return cls(name=name, pdf_object=filespec, parent=names)

    @classmethod
    def _get_names_array(cls, writer: Any) -> ArrayObject:
        root = writer.root_object
        if "/Names" not in root:
            root[NameObject("/Names")] = writer._add_object(DictionaryObject())
        names_dict = root["/Names"]
        if "/EmbeddedFiles" in names_dict:
            tree = names_dict["/EmbeddedFiles"]
            if "/Names" in tree:
                return tree["/Names"]
            if "/Kids" not in tree:
                raise PdfReadError("Got neither Names nor Kids in embedded files tree.")
        names = ArrayObject()
        if "/EmbeddedFiles" in names_dict:
            # A flat list keeps insertion simple. The file objects stay where they are.
            for kid in names_dict["/EmbeddedFiles"]["/Kids"].get_object():
                names.extend(kid.get_object().get_resolved("/Names", []))
        flat = DictionaryObject({NameObject("/Names"): names})
        names_dict[NameObject("/EmbeddedFiles")] = writer._add_object(flat)
        return names

    @classmethod
    def _get_insertion_index(cls, names_array: ArrayObject, name: str) -> int:
        keys = [names_array[i].encode("utf-8") for i in range(0, len(names_array), 2)]
        return bisect.bisect_right(keys, name.encode("utf-8")) * 2

    @property
    def alternative_name(self) -> str | None:
        # ISO 32000-2 table 43: /UF wins over /F when both are present.
        for key in ("/UF", "/F"):
            if key in self.pdf_object:
                value = self.pdf_object[key].get_object()
                if not is_null_or_none(value):
                    return value
        return None

    @alternative_name.setter
    def alternative_name(self, value: TextStringObject | None) -> None:
        if value is None:
            for key in ("/UF", "/F"):
                if key in self.pdf_object:
                    self.pdf_object[NameObject(key)] = NullObject()
        else:
            self.pdf_object[NameObject("/UF")] = value
            self.pdf_object[NameObject("/F")] = value

    @property
    def description(self) -> str | None:
        return _none_if_null(self.pdf_object.get_resolved("/Desc"))

    @description.setter
    def description(self, value: TextStringObject | None) -> None:
        self.pdf_object[NameObject("/Desc")] = NullObject() if value is None else value

    @property
    def associated_file_relationship(self) -> str:
        return self.pdf_object.get_resolved("/AFRelationship", NameObject("/Unspecified"))

    @associated_file_relationship.setter
    def associated_file_relationship(self, value: NameObject) -> None:
        self.pdf_object[NameObject("/AFRelationship")] = value

    @property
    def _embedded_file(self) -> StreamObject:
        if "/EF" not in self.pdf_object:
            raise PdfReadError(f"/EF entry not found: {self.pdf_object}")
        entries = self.pdf_object["/EF"]
        for key in ("/UF", "/F"):
            if key in entries:
                return entries[key].get_object()
        raise PdfReadError(f"No /(U)F key found in file dictionary: {entries}")

    @property
    def _params(self) -> DictionaryObject:
        return self._embedded_file.get_resolved("/Params", DictionaryObject()).get_object()

    @cached_property
    def _ensure_params(self) -> DictionaryObject:
        embedded_file = self._embedded_file
        if "/Params" not in embedded_file:
            embedded_file[NameObject("/Params")] = DictionaryObject()
        return embedded_file["/Params"]

    @property
    def subtype(self) -> str | None:
        return _none_if_null(self._embedded_file.get_resolved("/Subtype"))

    @subtype.setter
    def subtype(self, value: NameObject | None) -> None:
        self._embedded_file[NameObject("/Subtype")] = NullObject() if value is None else value

    @property
    def content(self) -> bytes:
        return self._embedded_file.get_data()

    @content.setter
    def content(self, value: str | bytes) -> None:
        data = value.encode("latin-1") if isinstance(value, str) else value
        self._embedded_file.set_data(data)

    @property
    def size(self) -> int | None:
        return _none_if_null(self._params.get_resolved("/Size"))

    @size.setter
    def size(self, value: NumberObject | None) -> None:
        self._ensure_params[NameObject("/Size")] = NullObject() if value is None else value

    def _set_date(self, key: str, value: datetime | None) -> None:
        text = NullObject() if value is None else TextStringObject(_format_pdf_date(value))
        self._ensure_params[NameObject(key)] = text

    @property
    def creation_date(self) -> datetime | None:
        return _parse_pdf_date(self._params.get_resolved("/CreationDate"))

    @creation_date.setter
    def creation_date(self, value: datetime | None) -> None:
        self._set_date("/CreationDate", value)

    @property
    def modification_date(self) -> datetime | None:
        return _parse_pdf_date(self._params.get_resolved("/ModDate"))

    @modification_date.setter
    def modification_date(self, value: datetime | None) -> None:
        self._set_date("/ModDate", value)

    @property
    def checksum(self) -> bytes | None:
        return _none_if_null(self._params.get_resolved("/CheckSum"))

    @checksum.setter
    def checksum(self, value: ByteStringObject | None) -> None:
        self._ensure_params[NameObject("/CheckSum")] = NullObject() if value is None else value

    def delete(self) -> None:
        if not self._parent:
            raise PyPdfError("Parent required to delete file from document.")
        reference = getattr(self.pdf_object, "indirect_reference", None)
        if self.pdf_object in self._parent:
            index = self._parent.index(self.pdf_object)
        elif reference is not None and reference in self._parent:
            index = self._parent.index(reference)
        else:
            raise PyPdfError("File not found in parent object.")
        del self._parent[index - 1 : index + 1]  # The name, then its reference.
        self.pdf_object = DictionaryObject()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"

    @classmethod
    def _load_from_names(cls, names: ArrayObject) -> Iterator[EmbeddedFile]:
        # A name tree leaf is [name_1, reference_1, name_2, reference_2, ...].
        for index, entry in enumerate(names):
            if not isinstance(entry, str):
                name = names[index - 1].get_object()
                yield cls(name=name, pdf_object=entry.get_object(), parent=names)

    @classmethod
    def _load(cls, catalog: DictionaryObject) -> Iterator[EmbeddedFile]:
        try:
            container = catalog["/Names"]["/EmbeddedFiles"]
        except KeyError:
            return
        if "/Kids" in container:
            for reference in container["/Kids"].get_object():
                kid = reference.get_object()
                if "/Names" in kid:
                    yield from cls._load_from_names(kid["/Names"])
        if "/Names" in container:
            yield from cls._load_from_names(container["/Names"])


__all__ = (
    "PAGE_FIT",
    "Destination",
    "EmbeddedFile",
    "Field",
    "Fit",
    "OutlineFontFlag",
    "OutlineItem",
    "TreeObject",
    "ViewerPreferences",
    "decode_pdfdocencoding",
    "encode_pdfdocencoding",
    "hex_to_rgb",
    "read_hex_string_from_stream",
    "read_string_from_stream",
)
