# Follows pypdf's form filling and appearance streams, Copyright (c) 2006-2008, Mathieu
# Fenniak, under its BSD-style license.
"""The parts of PdfWriter that fill forms, with the appearance streams of text fields.

A field's font is read from its resource dictionary, or from the standard 14 metrics. Text
goes out in one byte per character. pypdf's fallback of re-encoding an embedded TrueType font
for a character the font lacks, and its right-to-left shaping, are not here.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any, NamedTuple, cast

from core_adobe_fonts.agl.glyph_list import GLYPH_DATA
from core_pdf.impl.fonts_metrics import FONT_DATA, standard_14_widths

from ._page_merge import Transformation
from .constants import FieldDictionaryAttributes as FA  # noqa: N814
from .errors import PyPdfError
from .generic import (
    ArrayObject,
    BooleanObject,
    DecodedStreamObject,
    DictionaryObject,
    FloatObject,
    IndirectObject,
    NameObject,
    NumberObject,
    RectangleObject,
    StreamObject,
    TextStringObject,
)

logger = logging.getLogger(__name__)

DEFAULT_FONT_SIZE_IN_MULTILINE = 12
MINIMUM_FONT_SIZE = 4.0
FONT_SIZE_STEP = 0.2
GLYPH_UNITS = 1000
ENCODINGS = {
    "/WinAnsiEncoding": "cp1252",
    "/MacRomanEncoding": "mac_roman",
    "/StandardEncoding": "cp1252",
}
# What a standard 14 font can switch to for characters outside its usual encoding.
CORE_FONT_ENCODINGS = ("cp1250", "cp1252", "cp1254", "cp1257", "iso8859_15")
LEFT, CENTER, RIGHT = 0, 1, 2


def _character_table(encoding: str) -> tuple[str, ...]:
    return tuple(bytes((code,)).decode(encoding, "replace") for code in range(256))


class _Font(NamedTuple):
    """What an appearance stream needs to know about a font."""

    widths: Mapping[int, float]
    default_width: float
    bbox: tuple[float, float, float, float]
    ascent: float
    encoding: str = "cp1252"
    core_name: str | None = None

    @classmethod
    def core(cls, name: str = "Helvetica", encoding: str = "cp1252") -> _Font:
        props = cast(dict[str, Any], FONT_DATA[name]["props"])
        widths = standard_14_widths(name, _character_table(encoding)) or {}
        left, bottom, right, top = props["FontBBox"]
        return cls(widths, 0.0, (left, bottom, right, top), props["Ascent"], encoding, name)

    @classmethod
    def from_resource(cls, resource: DictionaryObject) -> _Font:
        encoding = ENCODINGS.get(str(resource.get_resolved("/Encoding")), "cp1252")
        base = str(resource.get_resolved("/BaseFont", "")).removeprefix("/").split("+")[-1]
        descriptor = resource.get_resolved("/FontDescriptor")
        fallback = cls.core(base if base in FONT_DATA else "Helvetica", encoding)
        if not isinstance(descriptor, DictionaryObject):
            return fallback
        widths = dict(fallback.widths)
        first = resource.get_resolved("/FirstChar", 0)
        listed = resource.get_resolved("/Widths", ())
        for offset, width in enumerate(listed):
            widths[first + offset] = float(width.get_object())
        box = descriptor.get_resolved("/FontBBox", fallback.bbox)
        left, bottom, right, top = (float(item.get_object()) for item in box)
        return cls(
            widths,
            float(descriptor.get_resolved("/MissingWidth", 0)),
            (left, bottom, right, top),
            float(descriptor.get_resolved("/Ascent", fallback.ascent)),
            encoding,
            None,
        )

    def text_width(self, text: str) -> float:
        codes = text.encode(self.encoding, "replace")
        return sum(self.widths.get(code, self.default_width) for code in codes)

    def can_encode(self, text: str) -> bool:
        try:
            text.encode(self.encoding)
        except UnicodeEncodeError:
            return False
        return True

    def core_resource(self) -> DictionaryObject:
        encoding: Any = NameObject("/WinAnsiEncoding")
        if self.encoding != "cp1252":
            names = {text: name for name, text in GLYPH_DATA.items()}
            table = _character_table(self.encoding)
            differences = ArrayObject([NumberObject(128)])
            differences.extend(NameObject(f"/{names.get(c, '.notdef')}") for c in table[128:])
            encoding = DictionaryObject({NameObject("/Differences"): differences})
        return DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject(f"/{self.core_name or 'Helvetica'}"),
                NameObject("/Encoding"): encoding,
            }
        )


class _Line(NamedTuple):
    width: float
    text: str


class _Layout(NamedTuple):
    rectangle: RectangleObject
    border_width: float
    border_style: str
    is_multiline: bool
    alignment: int
    is_comb: bool
    max_length: int | None


def _wrap(
    font: _Font, size: float, leading: float, width: float, height: float, text: str
) -> tuple[list[_Line], float]:
    """Break text into lines, reducing the size until they fit the field."""
    space = font.text_width(" ") * size / GLYPH_UNITS
    lines: list[_Line] = []
    for paragraph in text.splitlines():
        words: list[str] = []
        used = 0.0
        for word in paragraph.split(" "):
            word_width = font.text_width(word) * size / GLYPH_UNITS
            if words and used + space + word_width > width:
                lines.append(_Line(used, " ".join(words)))
                words, used = [word], word_width
            elif not words and word_width > width:
                lines.append(_Line(word_width, word))
            else:
                used += word_width + (space if words else 0)
                words.append(word)
        if words or not paragraph.strip():
            lines.append(_Line(used, " ".join(words)))
    needed = size + (len(lines) - 1) * leading * size
    if needed > height and size - FONT_SIZE_STEP >= MINIMUM_FONT_SIZE:
        return _wrap(font, size - FONT_SIZE_STEP, leading, width, height, text)
    return lines, round(size, 1)


def _escape(data: bytes) -> bytes:
    return data.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def _layout_lines(
    text: str,
    selection: list[str] | None,
    font: _Font,
    size: float,
    layout: _Layout,
    width: float,
    height: float,
) -> tuple[list[_Line], float, bool]:
    leading = (font.bbox[3] - font.bbox[1]) / GLYPH_UNITS
    multiline = layout.is_multiline
    if size == 0:
        if selection:
            # Lines are matched against the selection, so they are not wrapped.
            multiline = False
        if multiline:
            lines, size = _wrap(font, DEFAULT_FONT_SIZE_IN_MULTILINE, leading, width, height, text)
            return lines, size, multiline
        unscaled = font.text_width(text) / GLYPH_UNITS
        fitted = min(height / leading, width / (unscaled or 1))
        size = round(max(fitted, MINIMUM_FONT_SIZE), 1)
        return [_Line(unscaled * size, text)], size, multiline
    if layout.is_comb:
        limit = layout.max_length
        if limit and len(text) > limit:
            logger.warning(
                "Length of text %s exceeds maximum length (%d) of field, input truncated.",
                text,
                limit,
            )
        characters = text[: limit or len(text)]
        return (
            [_Line(font.text_width(c) * size / GLYPH_UNITS, c) for c in characters],
            size,
            multiline,
        )
    return (
        [_Line(font.text_width(line) * size / GLYPH_UNITS, line) for line in text.splitlines()],
        size,
        multiline,
    )


def _appearance_data(
    text: str,
    selection: list[str] | None,
    font: _Font,
    font_name: str,
    font_size: float,
    font_color: str,
    layout: _Layout,
) -> bytes:
    rectangle = layout.rectangle
    box_width, box_height = float(rectangle.width), float(rectangle.height)
    leading = (font.bbox[3] - font.bbox[1]) / GLYPH_UNITS
    margin = max(layout.border_width * (2 if layout.border_style in {"/B", "/I"} else 1), 1)
    height = box_height - 2 * margin
    width = box_width - 4 * margin
    lines, size, multiline = _layout_lines(text, selection, font, font_size, layout, width, height)
    if multiline:
        y_offset = box_height + margin - font.bbox[3] * size / GLYPH_UNITS
    else:
        y_offset = margin + (height - font.ascent * size / GLYPH_UNITS) / 2
    appearance = f"{font_name} {size} Tf {font_color}"
    output = bytearray(
        f"q\n/Tx BMC \nq\n{2 * margin} {margin} {width} {height} re\nW\nBT\n{appearance}\n".encode()
    )
    selected = "".join(selection or ())
    position: float = 0
    for number, (line_width, line) in enumerate(lines):
        if selected and line in selected:
            output += (
                f"1 {y_offset - number * size * leading - 1} {box_width - 2} {size + 2} re\n"
                f"0.5 0.5 0.5 rg s\n{appearance}\n"
            ).encode()
        if layout.is_comb and layout.max_length:
            cell = box_width / layout.max_length
            start = number * cell + (cell - line_width) / 2
        elif layout.alignment == RIGHT:
            start = box_width - margin * 2 - line_width
        elif layout.alignment == CENTER:
            start = (box_width - line_width) / 2
        else:
            start = margin * 2
        if number == 0:
            down = y_offset
        else:
            down = 0.0 if layout.is_comb else -size * leading
        output += f"{start - position} {down} Td\n".encode()
        position = start
        output += b"(" + _escape(line.encode(font.encoding, "replace")) + b") Tj\n"
    return bytes(output + b"ET\nQ\nEMC\nQ\n")


def _find_font(
    font_name: str, annotation: DictionaryObject, form: DictionaryObject, text: str
) -> tuple[str, _Font, Any]:
    """The font to draw with, its name, and its resource if the file has one."""
    resources = annotation.get_inherited("/DR", form.get_resolved("/DR", DictionaryObject()))
    fonts = resources.get_resolved("/Font", DictionaryObject())
    resource = fonts.get_resolved(font_name)
    if resource:
        font = _Font.from_resource(resource)
    else:
        if font_name.removeprefix("/") not in FONT_DATA:
            logger.warning("Font dictionary for %s not found; defaulting to Helvetica.", font_name)
            font_name = "/Helvetica"
        font = _Font.core(font_name.removeprefix("/"))
    if not font.can_encode(text) and font.core_name is not None:
        # A standard 14 font has no embedded program, so it can take another encoding.
        for encoding in CORE_FONT_ENCODINGS:
            other = _Font.core(font.core_name, encoding)
            if other.can_encode(text):
                return f"/PYPDF1{encoding}", other, None
    if not font.can_encode(text):
        logger.warning(
            "Text string '%s' contains characters not supported by font encoding. "
            "This may result in text corruption. "
            "Consider calling writer.update_page_form_field_values with auto_regenerate=True.",
            text,
        )
    return font_name, font, resource


def _add_rotation(stream: StreamObject, rectangle: RectangleObject, rotation: int) -> None:
    matrix = Transformation().rotate(rotation)
    width, height = float(rectangle.width), float(rectangle.height)
    corners = [matrix.apply_on(c) for c in ((width, 0.0), (0.0, height), (width, height))]
    matrix = matrix.translate(
        -min(0.0, *(c[0] for c in corners)), -min(0.0, *(c[1] for c in corners))
    )
    stream[NameObject("/Matrix")] = ArrayObject(FloatObject(round(i, 3)) for i in matrix.ctm)


def text_appearance(
    writer: Any,
    page: Any,
    flatten: bool,
    form: DictionaryObject,
    field: DictionaryObject,
    annotation: DictionaryObject,
    user_font_name: str = "",
    user_font_size: float = -1,
) -> DecodedStreamObject:
    """An appearance stream for a text or choice field."""
    left, bottom, right, top = (float(item) for item in annotation.get_resolved("/Rect"))
    width, height = abs(right - left), abs(top - bottom)
    if page.get_inherited("/Rotate") in {90, 270}:
        width, height = height, width
    rectangle = RectangleObject((0, 0, width, height))

    default_appearance = annotation.get_inherited("/DA", form.get_resolved("/DA"))
    if not default_appearance:
        default_appearance = TextStringObject("/Helv 0 Tf 0 g")

    flags = field.get_resolved(FA.Ff, 0)
    selection: list[str] = []
    if field.get_resolved(FA.FT, "/Tx") == "/Ch" and flags & FA.FfBits.Combo == 0:
        options = annotation.get_inherited(FA.Opt, [])
        text = "\n".join(_option_text(item) for item in options)
        value = field.get_resolved("/V", [])
        selection = list(value) if isinstance(value, list) else [value]
    else:
        text = field.get_resolved("/V", "")

    # /DA holds at least "font size Tf" (ISO 32000-2 12.7.4.3). The rest is the colour.
    properties = [item for item in re.split(r"\s", default_appearance) if item]
    da_font_name = properties.pop(properties.index("Tf") - 2)
    font_size = float(properties.pop(properties.index("Tf") - 1))
    properties.remove("Tf")
    font_size = user_font_size if user_font_size > 0 else font_size

    font_name, font, resource = _find_font(user_font_name or da_font_name, annotation, form, text)
    if font_name != da_font_name:
        annotation[NameObject("/DA")] = TextStringObject(
            default_appearance.replace(da_font_name, font_name)
        )

    reference = _sync_font(writer, font_name, font, resource, form, page if flatten else None)

    border_width, border_style = 1, "/S"
    border = field.get_resolved("/BS")
    if border is not None:
        border_width = border.get_resolved("/W", border_width)
        border_style = border.get_resolved("/S", border_style)
    comb = bool(flags & FA.FfBits.Comb)
    layout = _Layout(
        rectangle,
        border_width,
        border_style,
        bool(flags & FA.FfBits.Multiline),
        field.get_resolved("/Q", LEFT),
        comb,
        annotation.get_resolved("/MaxLen") if comb else None,
    )
    stream = DecodedStreamObject()
    stream[NameObject("/Type")] = NameObject("/XObject")
    stream[NameObject("/Subtype")] = NameObject("/Form")
    stream[NameObject("/BBox")] = rectangle
    characteristics = field.get_inherited("/MK")
    rotation = 0
    if isinstance(characteristics, DictionaryObject):
        rotation = int(characteristics.get_resolved("/R", 0)) % 360
    if rotation:
        _add_rotation(stream, rectangle, rotation)
    stream.set_data(
        _appearance_data(text, selection, font, font_name, font_size, " ".join(properties), layout)
    )
    stream[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject(font_name): reference})}
    )
    _keep_appearance_entries(annotation, stream, font_name, reference)
    return stream


def _option_text(option: Any) -> str:
    option = option.get_object()
    # An option is its text, or a pair of the exported value and its text.
    return option[1].get_object() if isinstance(option, list) else option


def _sync_font(
    writer: Any,
    font_name: str,
    font: _Font,
    resource: Any,
    form: DictionaryObject,
    page: Any,
) -> Any:
    """Make sure the form's resources, and the page's when flattening, have the font."""
    if "/DR" not in form:
        form[NameObject("/DR")] = DictionaryObject()
    resources = form["/DR"]
    if "/Font" not in resources:
        resources[NameObject("/Font")] = DictionaryObject()
    fonts = resources["/Font"]
    if font_name not in fonts:
        created = resource or font.core_resource()
        fonts[NameObject(font_name)] = writer._add_object(created)
    reference = fonts.get(font_name)
    if page is not None:
        page_resources = page["/Resources"]
        if "/Font" not in page_resources:
            page_resources[NameObject("/Font")] = DictionaryObject()
        page_fonts = page_resources["/Font"]
        if font_name not in page_fonts:
            page_fonts[NameObject(font_name)] = reference
    return reference


def _keep_appearance_entries(
    annotation: DictionaryObject, stream: DecodedStreamObject, font_name: str, reference: Any
) -> None:
    appearance = annotation.get_resolved("/AP")
    if appearance is None:
        return
    normal = appearance.get_resolved("/N")
    if not isinstance(normal, DictionaryObject):
        return
    for key, value in dict.items(normal):
        if key in {"/BBox", "/Length", "/Subtype", "/Type", "/Filter"}:
            continue
        if key != "/Resources":
            stream[NameObject(key)] = value
            continue
        resources = value.get_object()
        if "/Font" not in resources:
            resources[NameObject("/Font")] = DictionaryObject()
        resources["/Font"][NameObject(font_name)] = reference


class FormWriter:
    _objects: list[Any]
    _root_object: DictionaryObject
    pages: Any
    _add_object: Any
    _get_qualified_field_name: Any

    FFBITS_NUL = FA.FfBits(0)

    def set_need_appearances_writer(self, state: bool = True) -> None:
        if "/AcroForm" not in self._root_object:
            self._root_object[NameObject("/AcroForm")] = self._add_object(DictionaryObject())
        self._root_object["/AcroForm"][NameObject("/NeedAppearances")] = BooleanObject(state)

    def _merge_content_stream_to_page(self, page: Any, new_content_data: bytes) -> None:
        """Add content to a page, on top of what it already draws."""
        name = NameObject("/Contents")
        existing = page[name].get_object() if name in page else None
        if isinstance(existing, ArrayObject):
            stream = StreamObject()
            stream.set_data(new_content_data)
            existing.append(self._add_object(stream))
            page[name] = self._add_object(existing)
            return
        stream = StreamObject()
        if isinstance(existing, StreamObject):
            stream.set_data(existing.get_data() + b"\n" + new_content_data)
        else:
            stream.set_data(new_content_data)
        page[name] = self._add_object(stream)

    def _add_apstream_object(
        self, page: Any, appearance: Any, object_name: str, x_offset: float, y_offset: float
    ) -> None:
        """Draw an appearance stream on the page, as an XObject."""
        resources = page["/Resources"]
        reference = self._add_object(appearance)
        name = NameObject(f"/Fm_{object_name}")._sanitize()
        if "/XObject" not in resources:
            resources[NameObject("/XObject")] = DictionaryObject()
        xobjects = resources["/XObject"]
        if name in xobjects:
            logger.warning(
                "XObject %r already added to page resources. This might be an issue.", name
            )
        else:
            xobjects[name] = reference
        move = Transformation().translate(x_offset, y_offset)
        drawing = f"q\n{move._to_cm()}\n{name} Do\nQ".encode()
        self._merge_content_stream_to_page(page, drawing)

    def update_page_form_field_values(
        self,
        page: Any,
        fields: Mapping[str, Any],
        flags: FA.FfBits = FFBITS_NUL,
        auto_regenerate: bool | None = True,
        flatten: bool = False,
    ) -> None:
        if "/AcroForm" not in self._root_object:
            raise PyPdfError("No /AcroForm dictionary in PDF of PdfWriter Object")
        form = self._root_object["/AcroForm"]
        if "/Fields" not in form:
            raise PyPdfError("No /Fields dictionary in PDF of PdfWriter Object")
        if isinstance(auto_regenerate, bool):
            self.set_need_appearances_writer(auto_regenerate)
        if page is None:
            page = list(self.pages)
        if isinstance(page, list):
            for item in page:
                if "/Annots" in item:
                    self.update_page_form_field_values(item, fields, flags, None, flatten=flatten)
            return
        if "/Annots" not in page:
            logger.warning("No fields to update on this page")
            return
        for item in page["/Annots"]:
            annotation = item.get_object()
            if annotation.get_resolved("/Subtype", "") != "/Widget":
                continue
            if "/FT" in annotation and "/T" in annotation:
                parent = annotation
            else:
                parent = annotation.get_resolved("/Parent", DictionaryObject())
            for name, value in fields.items():
                if (
                    self._get_qualified_field_name(parent=parent) == name
                    or parent.get_resolved("/T") == name
                ):
                    self._fill_field(page, form, annotation, parent, name, value, flags, flatten)

    def _fill_field(
        self,
        page: Any,
        form: DictionaryObject,
        annotation: DictionaryObject,
        parent: DictionaryObject,
        name: str,
        value: Any,
        flags: Any,
        flatten: bool,
    ) -> None:
        kind = parent.get_resolved(FA.FT)
        if kind == "/Ch" and "/I" in parent:
            del parent["/I"]
        if flags:
            annotation[NameObject(FA.Ff)] = NumberObject(flags)
        if not (value is None and flatten):
            if isinstance(value, list):
                parent[NameObject(FA.V)] = ArrayObject(TextStringObject(v) for v in value)
            elif isinstance(value, tuple):
                annotation[NameObject(FA.V)] = TextStringObject(value[0])
            else:
                parent[NameObject(FA.V)] = TextStringObject(value)
        appearance: Any = None
        if kind == "/Btn":
            state = NameObject(value)
            normal = annotation["/AP"]["/N"]
            if state not in normal:
                state = NameObject("/Off")
            appearance = normal.get(state)
            annotation[NameObject("/AS")] = state
            annotation[NameObject(FA.V)] = state
        elif kind in {"/Tx", "/Ch"}:
            extra = (value[1], value[2]) if isinstance(value, tuple) else ()
            appearance = text_appearance(self, page, flatten, form, parent, annotation, *extra)
            self._store_appearance(annotation, appearance)
        elif annotation.get_resolved(FA.FT) == "/Sig":
            logger.warning("Signature forms not implemented yet")
        if appearance is not None and flatten:
            left, bottom = annotation["/Rect"][0], annotation["/Rect"][1]
            target = appearance.get_object() if appearance else appearance
            self._add_apstream_object(page, target, name, left, bottom)

    def _store_appearance(self, annotation: DictionaryObject, stream: StreamObject) -> None:
        if "/AP" not in annotation:
            annotation[NameObject("/AP")] = DictionaryObject()
        entries = annotation["/AP"]
        current = entries.get("/N")
        if isinstance(current, IndirectObject):
            self._objects[current.idnum - 1] = stream
            stream.indirect_reference = IndirectObject(current.idnum, 0, self)
        else:
            entries[NameObject("/N")] = self._add_object(stream)

    def reattach_fields(self, page: Any = None) -> list[DictionaryObject]:
        """Put widgets that are missing from the form's /Fields back in it."""
        if page is None:
            return [field for item in self.pages for field in self.reattach_fields(item)]
        catalog = self._root_object
        if "/AcroForm" not in catalog:
            catalog[NameObject("/AcroForm")] = DictionaryObject()
        form = catalog["/AcroForm"]
        if "/Fields" not in form:
            form[NameObject("/Fields")] = ArrayObject()
        fields = form["/Fields"]
        output: list[DictionaryObject] = []
        if "/Annots" not in page:
            return output
        annotations = page["/Annots"]
        for index, item in enumerate(list(annotations)):
            annotation = item.get_object()
            if annotation.get_resolved("/Subtype", "") != "/Widget" or "/FT" not in annotation:
                continue
            if annotation.indirect_reference in fields:
                continue
            if not isinstance(item, IndirectObject):
                annotations[index] = self._add_object(annotation)
            fields.append(annotation.indirect_reference)
            output.append(annotation)
        return output


__all__ = ["FormWriter"]
