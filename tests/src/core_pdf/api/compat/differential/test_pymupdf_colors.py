from importlib.resources import files
from io import BytesIO
from typing import Any

import pytest

from core_pdf.api.compat import pymupdf as compat_pymupdf

real_pymupdf = pytest.importorskip("pymupdf")
real_pypdf = pytest.importorskip("pypdf")
pytestmark = pytest.mark.compat_differential


@pytest.mark.parametrize("stroke", [False, True])
@pytest.mark.parametrize(
    "space", ["DeviceGray", "DeviceRGB", "ICCRGB", "DeviceCMYK", "Separation", "DeviceN", "Indexed"]
)
@pytest.mark.parametrize(
    "components",
    [(0, 0, 0, 1), (0.5, 0.3, 0, 0.1), (0.12, 0.23, 0.34, 0.45), (0.50001, 0.30001, 0, 0.10001)],
)
def test_device_span_colors_and_graphics_state(
    space: str, components: tuple[float, ...], stroke: bool
) -> None:
    generic = real_pypdf.generic
    name = generic.NameObject
    writer = real_pypdf.PdfWriter()
    page = writer.add_blank_page(width=300, height=200)
    font = generic.DictionaryObject(
        {
            name("/Type"): name("/Font"),
            name("/Subtype"): name("/Type1"),
            name("/BaseFont"): name("/Helvetica"),
        }
    )
    resources = generic.DictionaryObject(
        {name("/Font"): generic.DictionaryObject({name("/F1"): writer._add_object(font)})}
    )
    if space == "DeviceGray":
        color = str(components[0]) + (" G" if stroke else " g")
    elif space == "DeviceRGB":
        color = " ".join(str(value) for value in components[:3]) + (" RG" if stroke else " rg")
    elif space == "ICCRGB":
        profile = generic.DecodedStreamObject()
        profile.set_data(files("core_pdf._vendor.icc").joinpath("Artifex-sRGB.icc").read_bytes())
        profile[name("/N")] = generic.NumberObject(3)
        profile[name("/Alternate")] = name("/DeviceRGB")
        resources[name("/ColorSpace")] = generic.DictionaryObject(
            {name("/CS1"): generic.ArrayObject([name("/ICCBased"), writer._add_object(profile)])}
        )
        value = " ".join(str(component) for component in components[:3])
        color = f"/CS1 {'CS' if stroke else 'cs'} {value} {'SCN' if stroke else 'scn'}"
    elif space == "DeviceCMYK":
        color = " ".join(str(value) for value in components) + (" K" if stroke else " k")
    else:
        if space == "Indexed":
            color_space = generic.ArrayObject(
                [
                    name("/Indexed"),
                    name("/DeviceCMYK"),
                    generic.NumberObject(0),
                    generic.ByteStringObject(bytes(round(value * 255) for value in components)),
                ]
            )
            value = "0"
        else:
            function = generic.DictionaryObject(
                {
                    name("/FunctionType"): generic.NumberObject(2),
                    name("/Domain"): generic.ArrayObject(
                        [generic.NumberObject(0), generic.NumberObject(1)]
                    ),
                    name("/C0"): generic.ArrayObject([generic.NumberObject(0)] * 4),
                    name("/C1"): generic.ArrayObject(
                        [generic.FloatObject(value) for value in components]
                    ),
                    name("/N"): generic.NumberObject(1),
                }
            )
            color_space = generic.ArrayObject(
                [
                    name("/" + space),
                    generic.ArrayObject([name("/Example")])
                    if space == "DeviceN"
                    else name("/Example"),
                    name("/DeviceCMYK"),
                    writer._add_object(function),
                ]
            )
            value = "0.7"
        resources[name("/ColorSpace")] = generic.DictionaryObject({name("/CS1"): color_space})
        color = f"/CS1 {'CS' if stroke else 'cs'} {value} {'SCN' if stroke else 'scn'}"
    page[name("/Resources")] = resources
    stream = generic.DecodedStreamObject()
    stream.set_data(
        (
            f"BT /F1 12 Tf 20 150 Td {int(stroke)} Tr {color} (A) Tj "
            "q 0 0 0 1 k 0 0 0 1 K 0 -30 Td (B) Tj Q 20 0 Td (C) Tj ET"
        ).encode()
    )
    page[name("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)

    def snapshot(module: Any) -> list[tuple[str, int]]:
        with module.open(stream=output.getvalue(), filetype="pdf") as document:
            return [
                (span["text"], span["color"])
                for block in document[0].get_text("dict")["blocks"]
                for line in block.get("lines", [])
                for span in line["spans"]
            ]

    assert snapshot(compat_pymupdf) == snapshot(real_pymupdf)
