# SPDX-License-Identifier: AGPL-3.0-only

from collections.abc import Mapping
from typing import ClassVar, Literal

from core_pdf_spec.s_07_syntax_primitives.coercion import decoded_name
from core_records import Record, frozen_setattr

RenderingIntent = Literal[
    "RelativeColorimetric", "AbsoluteColorimetric", "Perceptual", "Saturation"
]
BlackPointCompensation = Literal["Default", "ON", "OFF"]


def parse_rendering_intent(value: object) -> RenderingIntent:
    name = decoded_name(value)
    if name is None:
        raise ValueError("rendering intent must be a PDF name")
    match name:
        case "AbsoluteColorimetric" | "Perceptual" | "Saturation":
            return name
        case _:
            return "RelativeColorimetric"


def parse_black_point_compensation(value: object) -> BlackPointCompensation:
    name = decoded_name(value)
    match name:
        case "Default":
            return "Default"
        case "ON":
            return "ON"
        case "OFF":
            return "OFF"
        case _:
            raise ValueError("UseBlackPtComp must be Default, ON, or OFF")


class ColorRendering(Record):
    __slots__ = ("intent", "black_point_compensation")

    intent: RenderingIntent
    black_point_compensation: BlackPointCompensation

    __fields__: ClassVar[tuple[str, ...]] = ("intent", "black_point_compensation")
    __match_args__ = ("intent", "black_point_compensation")

    def __init__(
        self,
        intent: RenderingIntent = "RelativeColorimetric",
        black_point_compensation: BlackPointCompensation = "Default",
    ) -> None:
        frozen_setattr(self, "intent", intent)
        frozen_setattr(self, "black_point_compensation", black_point_compensation)
        self._post_init()

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.intent == other.intent
            and self.black_point_compensation == other.black_point_compensation
        )

    def __hash__(self) -> int:
        return hash((self.intent, self.black_point_compensation))

    def _post_init(self) -> None:
        object.__setattr__(self, "intent", parse_rendering_intent(self.intent))
        object.__setattr__(
            self,
            "black_point_compensation",
            parse_black_point_compensation(self.black_point_compensation),
        )


DEFAULT_COLOR_RENDERING = ColorRendering()


def override_color_rendering(
    dictionary: Mapping[str, object], rendering: ColorRendering = DEFAULT_COLOR_RENDERING
) -> ColorRendering:
    intent = dictionary.get("RI")
    black_point = dictionary.get("UseBlackPtComp")
    return ColorRendering(
        rendering.intent if intent is None else parse_rendering_intent(intent),
        rendering.black_point_compensation
        if black_point is None
        else parse_black_point_compensation(black_point),
    )


def use_black_point_compensation(rendering: ColorRendering, *, default: bool) -> bool:
    if rendering.intent == "AbsoluteColorimetric":
        return False
    return (
        default
        if rendering.black_point_compensation == "Default"
        else (rendering.black_point_compensation == "ON")
    )


__all__ = (
    "BlackPointCompensation",
    "ColorRendering",
    "DEFAULT_COLOR_RENDERING",
    "RenderingIntent",
    "override_color_rendering",
    "parse_black_point_compensation",
    "parse_rendering_intent",
    "use_black_point_compensation",
)
