import math
import random
import struct

from core_pdf.impl.layout_lines import LayoutLine, layout_line_segment_char_bbox
from core_pdf.impl.runs import LayoutLineText, LayoutLineTextSegment
from core_pdf.impl.text import TextWord


def per_character_words(reconstructed: LayoutLineText) -> tuple[str, tuple[TextWord, ...]]:
    parts: list[str] = []
    words: list[TextWord] = []
    word = ""
    box = [0.0, 0.0, 0.0, 0.0]

    def flush() -> None:
        nonlocal word
        if word:
            words.append(TextWord(word, (box[0], box[1], box[2], box[3])))
            word = ""

    def space() -> None:
        if parts and parts[-1] == " ":
            return
        flush()
        parts.append(" ")

    for segment in reconstructed.segments:
        if segment.separator_before:
            space()
        for index, char in enumerate(segment.text):
            if char.isspace():
                space()
                continue
            bx0, by0, bx1, by1 = layout_line_segment_char_bbox(segment, index, len(segment.text))
            if word:
                box[0] = min(box[0], bx0)
                box[1] = min(box[1], by0)
                box[2] = max(box[2], bx1)
                box[3] = max(box[3], by1)
            else:
                box[:] = [bx0, by0, bx1, by1]
            word += char
            parts.append(char)
    flush()
    return "".join(parts).rstrip(), tuple(words)


def exact(result: tuple[str, tuple[TextWord, ...]]) -> object:
    text, words = result
    return text, [
        (word.text, [struct.pack("<d", value) for value in word.bbox or ()]) for word in words
    ]


def random_segment(rng: random.Random) -> LayoutLineTextSegment:
    text = "".join(rng.choice("ab c \tde fg") for _ in range(rng.randint(0, 9)))
    x0 = rng.choice([0.0, -0.0, rng.uniform(-50, 600), 1e-300])
    width = rng.choice([0.0, -3.5, rng.uniform(0, 80), 1e-310, math.nan, math.inf])
    y0 = rng.uniform(-50, 800)
    box = (x0, y0, x0 + width, y0 + rng.uniform(0, 12))
    return LayoutLineTextSegment(text, rng.choice(["", " "]), box, rng.choice([0, 0, 90, 180, 270]))


def test_run_boxes_match_the_per_character_walk() -> None:
    rng = random.Random(7)
    for _ in range(20_000):
        segments = tuple(random_segment(rng) for _ in range(rng.randint(1, 4)))
        reconstructed = LayoutLineText("".join(s.text for s in segments), segments)
        got = LayoutLine.text_and_words(LayoutLine.__new__(LayoutLine), reconstructed)
        assert exact(got) == exact(per_character_words(reconstructed))
