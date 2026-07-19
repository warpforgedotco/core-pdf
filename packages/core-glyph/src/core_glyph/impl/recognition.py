from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

RecognizedGlyph = tuple[str, float]


class GlyphRecognitionBackend(Protocol):
    def recognize(
        self, bitmap: tuple[int, ...], width: int, height: int
    ) -> RecognizedGlyph | None: ...


@dataclass(frozen=True, slots=True)
class TesseractGlyphRecognizer:
    executable: str
    scale: int = 6
    border: int = 18
    minimum_confidence: float = 55.0

    @classmethod
    def from_system(cls) -> TesseractGlyphRecognizer | None:
        executable = shutil.which("tesseract")
        return cls(executable) if executable else None

    def recognize(self, bitmap: tuple[int, ...], width: int, height: int) -> RecognizedGlyph | None:
        return _recognize_cached(
            self.executable,
            self.scale,
            self.border,
            self.minimum_confidence,
            bitmap,
            width,
            height,
        )


@lru_cache(maxsize=8192)
def _recognize_cached(
    executable: str,
    scale: int,
    border: int,
    minimum_confidence: float,
    bitmap: tuple[int, ...],
    width: int,
    height: int,
) -> RecognizedGlyph | None:
    if width <= 0 or height <= 0 or not bitmap:
        return None
    image = bitmap_to_pgm(bitmap, width, height, scale, border)
    try:
        result = subprocess.run(
            [
                executable,
                "stdin",
                "stdout",
                "--psm",
                "10",
                "tsv",
                "-c",
                "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,-+/()[]{}<>|_~",
            ],
            input=image,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_tesseract_symbol(result.stdout, minimum_confidence)


def bitmap_to_pgm(
    bitmap: tuple[int, ...], width: int, height: int, scale: int, border: int
) -> bytes:
    scale = max(1, scale)
    border = max(0, border)
    output_width = width * scale + border * 2
    output_height = height * scale + border * 2
    pixels = bytearray(b"\xff" * (output_width * output_height))
    for y, row in enumerate(bitmap[:height]):
        for x in range(width):
            if not row & (1 << (width - 1 - x)):
                continue
            for output_y in range(border + y * scale, border + (y + 1) * scale):
                start = output_y * output_width + border + x * scale
                pixels[start : start + scale] = b"\x00" * scale
    header = f"P5\n{output_width} {output_height}\n255\n".encode("ascii")
    return header + pixels


def parse_tesseract_symbol(data: bytes, minimum_confidence: float) -> RecognizedGlyph | None:
    lines = data.decode("utf-8", "replace").splitlines()
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) < 12 or fields[0] != "5":
            continue
        try:
            confidence = float(fields[10])
        except ValueError:
            continue
        text = fields[11].strip()
        if confidence < minimum_confidence or len(text) != 1:
            continue
        if text in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,-+/()[]{}<>|_~":
            return text, confidence / 100.0
    return None
