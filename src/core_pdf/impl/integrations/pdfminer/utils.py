# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import io
from pathlib import PurePath
from typing import Any, BinaryIO, TextIO, cast

FileOrName = PurePath | str | io.IOBase
AnyIO = TextIO | BinaryIO

PDF_DOC_ENCODING = (
    "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f"
    "\x10\x11\x12\x13\x14\x15\x17\x17˘ˇˆ˙˝˛˚˜\x20!\"#$%&'()*+,-./"
    "0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\x00"
    "•†‡…—–ƒ⁄‹›−‰„“”‘’‚™ﬁﬂŁŒŠŸŽıłœšž\x00€¡¢£¤¥¦§¨©ª«¬\x00®¯"
    "°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞß"
    "àáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ"
)


class open_filename:
    def __init__(self, filename: FileOrName, *args: Any, **kwargs: Any) -> None:
        if isinstance(filename, PurePath):
            filename = str(filename)
        if isinstance(filename, str):
            self.file_handler: AnyIO = open(filename, *args, **kwargs)  # noqa: SIM115
            self.closing = True
        elif isinstance(filename, io.IOBase):
            self.file_handler = cast(AnyIO, filename)
            self.closing = False
        else:
            raise TypeError(f"Unsupported input type: {type(filename)}")

    def __enter__(self) -> AnyIO:
        return self.file_handler

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self.closing:
            self.file_handler.close()


def decode_text(value: bytes) -> str:
    if value.startswith(b"\xfe\xff"):
        return value[2:].decode("utf-16be", "ignore")
    return "".join(PDF_DOC_ENCODING[byte] for byte in value)


__all__ = ("AnyIO", "FileOrName", "decode_text", "open_filename")
