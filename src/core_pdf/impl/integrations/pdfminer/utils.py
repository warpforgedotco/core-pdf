# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import io
from pathlib import PurePath
from typing import Any, BinaryIO, TextIO, cast

FileOrName = PurePath | str | io.IOBase
AnyIO = TextIO | BinaryIO


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
    return value.decode("latin-1")


__all__ = ("AnyIO", "FileOrName", "decode_text", "open_filename")
