# SPDX-License-Identifier: AGPL-3.0-only

from typing import overload

from core_pdf_spec.s_07_syntax_primitives.coercion import decoded_name, parse_float, parse_int
from core_pdf_spec.types import PdfName


def recover_pdf_name(value: object, default: str | None = None) -> str | None:
    name = decoded_name(value, default)
    if not isinstance(value, PdfName) and name is not None and name.startswith("/"):
        return name[1:]
    return name


@overload
def lenient_int(value: object, default: None = None) -> int | None: ...


@overload
def lenient_int(value: object, default: int) -> int: ...


def lenient_int(value: object, default: int | None = None) -> int | None:
    return parse_int(value, default, python_syntax=True)


@overload
def lenient_float(value: object, default: None) -> float | None: ...


@overload
def lenient_float(value: object, default: float = 0.0) -> float: ...


def lenient_float(value: object, default: float | None = 0.0) -> float | None:
    return parse_float(value, default, python_syntax=True)
