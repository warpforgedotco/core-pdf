# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations


class PSException(Exception):
    pass


class PSSyntaxError(PSException):
    pass


class PSEOF(PSException):
    pass


__all__ = ("PSEOF", "PSException", "PSSyntaxError")
