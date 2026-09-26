# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from core_pdf_spec.s_07_syntax.trees import MalformedFn, raise_malformed


def skip_malformed(message: str) -> None: ...


def malformed_policy(recover: bool) -> MalformedFn:
    return skip_malformed if recover else raise_malformed


__all__ = ("MalformedFn", "malformed_policy")
