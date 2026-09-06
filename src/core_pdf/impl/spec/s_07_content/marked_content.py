# SPDX-License-Identifier: AGPL-3.0-only
"""Properties scoped by PDF marked-content operators."""

from dataclasses import dataclass


@dataclass(slots=True)
class MarkedContentEntry:
    layer: str | None = None
    actual_text: str | None = None
    mcid: int | None = None
