# SPDX-License-Identifier: AGPL-3.0-only
"""Stable Markdown rendering for native extraction regression snapshots."""

from __future__ import annotations

from typing import Any


def native_snapshot(fixture_name: str, page: Any, result: Any) -> str:
    """Render one native extraction result as a stable Markdown snapshot."""
    lines = [
        "---",
        f"fixture: {fixture_name}",
        f"rotation: {page.rotation}",
        f"page_class: {result.page_class}",
        f"base_route: {result.base_route}",
        f"block_count: {len(result.blocks)}",
        f"line_count: {len(result.resolved_lines)}",
        "---",
        "",
    ]
    for index, line in enumerate(result.resolved_lines, 1):
        lines.extend(
            (
                f"<!-- line: {index:03d}; break_before: {line.break_before}; "
                f"kind: {line.kind}; source: {line.source} -->",
                "```text",
                line.text,
                "```",
            )
        )
    return "\n".join(lines) + "\n"


__all__ = ("native_snapshot",)
