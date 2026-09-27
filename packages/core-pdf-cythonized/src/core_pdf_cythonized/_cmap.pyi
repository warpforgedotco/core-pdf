# SPDX-License-Identifier: AGPL-3.0-only

def scan_to_unicode_cmap(
    data: bytes,
) -> tuple[dict[bytes, str], list[list[bytes]], str | None] | None: ...
