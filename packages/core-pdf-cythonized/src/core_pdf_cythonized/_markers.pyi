# SPDX-License-Identifier: AGPL-3.0-only

from collections.abc import Buffer, Sequence

def markers_present(data: Buffer, needles: tuple[bytes, ...]) -> tuple[bool, ...]: ...
def regions_with_markers(
    data: Buffer, needles: tuple[bytes, ...], starts: Sequence[int], data_len: int
) -> list[set[int]]: ...
