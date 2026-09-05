# SPDX-License-Identifier: AGPL-3.0-only
internal_JP2_SIGNATURE = b"\x00\x00\x00\x0cjP  \r\n\x87\n"


def internal_jpx_uses_irreversible_wavelet(data: bytes | memoryview) -> bool | None:
    """Return whether a JPEG 2000 image uses the irreversible 9/7 transform.

    JPEG 2000 records the wavelet transform in the ``QMFBID`` byte of COD and
    COC marker segments.  A zero value selects the irreversible 9/7 transform;
    one selects the reversible 5/3 transform.  ``None`` means that the stream
    is malformed or uses a layout this small classifier cannot inspect safely.

    Both raw codestreams and JP2/JPX files containing contiguous codestream
    boxes are supported.  Returning a tri-state result is intentional: callers
    must not silently treat an unclassified stream as reversible.
    """

    view = internal_byte_view(data)
    if view is None:
        return None
    codestreams = internal_contiguous_codestreams(view)
    if codestreams is None:
        return None

    saw_reversible = False
    saw_unknown = False
    for codestream in codestreams:
        result = internal_codestream_uses_irreversible_wavelet(codestream)
        if result is True:
            return True
        if result is False:
            saw_reversible = True
        else:
            saw_unknown = True
    if saw_reversible and not saw_unknown:
        return False
    return None


def internal_byte_view(data: bytes | memoryview) -> memoryview | None:
    try:
        view = memoryview(data)
        if view.ndim != 1 or view.format != "B":
            view = view.cast("B")
    except (TypeError, ValueError):
        return None
    return view


def internal_contiguous_codestreams(view: memoryview) -> tuple[memoryview, ...] | None:
    if len(view) >= 2 and view[0] == 0xFF and view[1] == 0x4F:
        return (view,)
    if len(view) < len(internal_JP2_SIGNATURE):
        return None
    if view[: len(internal_JP2_SIGNATURE)].tobytes() != internal_JP2_SIGNATURE:
        return None

    codestreams: list[memoryview] = []
    offset = 0
    while offset < len(view):
        if len(view) - offset < 8:
            return None
        box_length = int.from_bytes(view[offset : offset + 4], "big")
        box_type = view[offset + 4 : offset + 8].tobytes()
        header_length = 8
        if box_length == 1:
            if len(view) - offset < 16:
                return None
            box_length = int.from_bytes(view[offset + 8 : offset + 16], "big")
            header_length = 16
        elif box_length == 0:
            box_length = len(view) - offset

        if box_length < header_length or box_length > len(view) - offset:
            return None
        box_end = offset + box_length
        if box_type == b"jp2c":
            codestreams.append(view[offset + header_length : box_end])
        offset = box_end

    return tuple(codestreams) or None


def internal_codestream_uses_irreversible_wavelet(view: memoryview) -> bool | None:
    if len(view) < 2 or view[0] != 0xFF or view[1] != 0x4F:
        return None

    offset = 2
    tile_part_end: int | None = None
    component_count: int | None = None
    saw_siz = False
    saw_cod = False
    saw_sot = False
    saw_sod = False
    saw_irreversible = False
    saw_reversible = False
    saw_unknown = False

    while offset < len(view):
        if view[offset] != 0xFF:
            return None
        while offset < len(view) and view[offset] == 0xFF:
            offset += 1
        if offset >= len(view):
            return None
        marker_start = offset - 1
        marker = view[offset]
        offset += 1

        if marker == 0xD9:  # EOC
            return internal_wavelet_result(
                structurally_complete=saw_siz and saw_cod and saw_sot and saw_sod,
                saw_irreversible=saw_irreversible,
                saw_reversible=saw_reversible,
                saw_unknown=saw_unknown,
            )
        if marker == 0x93:  # SOD
            if tile_part_end is None or not saw_sot:
                return None
            saw_sod = True
            if tile_part_end == 0:
                complete = (
                    len(view) >= 2 and view[-2] == 0xFF and view[-1] == 0xD9 and saw_siz and saw_cod
                )
                return internal_wavelet_result(
                    structurally_complete=complete,
                    saw_irreversible=saw_irreversible,
                    saw_reversible=saw_reversible,
                    saw_unknown=saw_unknown,
                )
            if tile_part_end < offset or tile_part_end > len(view):
                return None
            offset = tile_part_end
            tile_part_end = None
            continue
        if marker in {0x01, 0x4F, 0x92}:
            # TEM, a second SOC, and EPH have no segment length.  None belongs
            # in a codestream header, so continuing would risk reading packet
            # bytes as marker segments.
            return None

        if len(view) - offset < 2:
            return None
        segment_length = int.from_bytes(view[offset : offset + 2], "big")
        if segment_length < 2 or segment_length > len(view) - offset:
            return None
        segment_end = offset + segment_length
        body = view[offset + 2 : segment_end]

        if marker == 0x51:  # SIZ
            if saw_siz or saw_sot:
                return None
            component_count = internal_siz_component_count(body)
            if component_count is None:
                return None
            saw_siz = True
        elif marker == 0x52:  # COD
            transform = internal_cod_transform(body)
            if transform == 0:
                saw_irreversible = True
            elif transform == 1:
                saw_reversible = True
            else:
                saw_unknown = True
            if not saw_sot:
                saw_cod = True
        elif marker == 0x53:  # COC
            transform = internal_coc_transform(body, component_count)
            if transform == 0:
                saw_irreversible = True
            elif transform == 1:
                saw_reversible = True
            else:
                saw_unknown = True
        elif marker == 0x90:  # SOT
            if len(body) != 8 or tile_part_end is not None or not saw_siz or not saw_cod:
                return None
            saw_sot = True
            declared_length = int.from_bytes(body[2:6], "big")
            if declared_length == 0:
                tile_part_end = 0
            else:
                tile_part_end = marker_start + declared_length
                if tile_part_end <= segment_end or tile_part_end > len(view):
                    return None

        offset = segment_end

    return None


def internal_wavelet_result(
    *,
    structurally_complete: bool,
    saw_irreversible: bool,
    saw_reversible: bool,
    saw_unknown: bool,
) -> bool | None:
    if not structurally_complete or saw_unknown:
        return None
    if saw_irreversible:
        return True
    if saw_reversible:
        return False
    return None


def internal_siz_component_count(body: memoryview) -> int | None:
    if len(body) < 36:
        return None
    component_count = int.from_bytes(body[34:36], "big")
    if component_count <= 0 or len(body) != 36 + 3 * component_count:
        return None
    return component_count


def internal_cod_transform(body: memoryview) -> int | None:
    if len(body) < 10:
        return None
    decomposition_levels = body[5]
    expected_length = 10 + (decomposition_levels + 1 if body[0] & 1 else 0)
    if len(body) != expected_length:
        return None
    return body[9]


def internal_coc_transform(body: memoryview, component_count: int | None) -> int | None:
    if component_count is None:
        return None
    component_bytes = 1 if component_count < 257 else 2
    fixed_length = component_bytes + 6
    if len(body) < fixed_length:
        return None
    component = int.from_bytes(body[:component_bytes], "big")
    if component >= component_count:
        return None
    coding_style = body[component_bytes]
    decomposition_levels = body[component_bytes + 1]
    expected_length = fixed_length + (decomposition_levels + 1 if coding_style & 1 else 0)
    if len(body) != expected_length:
        return None
    return body[component_bytes + 5]
