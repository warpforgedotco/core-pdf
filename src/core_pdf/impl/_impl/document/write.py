"""Emit a complete PDF revision from native objects."""

from __future__ import annotations

from collections.abc import Mapping

from core_pdf.impl.spec.s_07_syntax.serialize import serialize_object


def write_pdf(
    objects: Mapping[int, tuple[int, object]],
    trailer: Mapping[object, object],
    *,
    size: int,
    version: str = "1.7",
) -> bytes:
    if "Encrypt" in trailer:
        raise NotImplementedError("encrypted document serialization is not implemented")
    output = bytearray(f"%PDF-{version}\n".encode("ascii") + b"%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    for number, (generation, obj) in sorted(objects.items()):
        offsets[number] = len(output)
        output.extend(f"{number} {generation} obj\n".encode("ascii"))
        output.extend(serialize_object(obj))
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {size}\n".encode("ascii"))
    free = [number for number in range(size) if number not in offsets]
    next_free = dict(zip(free, [*free[1:], 0]))
    for number in range(size):
        if number in offsets:
            output.extend(f"{offsets[number]:010d} {objects[number][0]:05d} n \n".encode("ascii"))
        else:
            output.extend(
                f"{next_free[number]:010d} {65535 if number == 0 else 0:05d} f \n".encode("ascii")
            )
    final_trailer = {
        k: v
        for k, v in trailer.items()
        if str(k)
        not in {"Size", "Prev", "XRefStm", "Type", "W", "Index", "Length", "Filter", "DecodeParms"}
    }
    final_trailer["Size"] = size
    output.extend(b"trailer\n" + serialize_object(final_trailer))
    output.extend(f"\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii"))
    return bytes(output)


__all__ = ("write_pdf",)
