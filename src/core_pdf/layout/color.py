"""Stream color space conversion helpers."""

from __future__ import annotations

import typing
from typing import Any

from core_pdf.syntax.primitives import PdfStream, normalize_pdf_name, parse_int, parse_name

if typing.TYPE_CHECKING:
    from collections.abc import Callable


def linear_to_srgb(c: float) -> float:
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055


def xyz_to_srgb(x: float, y: float, z: float) -> tuple[float, float, float]:
    rl = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    gl = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    return linear_to_srgb(rl), linear_to_srgb(gl), linear_to_srgb(bl)


def lab_to_xyz(
    l_star: float, a: float, b: float, wp: list[float] | tuple[float, float, float]
) -> tuple[float, float, float]:
    fy = (l_star + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    xr = fx**3 if fx**3 > eps else (116.0 * fx - 16.0) / kappa
    yr = ((l_star + 16.0) / 116.0) ** 3 if l_star > kappa * eps else l_star / kappa
    zr = fz**3 if fz**3 > eps else (116.0 * fz - 16.0) / kappa
    return xr * wp[0], yr * wp[1], zr * wp[2]


def cs_param(params: Any, key: str, default: Any = None) -> Any:
    if isinstance(params, dict):
        return params.get(key, default)
    return default


def cs_param_floats(params: Any, key: str, count: int, default: list[float]) -> list[float]:
    raw = cs_param(params, key, default)
    if isinstance(raw, (list, tuple)) and len(raw) >= count:
        return [float(v) for v in raw[:count]]
    return default


def cs_name(value: Any, default: str | None = None) -> str | None:
    return parse_name(value, default)


def icc_profile_alt_name(profile: bytes | None, channels: int) -> str | None:
    if profile is None or len(profile) < 128:
        return {1: "DeviceGray", 3: "DeviceRGB", 4: "DeviceCMYK"}.get(channels)
    pcs = profile[20:24]
    if pcs == b"XYZ ":
        return {1: "DeviceGray", 3: "DeviceRGB", 4: "DeviceCMYK"}.get(channels)
    color_space = profile[16:20]
    if color_space == b"GRAY":
        return "DeviceGray"
    if color_space == b"RGB ":
        return "DeviceRGB"
    if color_space == b"CMYK":
        return "DeviceCMYK"
    return {1: "DeviceGray", 3: "DeviceRGB", 4: "DeviceCMYK"}.get(channels)


class ImageColorSpec:
    __slots__ = (
        "kind",
        "params",
        "bits_per_component",
        "base",
        "hival",
        "lookup",
        "alt",
        "tint_fn",
        "names",
        "channels",
    )

    kind: str | None
    params: dict[str, Any]
    bits_per_component: int
    base: str | None
    hival: int
    lookup: bytes | None
    alt: str | None
    tint_fn: Any
    names: list[Any] | tuple[Any, ...] | None
    channels: int

    def __init__(
        self,
        kind: str | None,
        params: dict[str, Any],
        bits_per_component: int = 8,
        base: str | None = None,
        hival: int = 0,
        lookup: bytes | None = None,
        alt: str | None = None,
        tint_fn: Any = None,
        names: list[Any] | tuple[Any, ...] | None = None,
        channels: int = 1,
    ) -> None:
        self.kind = kind
        self.params = params
        self.bits_per_component = bits_per_component
        self.base = base
        self.hival = hival
        self.lookup = lookup
        self.alt = alt
        self.tint_fn = tint_fn
        self.names = names
        self.channels = channels


def normalize_color_space_name(value: Any) -> str | None:
    result = normalize_pdf_name(value)
    if result is not None:
        return result
    if value is None:
        return None
    text = str(value)
    return text[1:] if text.startswith("/") else text


def normalize_image_color_spec(image_dict: dict[str, Any]) -> ImageColorSpec:
    bits_per_component = parse_int(image_dict.get("BitsPerComponent", 8), 8) or 8
    color_space = image_dict.get("ColorSpace")
    if isinstance(color_space, (list, tuple)) and color_space:
        kind = normalize_color_space_name(color_space[0])
        if kind == "Indexed" and len(color_space) >= 4:
            lookup = color_space[3]
            lookup_bytes = (
                lookup.data
                if isinstance(lookup, PdfStream)
                else (lookup if isinstance(lookup, bytes) else None)
            )
            return ImageColorSpec(
                kind="Indexed",
                params={},
                bits_per_component=bits_per_component,
                base=normalize_color_space_name(color_space[1]),
                hival=parse_int(color_space[2], 0) or 0,
                lookup=lookup_bytes,
            )
        if kind == "Indexed":
            raise ValueError("invalid Indexed color space")
        if (
            kind == "ICCBased"
            and len(color_space) >= 2
            and isinstance(color_space[1], (dict, PdfStream))
        ):
            icc_stream = color_space[1]
            icc_dict = icc_stream.dictionary if isinstance(icc_stream, PdfStream) else icc_stream
            alt = normalize_color_space_name(icc_dict.get("Alternate"))
            n = icc_dict.get("N", 3)
            channels = parse_int(n, 3) or 3
            if alt is None and isinstance(icc_stream, PdfStream):
                alt = icc_profile_alt_name(icc_stream.data, channels)
            return ImageColorSpec(
                kind="ICCBased",
                params=icc_dict,
                bits_per_component=bits_per_component,
                alt=alt,
                channels=channels,
            )
        if kind == "ICCBased":
            raise ValueError("invalid ICCBased color space")
        if (
            kind in {"Lab", "CalGray", "CalRGB"}
            and len(color_space) >= 2
            and isinstance(color_space[1], dict)
        ):
            return ImageColorSpec(
                kind=kind, params=color_space[1], bits_per_component=bits_per_component
            )
        if kind in {"Lab", "CalGray", "CalRGB"}:
            raise ValueError(f"invalid {kind} color space")
        if kind in {"Separation", "DeviceN"}:
            if len(color_space) < 4:
                raise ValueError(f"invalid {kind} color space")
            names = color_space[1] if isinstance(color_space[1], (list, tuple)) else None
            if kind == "DeviceN" and names is None:
                raise ValueError("invalid DeviceN color space")
            alt = normalize_color_space_name(color_space[2])
            return ImageColorSpec(
                kind=kind,
                params={},
                bits_per_component=bits_per_component,
                alt=alt,
                tint_fn=color_space[3] if len(color_space) >= 4 else None,
                names=list(names)
                if kind == "DeviceN" and isinstance(names, (list, tuple))
                else None,
                channels=len(names)
                if kind == "DeviceN" and isinstance(names, (list, tuple))
                else 1,
            )
        if kind in {"Separation", "DeviceN"}:
            raise ValueError(f"invalid {kind} color space")
        return ImageColorSpec(kind=kind, params={}, bits_per_component=bits_per_component)
    return ImageColorSpec(
        kind=normalize_color_space_name(color_space),
        params={},
        bits_per_component=bits_per_component,
    )


def adapt_d50_to_d65(x: float, y: float, z: float) -> tuple[float, float, float]:
    ax = 0.955473 * x - 0.023098 * y + 0.063259 * z
    ay = -0.028369 * x + 1.009995 * y + 0.021300 * z
    az = 0.012314 * x - 0.020507 * y + 1.330365 * z
    return ax, ay, az


class ImageColorManager:
    @staticmethod
    def convert_image_data(raw: bytes, image_dict: dict[str, Any] | ImageColorSpec) -> bytes | None:
        current: dict[str, Any] | ImageColorSpec = image_dict
        depth = 0

        while depth <= 3:
            spec = (
                current
                if isinstance(current, ImageColorSpec)
                else normalize_image_color_spec(current)
            )
            if spec.bits_per_component != 8:
                return None
            cs_kind = spec.kind
            if cs_kind is None:
                return None

            if cs_kind == "DeviceRGB":
                return raw
            if cs_kind == "DeviceGray":
                return ImageColorManager.convert_gray(raw)
            if cs_kind == "DeviceCMYK":
                return ImageColorManager.convert_cmyk(raw)
            if cs_kind == "Lab":
                return ImageColorManager.convert_lab_raw(raw, spec.params)
            if cs_kind == "CalGray":
                return ImageColorManager.convert_calgray(raw, spec.params)
            if cs_kind == "CalRGB":
                return ImageColorManager.convert_calrgb(raw, spec.params)
            if cs_kind == "Indexed":
                return ImageColorManager.convert_indexed(raw, spec)
            if cs_kind == "ICCBased":
                if spec.alt is not None:
                    current = ImageColorSpec(kind=spec.alt, params={})
                    depth += 1
                    continue
                if spec.channels == 1:
                    return ImageColorManager.convert_gray(raw)
                if spec.channels == 3:
                    current = ImageColorSpec(kind="DeviceRGB", params={})
                    depth += 1
                    continue
                if spec.channels == 4:
                    current = ImageColorSpec(kind="DeviceCMYK", params={})
                    depth += 1
                    continue
                raise ValueError("invalid ICCBased color space")
            if cs_kind == "Separation":
                return ImageColorManager.convert_separation(raw, spec)
            if cs_kind == "DeviceN":
                return ImageColorManager.convert_devicen(raw, spec)
            return None
        return None

    @staticmethod
    def convert_separation(
        raw: bytes, color_space: ImageColorSpec | list[Any] | tuple[Any, ...]
    ) -> bytes | None:
        if isinstance(color_space, ImageColorSpec):
            alt_name = color_space.alt or "DeviceGray"
            tint_fn = color_space.tint_fn
        else:
            if len(color_space) < 4:
                raise ValueError("invalid Separation color space")
            alt_cs = color_space[2]
            if not isinstance(alt_cs, dict):
                raise ValueError("invalid Separation color space")
            alt_name = cs_name(alt_cs.get("ColorSpace"), "DeviceGray") or "DeviceGray"
            tint_fn = color_space[3] if len(color_space) > 3 else None

        result = bytearray()
        for byte in raw:
            v = byte / 255.0
            if tint_fn is None:
                components: list[Any] = [v]
            elif isinstance(tint_fn, (list, tuple)):
                if len(tint_fn) >= 1 and callable(tint_fn[0]):
                    try:
                        components = tint_fn[0](v)
                    except Exception as exc:
                        raise ValueError("invalid separation tint function") from exc
                else:
                    components = list(tint_fn)
            else:
                components = [v]
            expected = (
                1
                if alt_name == "DeviceGray"
                else 3
                if alt_name == "DeviceRGB"
                else 4
                if alt_name == "DeviceCMYK"
                else None
            )
            if expected is None:
                raise ValueError("invalid Separation color space")
            if len(components) != expected:
                raise ValueError("invalid separation tint function")
            rgb = ImageColorManager.apply_alt_color(components, alt_name)
            if rgb is not None:
                result.extend(rgb)
        return bytes(result)

    @staticmethod
    def convert_devicen(
        raw: bytes, color_space: ImageColorSpec | list[Any] | tuple[Any, ...]
    ) -> bytes | None:
        if isinstance(color_space, ImageColorSpec):
            alt_name = color_space.alt or ""
            n = color_space.channels
            tint_fn = color_space.tint_fn
        else:
            if len(color_space) < 4:
                raise ValueError("invalid DeviceN color space")
            names = color_space[1]
            alt_cs_name = color_space[2]
            alt_name = cs_name(alt_cs_name, "") or ""
            n = len(names) if isinstance(names, (list, tuple)) else 1
            tint_fn = color_space[3] if len(color_space) > 3 else None
        if n <= 0:
            raise ValueError("invalid DeviceN color space")
        if len(raw) % n != 0:
            raise ValueError("invalid DeviceN color sample data")

        result = bytearray()
        step = n
        for i in range(0, len(raw), step):
            components: list[Any] = [raw[i + j] / 255.0 for j in range(step)]
            if isinstance(tint_fn, (list, tuple)) and len(tint_fn) >= 1 and callable(tint_fn[0]):
                try:
                    components = tint_fn[0](*components)
                except Exception as exc:
                    raise ValueError("invalid DeviceN tint function") from exc
            expected = (
                1
                if alt_name == "DeviceGray"
                else 3
                if alt_name == "DeviceRGB"
                else 4
                if alt_name == "DeviceCMYK"
                else None
            )
            if expected is None:
                raise ValueError("invalid DeviceN color space")
            if len(components) != expected:
                raise ValueError("invalid DeviceN tint function")
            rgb = ImageColorManager.apply_alt_color(components, alt_name)
            if rgb is not None:
                result.extend(rgb)
        return bytes(result)

    @staticmethod
    def convert_gray(raw: bytes) -> bytes:
        result = bytearray(len(raw) * 3)
        for i, byte in enumerate(raw):
            idx = i * 3
            result[idx] = result[idx + 1] = result[idx + 2] = byte
        return bytes(result)

    @staticmethod
    def convert_cmyk(raw: bytes) -> bytes:
        result = bytearray()
        n = len(raw)
        inv255 = 1.0 / 255.0
        if n % 4 != 0:
            raise ValueError("invalid color sample data")
        for i in range(0, n, 4):
            c, m, y, k = raw[i], raw[i + 1], raw[i + 2], raw[i + 3]
            r = int(255 * (1 - c * inv255) * (1 - k * inv255))
            g = int(255 * (1 - m * inv255) * (1 - k * inv255))
            b = int(255 * (1 - y * inv255) * (1 - k * inv255))
            result.extend([max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))])
        return bytes(result)

    @staticmethod
    def convert_indexed(raw: bytes, spec: ImageColorSpec) -> bytes | None:
        if spec.lookup is None:
            return None
        result = bytearray()
        lookup = spec.lookup
        hival = spec.hival
        if spec.base == "DeviceRGB":
            if len(lookup) < (hival + 1) * 3:
                raise ValueError("invalid Indexed color lookup")
            for byte in raw:
                if byte > hival:
                    raise ValueError("invalid Indexed color sample")
                start = byte * 3
                result.extend(lookup[start : start + 3])
        elif spec.base == "DeviceGray":
            if len(lookup) < hival + 1:
                raise ValueError("invalid Indexed color lookup")
            for byte in raw:
                if byte > hival:
                    raise ValueError("invalid Indexed color sample")
                val = lookup[byte]
                result.extend([val, val, val])
        else:
            raise ValueError("invalid Indexed color space")
        return bytes(result)

    @staticmethod
    def convert_to_rgb(
        raw: bytes, fn: Callable[..., tuple[float, float, float]], channels: int = 1
    ) -> bytes:
        if channels <= 0:
            raise ValueError("invalid color channel count")
        if len(raw) % channels != 0:
            raise ValueError("invalid color sample data")
        result = bytearray(len(raw) * 3)
        inv255 = 1.0 / 255.0
        for i in range(0, len(raw), channels):
            if channels == 1:
                r, g, b = fn(raw[i] * inv255)
            elif i + channels - 1 < len(raw):
                r, g, b = fn(*(raw[i + j] * inv255 for j in range(channels)))
            else:
                raise ValueError("invalid color sample data")
            result[i * 3] = max(0, min(255, int(r * 255.0)))
            result[i * 3 + 1] = max(0, min(255, int(g * 255.0)))
            result[i * 3 + 2] = max(0, min(255, int(b * 255.0)))
        return bytes(result)

    @staticmethod
    def convert_calgray(raw: bytes, params: Any) -> bytes:
        wp = cs_param_floats(params, "WhitePoint", 3, [0.9505, 1.0, 1.089])
        bp = cs_param_floats(params, "BlackPoint", 3, [0.0, 0.0, 0.0])
        gamma = float(cs_param(params, "Gamma", 1.0))

        lut = bytearray(768)
        inv255 = 1.0 / 255.0

        for i in range(256):
            v = i * inv255
            vg = pow(v, gamma) if gamma != 1.0 else v
            x = bp[0] + vg * (wp[0] - bp[0])
            y = bp[1] + vg * (wp[1] - bp[1])
            z = bp[2] + vg * (wp[2] - bp[2])
            ax, ay, az = adapt_d50_to_d65(x, y, z)
            r, g, b = xyz_to_srgb(ax, ay, az)
            lut[i * 3] = max(0, min(255, int(r * 255.0)))
            lut[i * 3 + 1] = max(0, min(255, int(g * 255.0)))
            lut[i * 3 + 2] = max(0, min(255, int(b * 255.0)))

        result = bytearray(len(raw) * 3)
        data_getitem = raw.__getitem__
        result_setitem = result.__setitem__
        for idx in range(len(raw)):
            off = data_getitem(idx) * 3
            result_setitem(idx * 3, lut[off])
            result_setitem(idx * 3 + 1, lut[off + 1])
            result_setitem(idx * 3 + 2, lut[off + 2])
        return bytes(result)

    @staticmethod
    def convert_calrgb(raw: bytes, params: Any) -> bytes:
        bp = cs_param_floats(params, "BlackPoint", 3, [0.0, 0.0, 0.0])
        gamma = cs_param_floats(params, "Gamma", 3, [1.0, 1.0, 1.0])
        matrix = cs_param_floats(params, "Matrix", 9, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

        def fn(r: float, g: float, b: float) -> tuple[float, float, float]:
            rg = pow(r, gamma[0]) if gamma[0] != 1.0 else r
            gg = pow(g, gamma[1]) if gamma[1] != 1.0 else g
            bg = pow(b, gamma[2]) if gamma[2] != 1.0 else b
            x = bp[0] + matrix[0] * rg + matrix[1] * gg + matrix[2] * bg
            y = bp[1] + matrix[3] * rg + matrix[4] * gg + matrix[5] * bg
            z = bp[2] + matrix[6] * rg + matrix[7] * gg + matrix[8] * bg
            ax, ay, az = adapt_d50_to_d65(x, y, z)
            return xyz_to_srgb(ax, ay, az)

        return ImageColorManager.convert_to_rgb(raw, fn, channels=3)

    @staticmethod
    def convert_lab_raw(raw: bytes, params: Any) -> bytes:
        wp = cs_param_floats(params, "WhitePoint", 3, [0.9505, 1.0, 1.089])
        range_a = cs_param_floats(params, "Range", 2, [-100.0, 100.0])

        def fn(l_byte: float, a_byte: float, b_byte: float) -> tuple[float, float, float]:
            l_star = l_byte * 100.0
            a_star = a_byte * (range_a[1] - range_a[0]) + range_a[0]
            b_star = b_byte * (range_a[1] - range_a[0]) + range_a[0]
            x, y, z = lab_to_xyz(l_star, a_star, b_star, wp)
            ax, ay, az = adapt_d50_to_d65(x, y, z)
            return xyz_to_srgb(ax, ay, az)

        return ImageColorManager.convert_to_rgb(raw, fn, channels=3)

    @staticmethod
    def apply_alt_color(components: list[Any], alt_name: str) -> bytes | None:
        if alt_name == "DeviceGray":
            v = max(0.0, min(1.0, components[0] if components else 0.0))
            v_byte = max(0, min(255, int(round(v * 255.0))))
            return bytes([v_byte, v_byte, v_byte])
        if alt_name == "DeviceRGB":
            r = max(0.0, min(1.0, components[0] if len(components) >= 1 else 0.0))
            g = max(0.0, min(1.0, components[1] if len(components) >= 2 else r))
            b = max(0.0, min(1.0, components[2] if len(components) >= 3 else r))
            return bytes(
                [
                    max(0, min(255, int(round(r * 255.0)))),
                    max(0, min(255, int(round(g * 255.0)))),
                    max(0, min(255, int(round(b * 255.0)))),
                ]
            )
        if alt_name == "DeviceCMYK":
            c = components[0] if len(components) >= 1 else 0.0
            m = components[1] if len(components) >= 2 else 0.0
            y = components[2] if len(components) >= 3 else 0.0
            k = components[3] if len(components) >= 4 else 0.0
            r = int(255 * (1 - c) * (1 - k))
            g_ = int(255 * (1 - m) * (1 - k))
            b_ = int(255 * (1 - y) * (1 - k))
            return bytes([max(0, min(255, r)), max(0, min(255, g_)), max(0, min(255, b_))])
        return None
