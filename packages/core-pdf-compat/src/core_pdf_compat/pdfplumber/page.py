CP936_FONTNAMES = {
    b"\xcb\xce\xcc\xe5": "SimSun,Regular",
    b"\xba\xda\xcc\xe5": "SimHei,Regular",
    b"\xbf\xac\xcc\xe5_GB2312": "SimKai,Regular",
    b"\xb7\xc2\xcb\xce_GB2312": "SimFang,Regular",
    b"\xc1\xa5\xca\xe9": "SimLi,Regular",
}


def fix_fontname_bytes(fontname: bytes) -> str:
    split_at = fontname.index(b"+") + 1 if b"+" in fontname else 0
    prefix, suffix = fontname[:split_at], fontname[split_at:]
    return str(prefix)[2:-1] + CP936_FONTNAMES.get(suffix, str(suffix)[2:-1])


__all__ = ("CP936_FONTNAMES", "fix_fontname_bytes")
