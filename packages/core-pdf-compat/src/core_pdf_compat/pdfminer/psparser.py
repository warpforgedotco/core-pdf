class PSObject:
    pass


class PSLiteral(PSObject):
    def __init__(self, name: str | bytes) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"/{self.name!r}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PSLiteral) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


class PSKeyword(PSObject):
    def __init__(self, name: bytes) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"/{self.name!r}"


def LIT(name: str) -> PSLiteral:  # noqa: N802
    return PSLiteral(name)


def KWD(name: bytes) -> PSKeyword:  # noqa: N802
    return PSKeyword(name)


__all__ = ("KWD", "LIT", "PSKeyword", "PSLiteral")
