import pytest

from core_pdf.impl.pdf_names import lenient_float, lenient_int
from core_pdf.impl.recovery_policy import (
    LENIENT,
    STRICT,
    malformed_policy,
    recovery_policy,
)


def fail(value: str) -> str:
    raise ValueError(value)


def repair(value: str) -> str:
    return f"repaired {value}"


def test_strict_recovery_raises_what_lenient_recovery_skips() -> None:
    with pytest.raises(ValueError, match="bad"):
        STRICT.malformed("bad")
    assert LENIENT.malformed("bad") is None
    with pytest.raises(ValueError, match="bad"):
        malformed_policy(False)("bad")
    assert malformed_policy(True)("bad") is None


def test_attempt_repairs_only_when_lenient_and_only_the_named_errors() -> None:
    assert LENIENT.attempt(fail, repair, "x") == "repaired x"
    with pytest.raises(ValueError, match="x"):
        STRICT.attempt(fail, repair, "x")
    with pytest.raises(ValueError, match="x"):
        LENIENT.attempt(fail, repair, "x", errors=KeyError)


def test_reject_returns_the_fallback_or_raises_the_error() -> None:
    error = ValueError("bad")
    assert LENIENT.reject(error, "context", 7) == 7
    with pytest.raises(ValueError, match="bad") as raised:
        STRICT.reject(error, "context", 7)
    assert raised.value is error


def test_the_policy_accepts_a_flag_or_a_recovery() -> None:
    assert recovery_policy(True) is LENIENT
    assert recovery_policy(False) is STRICT
    assert recovery_policy(STRICT) is STRICT


def test_lenient_numbers_accept_python_syntax() -> None:
    assert lenient_int("1_000") == 1000
    assert lenient_int("x") is None
    assert lenient_int("x", 3) == 3
    assert lenient_float(b"1e3") == 1000.0
    assert lenient_float("x", None) is None
    assert lenient_float("x") == 0.0
