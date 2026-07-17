# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest

from core_pdf.impl.engine.spec.s_07_content.operations import content_stream_may_show_text


@pytest.mark.parametrize(
    "data",
    [
        b"(Tj TJ Do ' \\\" (nested Tj))",
        b"<546a 544a 446f>",
        b"% Tj TJ Do ' \\\"\nq Q",
        b"/Tj /TJ /Do /' /\\\" gs",
        b"prefixTjsuffix prefixDosuffix",
        b"BI /F /A85 ID Tj TJ Do ' \\\"~>\nEI Q",
    ],
)
def test_content_stream_may_show_text_ignores_operand_and_image_bytes(data: bytes) -> None:
    assert not content_stream_may_show_text(data)


@pytest.mark.parametrize(
    "data",
    [
        b"(hello) Tj",
        b"[(hello)] TJ",
        b"/XObject Do",
        b"() '",
        b'0 0 () "',
    ],
)
def test_content_stream_may_show_text_accepts_delimited_operators(data: bytes) -> None:
    assert content_stream_may_show_text(data)


def test_content_stream_may_show_text_supports_sliced_memoryview() -> None:
    content = b"(embedded Tj) q Q"
    data = memoryview(b"prefix" + content + b"suffix")[len(b"prefix") : -len(b"suffix")]

    assert not content_stream_may_show_text(data)
