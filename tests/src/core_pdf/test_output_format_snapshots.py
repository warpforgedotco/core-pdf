import pytest

from tests.src.core_pdf.output_format_snapshot import (
    FIXTURES,
    OUTPUT_FORMAT_FIXTURES,
    SYNTHETIC,
    output_format_digests,
    recorded,
)

RECORDED = recorded()


@pytest.mark.parametrize("fixture", OUTPUT_FORMAT_FIXTURES)
def test_every_output_format_matches_snapshot(fixture: str) -> None:
    path = FIXTURES / fixture
    if fixture != SYNTHETIC and (not path.is_file() or path.stat().st_size == 0):
        pytest.skip("fixture corpus not initialized")
    assert output_format_digests(fixture) == RECORDED[fixture]
