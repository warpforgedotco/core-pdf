import pytest

from tests.src.core_pdf.extraction_snapshot import (
    FIXTURES,
    extraction_digest,
    fixture_paths,
    recorded,
)

RECORDED = recorded()


@pytest.mark.parametrize("fixture", fixture_paths())
def test_extraction_matches_snapshot(fixture: str) -> None:
    path = FIXTURES / fixture
    if not path.is_file() or path.stat().st_size == 0:
        pytest.skip("fixture corpus not initialized")
    if fixture not in RECORDED:
        pytest.fail(
            f"no recorded snapshot for {fixture}; "
            "run `uv run python -m tests.src.core_pdf.extraction_snapshot`"
        )
    assert extraction_digest(fixture) == RECORDED[fixture]
