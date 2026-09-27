import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from core_pdf import PdfDocument

REPOSITORY = Path(__file__).resolve().parents[3]
FIXTURES = REPOSITORY / "tests" / "fixtures"
SNAPSHOT = Path(__file__).with_name("extraction_snapshots.json")
PAGE_LIMIT = 3


def fixture_paths() -> list[str]:
    return sorted(
        path.relative_to(FIXTURES).as_posix()
        for path in FIXTURES.rglob("*.pdf", recurse_symlinks=True)
        if "specifications" not in path.relative_to(FIXTURES).parts
    )


def canonical(value: Any) -> Any:
    if isinstance(value, float):
        rounded = round(value, 3)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items()}
    if isinstance(value, list):
        return [canonical(item) for item in value]
    return value


def extraction_digest(fixture: str) -> str:
    try:
        with PdfDocument(FIXTURES / fixture) as document:
            count = document.page_count()
            pages = range(1, min(count, PAGE_LIMIT) + 1) if count else None
            payload = json.loads(document.extract(pages=pages).to_json())
    except Exception as error:  # noqa: BLE001
        return f"error:{type(error).__name__}"
    encoded = json.dumps(canonical(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def recorded() -> dict[str, str]:
    return json.loads(SNAPSHOT.read_text()) if SNAPSHOT.exists() else {}


def update() -> None:
    fixtures = [
        fixture
        for fixture in fixture_paths()
        if (FIXTURES / fixture).is_file() and (FIXTURES / fixture).stat().st_size
    ]
    with ProcessPoolExecutor() as pool:
        digests = dict(
            zip(fixtures, pool.map(extraction_digest, fixtures, chunksize=4), strict=True)
        )
    SNAPSHOT.write_text(json.dumps(digests, indent=1, sort_keys=True) + "\n")
    print(f"recorded {len(digests)} extraction snapshots", file=sys.stderr)


if __name__ == "__main__":
    update()
