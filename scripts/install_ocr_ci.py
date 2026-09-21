from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def install_model(destination: Path, version: str, pins: dict[str, str]) -> None:
    if version.splitlines()[0] != pins["version"]:
        raise ValueError(f"linked OCR engine mismatch: {version}")
    with urllib.request.urlopen(pins["model_url"], timeout=60) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != pins["model_sha256"]:
        raise ValueError(f"OCR model SHA256 mismatch: {digest}")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "eng.traineddata").write_bytes(data)


def main() -> None:
    import tesserocr

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    manifest = (
        Path(__file__).resolve().parents[1] / "packages/core-pdf-ocr/tests/fixtures/engine.json"
    )
    install_model(args.destination, tesserocr.tesseract_version(), json.loads(manifest.read_text()))


if __name__ == "__main__":
    main()
