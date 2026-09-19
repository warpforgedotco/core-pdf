# Authored OCR fixtures

These fixtures contain synthetic invoice text, rendered with the repository's
Liberation Sans font. They contain no customer documents or reference-corpus material.

- `invoice.png`: three grayscale text lines, used for cropped recognition and cancellation.
- `invoice.pdf`: the same pixels embedded losslessly in an image-only PDF.
- `rotated-image.pdf`: the image pixels rotated 90 degrees and placed with a compensating
  PDF transform. It must produce the same upright text through public extraction.

Regenerate from the repository root with the installed test environment:

```sh
.venv/bin/python packages/core-pdf-ocr/tests/fixtures/generate.py
```

The generator used Pillow 12.3.0 and the bundled `LiberationSans-Regular.ttf`.
PDF streams and cross-reference offsets are written directly, without timestamps.
The checked-in files are the test inputs; tests do not regenerate them.

## Pinned real-engine tests

`test_real_engine.py` skips unless `CORE_PDF_TESSERACT_TESTS=1`. Once enabled,
missing dependencies, recognition failures, and pin mismatches fail the suite.
It requires:

- **Tesseract 5.5.1 as linked by tesserocr**, not merely a matching command-line executable.
- English data from [tessdata_fast 4.1.0](https://github.com/tesseract-ocr/tessdata_fast/blob/4.1.0/eng.traineddata),
  SHA-256 `7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2`.
- The repository's locked Python environment, including tesserocr.

Check the linked version with:

```sh
.venv/bin/python -c 'import tesserocr; print(tesserocr.tesseract_version())'
```

Use a tesserocr wheel linked to 5.5.1, or build the binding against that Tesseract
release. This is an integration-test pin; the companion's supported dependency
range is unchanged. A separately installed `tesseract --version` may differ.

Place the pinned English model in a directory and point `TESSDATA_PREFIX` to it
if the default model discovery does not already resolve that model. Then run:

```sh
CORE_PDF_TESSERACT_TESTS=1 \
  uv run --locked --all-packages --extra unstructured --group test --group vendor-test \
  pytest packages/core-pdf-ocr/tests/test_real_engine.py -q
```

The suite exercises public OCR extraction, rotated image placement, cropped text
and page coordinates, and cancellation after real recognition. A delegating API
wrapper counts `End()` calls, verifying each real engine is released exactly once.
Deterministic engine-failure tests remain in the ordinary suite. These small fixtures
are regression checks, not an OCR accuracy benchmark for arbitrary documents.
