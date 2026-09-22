# Benchmarks

Performance benchmarks for the native extraction pipeline, measured through
[CodSpeed](https://codspeed.io). `tests/benchmarks` is deliberately absent from
`testpaths` in `pyproject.toml`, so a bare `pytest` run never collects them;
name the directory explicitly to run them.

```sh
# plain timings, no CodSpeed runner needed
uv run --group benchmark pytest tests/benchmarks --codspeed

# recorded run, deterministic instruction counts and flamegraphs
codspeed run -m simulation -- uv run --group benchmark pytest tests/benchmarks --codspeed
```

Simulation mode needs Linux and CodSpeed's patched valgrind, so recorded runs
happen on the profiling VM rather than on a laptop.

## What is measured, and why these files

`corpus.py` holds the selection and the measurement behind each entry. It came
from sweeping all 795 fixture PDFs, so the samples span the shapes that actually
drive cost -- text density, tables, figures, sparse pages -- rather than the
largest files. The module docstring has the details.

Three axes are covered:

- `test_open_document` -- xref parsing, trailer and page tree, no content.
- `test_extract_first_page` -- one page, across the shape spread.
- `test_extract_page_slice` -- three pages, for costs that only appear across
  page boundaries, such as shared font and resource caches.

## Adding a sample

Keep the default suite cheap. Callgrind runs roughly 45x slower than native, so
a 500 ms sample costs about 22 s per recorded run. Anything heavier belongs in
`DEEP_DIVE`, which is documentation for one-off profiling rather than part of
the suite. Record the sample's measured `native_ms` so the budget stays
reviewable, and prefer a file that isolates one kind of work.
