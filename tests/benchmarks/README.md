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
happen in CI or on a Linux profiling box rather than on a laptop.

## In CI

`.github/workflows/codspeed.yml` runs the suite in simulation mode on every
pull request and on pushes to `main`, which is what gives CodSpeed a baseline
to compare a pull request against.

Two details there are easy to get wrong:

- The job checks out only the fixture submodules `corpus.py` draws from, via
  `.github/actions/cache-submodules`. The full set is well over a gigabyte.
  Adding a sample from a submodule that is not in that list means adding it to
  the workflow too, or the benchmark will not have its file.
- It sets `CORE_PDF_BENCHMARK_REQUIRE_FIXTURES=1`. Without it a submodule that
  failed to check out would skip every benchmark and the job would pass having
  measured nothing.

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

## What a recorded run costs

Measured on the profiling VM: these 12 benchmarks take **about 16 minutes of
wall clock** under simulation, for roughly 3.7 s of native work. Two rules of
thumb, both from that run:

- Wall clock lands near **25x the sum of the reported values**, which are
  modelled times rather than measured ones.
- Every benchmark pays valgrind's startup, so the suite total is dominated by
  per-benchmark fixed cost rather than by any single sample.

A per-sample multiplier is the wrong way to budget this. An earlier version of
this file quoted "roughly 45x native", taken from timing one file end to end;
it understates a suite badly enough to be misleading, so prefer the wall-clock
figure above.

## Adding a sample

Keep the default suite cheap: prefer a file that isolates one kind of work, and
keep its extraction cost under roughly 500 ms natively. Anything heavier
belongs in `DEEP_DIVE`, which is documentation for one-off profiling rather
than part of the suite. Record the sample's measured `native_ms` so the budget
stays reviewable.

`native_ms` is extraction cost with the document already open, so benchmarks
must keep `PdfDocument(...)` outside the measured region or the two stop being
comparable. Opening is its own axis, measured by `test_open_document`.
