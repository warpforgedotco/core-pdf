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

Four axes are covered:

- `test_open_document` -- xref parsing, trailer and page tree, no content.
- `test_extract_first_page` -- one page, across the shape spread.
- `test_extract_page_slice` -- three pages, for costs that only appear across
  page boundaries, such as shared font and resource caches.
- `test_compose_page` / `test_rasterize_page` -- rendering, which is a sibling
  of extraction rather than a stage of it. Both consume the captured page
  program and only rendering produces pixels, so a change moves one and not
  the other: the glyph caches moved extraction alone, the rasterizer kernels
  moved rendering alone. Until these existed, rendering had no recorded
  measurement at all and its numbers came from laptop wall clock.

## What a recorded run costs

Measured from the CodSpeed job itself, which is the number that matters:

| suite | benchmarks | job wall clock |
| ----- | ---------: | -------------: |
| extraction only | 12 | 7.2 min |
| plus rendering  | 18 | 10.4 min |

Budget from that table, not from the modelled values the report prints. Those
are instruction-derived and do not scale to wall clock: the six rendering
benchmarks report 8.2 s between them -- `test_rasterize_page[tables-…]` alone
reports 4.7 s, the most expensive entry in the suite -- yet cost 3.2 minutes of
job time. Rasterizing is instruction-dense rather than call-dense, so it
inflates the modelled figure far more than the clock.

Two earlier attempts at a rule of thumb here were both wrong and are worth
recording as such: "roughly 45x native", taken from timing one file end to
end, and "25x the sum of the reported values". Neither survives contact with
the table above. What does hold is that every benchmark pays valgrind's
startup, so the count is the thing to economise on.

## Adding a sample

Keep the default suite cheap: prefer a file that isolates one kind of work, and
keep its extraction cost under roughly 500 ms natively. That heuristic is about
extraction; it does not transfer to rendering, where a 274 ms native rasterize
reports 4.7 s modelled. For a rendering sample, check the reported value after
its first recorded run rather than trusting the native figure. Anything heavier
belongs in `DEEP_DIVE`, which is documentation for one-off profiling rather
than part of the suite. Record the sample's measured `native_ms` so the budget
stays reviewable.

`native_ms` is extraction cost with the document already open, so benchmarks
must keep `PdfDocument(...)` outside the measured region or the two stop being
comparable. Opening is its own axis, measured by `test_open_document`. For
`RENDER_SAMPLES` the figure is the rasterize cost with the page already
composed, for the same reason: capture and compose are measured by their own
benchmarks and stay outside the region.
