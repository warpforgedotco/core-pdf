"""Time each core_pdf_compat facade against the library it stands for, on the upstream PDFs.

    uv run --all-packages python scripts/benchmark_facades.py
    uv run --all-packages python scripts/benchmark_facades.py --library unstructured \
        --files DA-619p.pdf --repeats 1

The files and the calls are the comparison scripts' (scripts/check_*_output.py and
check_pypdf_*.py), and the real libraries are the versions those scripts pin, read from the
suites in scripts/run_upstream_suite.py. Each library runs in two child processes started
with `uv run --with <pins>`, one importing only the real library and one only the facade,
so neither side's imports or caches touch the other's, and they run one after the other.

Every call runs --repeats times (default 3) per file and the median counts. Import time is
measured once per process and reported apart. Totals are over the files both sides finished;
a file a side takes over --timeout seconds on, or crashes on, is left out of both and
counted. A call that raises is timed like one that returns, since the facades raise what the
libraries raise.

pypdf: text and layout are page.extract_text() in both modes over the first 30 pages (a fresh
    reader each); incremental, clone and append write the file as check_pypdf_bytes.py does;
    reads reads every object as check_pypdf_reads.py does, unencrypted.
pdfminer: extract_text and extract_pages (the tree walked) over the first 5 pages.
pdfplumber: open, then per page of the first 30 the checks of check_pdfplumber_output.py in
    its order (pages parse lazily, so a page's parsing is timed under chars), then close;
    "all" is the whole.
unstructured: partition_pdf(strategy="fast"), every element's text, id and metadata read;
    files over 30 pages are skipped, as the check does, unless named with --files.
llamaindex: PDFReader().load_data in the check's three modes.
xray: inspect() given a path and given the bytes.

--library and --operation pick cases, --files picks files by name or path. --profile DIR
also runs each call once under cProfile and writes DIR/<library>-<side>-<operation>-<file>.prof.
--json PATH keeps every timing.
"""

import argparse
import asyncio
import contextlib
import cProfile
import gc
import glob
import io
import json
import logging
import os
import platform
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_upstream_suite import ROOT, SUITES  # noqa: E402

UPSTREAM = "tests/upstream/**/*.pdf"
DOWNLOADS = "tests/upstream/pypdf/tests/pdf_cache/"
# The patterns of check_pypdf_*.py, check_llamaindex_output.py and check_xray_output.py.
SAMPLES = (
    "tests/upstream/pdfplumber/tests/pdfs/*.pdf",
    "tests/upstream/pdfminer.six/samples/*.pdf",
    "tests/upstream/pdfminer.six/samples/contrib/*.pdf",
    "tests/upstream/pypdf/resources/*.pdf",
    "tests/upstream/pypdf/sample-files/**/*.pdf",
)
XRAY_ASSETS = "tests/upstream/x-ray/tests/assets/*.pdf"
PYPDF_DOWNLOADS = Path("tests/compat/pypdf_test_files.tsv")
PYPDF_PAGES = 30
PDFMINER_PAGES = 5
PDFPLUMBER_PAGES = 30
UNSTRUCTURED_PAGES = 30
PDFPLUMBER_CHECKS = (
    *("chars", "rects", "lines", "curves", "images", "words", "text", "layout", "tables"),
)
WORST = 5
NOISE = 0.001
WARM_UP = "tests/upstream/pdfminer.six/samples/simple1.pdf"


class Phases(dict[str, float]):
    """Seconds per phase of one call, which the report shows as operations of their own."""


Operation = Callable[[str], object]


class Library(NamedTuple):
    suite: str
    operations: tuple[str, ...]
    uv_options: tuple[str, ...] = ()


LIBRARIES = {
    "pypdf": Library("pypdf", ("text", "layout", "incremental", "clone", "append", "reads")),
    "pdfminer": Library("pdfminer", ("extract_text", "extract_pages")),
    "pdfplumber": Library("pdfplumber", ("open", *PDFPLUMBER_CHECKS, "close", "all")),
    "unstructured": Library("unstructured", ("partition_fast",), ("--extra", "unstructured")),
    "llamaindex": Library("llamaindex", ("pages", "full", "extra")),
    "xray": Library("xray", ("path", "bytes")),
}
VERSIONED = {
    "pypdf": ("pypdf",),
    "pdfminer": ("pdfminer.six",),
    "pdfplumber": ("pdfplumber", "pdfminer.six"),
    "unstructured": ("unstructured", "pdfminer.six"),
    "llamaindex": ("llama-index-readers-file", "pypdf"),
    "xray": ("x-ray", "pymupdf"),
}


class Timeout(Exception):
    pass


# ---------------------------------------------------------------------------------------------
# Files


def every_pdf() -> list[str]:
    return sorted(
        path
        for path in glob.glob(UPSTREAM, recursive=True)
        if not path.startswith(DOWNLOADS) and not os.path.islink(path)
    )


def samples(*extra: str) -> list[str]:
    return sorted({path for pattern in (*extra, *SAMPLES) for path in glob.glob(pattern)})


def pypdf_files() -> list[str]:
    """check_pypdf_text.py's files: the samples, and pypdf's downloads when fetched."""
    names = [line.split("\t")[0] for line in PYPDF_DOWNLOADS.read_text().splitlines() if line]
    found = [DOWNLOADS + name for name in names if name.endswith(".pdf")]
    return sorted(set(samples()).union(path for path in found if os.path.exists(path)))


def short_files() -> list[str]:
    from core_pdf import PdfDocument  # noqa: PLC0415

    def pages(path: str) -> int:
        try:
            with PdfDocument.open(path) as document:
                return len(document.pages)
        except Exception:  # noqa: BLE001
            return 0

    return [path for path in every_pdf() if pages(path) <= UNSTRUCTURED_PAGES]


def discover(library: str, wanted: list[str]) -> list[str]:
    if wanted:
        everything = sorted(set(every_pdf()).union(samples(XRAY_ASSETS), pypdf_files()))
        return [
            path
            for path in everything
            if any(path == want or Path(path).name == want for want in wanted)
        ]
    if library == "pypdf":
        return pypdf_files()
    if library in {"pdfminer", "pdfplumber"}:
        return every_pdf()
    if library == "unstructured":
        return short_files()
    if library == "xray":
        return samples(XRAY_ASSETS)
    return samples()


# ---------------------------------------------------------------------------------------------
# The calls, run in the worker process for one side


def pypdf_operations(side: str) -> dict[str, Operation]:
    if side == "real":
        import pypdf as library  # noqa: PLC0415  # ty: ignore[unresolved-import]
    else:
        from core_pdf_compat import pypdf as library  # noqa: PLC0415

    def text(options: dict[str, Any]) -> Operation:
        def run(path: str) -> None:
            reader = library.PdfReader(io.BytesIO(Path(path).read_bytes()))
            if reader.is_encrypted:
                reader.decrypt("")
            for index in range(min(len(reader.pages), PYPDF_PAGES)):
                with contextlib.suppress(Exception):
                    reader.pages[index].extract_text(**options)

        return run

    def write(make: Callable[[bytes], Any]) -> Operation:
        def run(path: str) -> None:
            writer = make(Path(path).read_bytes())
            writer.write(io.BytesIO())

        return run

    def incremental(data: bytes) -> Any:
        writer = library.PdfWriter(io.BytesIO(data), incremental=True)
        writer.add_metadata({"/Title": "x"})
        return writer

    def append(data: bytes) -> Any:
        writer = library.PdfWriter()
        writer.append(library.PdfReader(io.BytesIO(data)))
        return writer

    def reads(path: str) -> None:
        reader = library.PdfReader(io.BytesIO(Path(path).read_bytes()))
        numbers = [(n, g) for g, entries in reader.xref.items() for n in entries]
        numbers += [(n, 0) for n in reader.xref_objStm]
        for number, generation in numbers:
            try:
                value = reader.get_object(
                    library.generic.IndirectObject(number, generation, reader)
                )
                if value is not None:
                    value.write_to_stream(io.BytesIO())
                if isinstance(value, library.generic.StreamObject):
                    value.get_data()
            except Exception:  # noqa: BLE001, S112
                continue

    return {
        "text": text({}),
        "layout": text({"extraction_mode": "layout"}),
        "incremental": write(incremental),
        "clone": write(lambda data: library.PdfWriter(clone_from=io.BytesIO(data))),
        "append": write(append),
        "reads": reads,
    }


def pdfminer_operations(side: str) -> dict[str, Operation]:
    if side == "real":
        from pdfminer import high_level  # noqa: PLC0415  # ty: ignore[unresolved-import]
    else:
        import core_pdf_compat.pdfminer as high_level  # noqa: PLC0415

    def walk(item: Any, seen: list[object]) -> None:
        seen.append(item.get_text() if hasattr(item, "get_text") else getattr(item, "bbox", ()))
        if hasattr(type(item), "__iter__"):
            for child in item:
                walk(child, seen)

    def extract_pages(path: str) -> None:
        for page in high_level.extract_pages(path, maxpages=PDFMINER_PAGES):
            walk(page, [])

    return {
        "extract_text": lambda path: high_level.extract_text(path, maxpages=PDFMINER_PAGES),
        "extract_pages": extract_pages,
    }


def pdfplumber_operations(side: str) -> dict[str, Operation]:
    if side == "real":
        import pdfplumber as library  # noqa: PLC0415  # ty: ignore[unresolved-import]
    else:
        from core_pdf_compat import pdfplumber as library  # noqa: PLC0415

    checks: dict[str, Callable[[Any], object]] = {
        "chars": lambda page: page.chars,
        "rects": lambda page: page.rects,
        "lines": lambda page: page.lines,
        "curves": lambda page: page.curves,
        "images": lambda page: page.images,
        "words": lambda page: page.extract_words(),
        "text": lambda page: page.extract_text(),
        "layout": lambda page: page.extract_text(layout=True),
        "tables": lambda page: page.extract_tables(),
    }

    def phased(path: str) -> Phases:
        phases = Phases(dict.fromkeys(("open", *checks, "close"), 0.0))
        started = time.perf_counter()
        pdf, pages = None, []
        with contextlib.suppress(Exception):
            pdf = library.open(path)
            pages = pdf.pages
        phases["open"] = time.perf_counter() - started
        if pdf is None:
            return phases
        for page in pages[:PDFPLUMBER_PAGES]:
            for name, check in checks.items():
                started = time.perf_counter()
                with contextlib.suppress(Exception):
                    check(page)
                phases[name] += time.perf_counter() - started
            page.close()
        started = time.perf_counter()
        with contextlib.suppress(Exception):
            pdf.close()
        phases["close"] = time.perf_counter() - started
        return phases

    return {"all": phased}


def unstructured_operations(side: str) -> dict[str, Operation]:
    if side == "real":
        from unstructured.partition.pdf import partition_pdf  # noqa: PLC0415  # ty: ignore[unresolved-import]
    else:
        from core_pdf_compat.unstructured import partition_pdf  # noqa: PLC0415

    def partition(path: str) -> list[object]:
        seen: list[object] = []
        for element in partition_pdf(path, strategy="fast"):
            data = element.metadata
            seen += (element.text, element.id, element.category, data.coordinates)
            seen += (data.page_number, data.languages, data.links, data.parent_id)
            seen += (data.filename, data.file_directory, data.filetype)
        return seen

    return {"partition_fast": partition}


def llamaindex_operations(side: str) -> dict[str, Operation]:
    if side == "real":
        from llama_index.readers.file import PDFReader  # noqa: PLC0415  # ty: ignore[unresolved-import]
    else:
        from core_pdf_compat.llamaindex import PDFReader  # noqa: PLC0415

    extra_info = {"source": "check", "file_name": "renamed.pdf"}
    return {
        "pages": lambda path: PDFReader().load_data(Path(path)),
        "full": lambda path: PDFReader(return_full_document=True).load_data(Path(path)),
        "extra": lambda path: PDFReader().load_data(Path(path), dict(extra_info)),
    }


def xray_operations(side: str) -> dict[str, Operation]:
    if side == "real":
        import fitz  # noqa: PLC0415  # ty: ignore[unresolved-import]
        import xray as library  # noqa: PLC0415  # ty: ignore[unresolved-import]

        fitz.TOOLS.mupdf_display_errors(False)  # noqa: FBT003
        fitz.TOOLS.mupdf_display_warnings(False)  # noqa: FBT003
    else:
        from core_pdf_compat import xray as library  # noqa: PLC0415

    return {
        "path": lambda path: asyncio.run(library.inspect(path)),
        "bytes": lambda path: asyncio.run(library.inspect(Path(path).read_bytes())),
    }


OPERATIONS: dict[str, Callable[[str], dict[str, Operation]]] = {
    "pypdf": pypdf_operations,
    "pdfminer": pdfminer_operations,
    "pdfplumber": pdfplumber_operations,
    "unstructured": unstructured_operations,
    "llamaindex": llamaindex_operations,
    "xray": xray_operations,
}


def on_alarm(signum: int, frame: Any) -> None:
    raise Timeout


def call(operation: Operation, path: str, timeout: int) -> tuple[dict[str, float], str | None]:
    """Seconds taken (one entry, or one per phase), and the exception raised, if any."""
    gc.collect()
    error = None
    signal.alarm(timeout)
    started = time.perf_counter()
    try:
        result = operation(path)
    except Timeout:
        raise
    except Exception as raised:  # noqa: BLE001
        result, error = None, type(raised).__name__
    finally:
        signal.alarm(0)
    elapsed = time.perf_counter() - started
    phases = result if isinstance(result, Phases) else {}
    return {**phases, "": elapsed}, error


def worker(arguments: argparse.Namespace) -> int:
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    signal.signal(signal.SIGALRM, on_alarm)
    library, side = arguments.worker
    with open(arguments.out, "a", encoding="utf-8") as out:  # noqa: PTH123

        def emit(**record: Any) -> None:
            out.write(json.dumps(record) + "\n")
            out.flush()

        started = time.perf_counter()
        operations = OPERATIONS[library](side)
        imported = time.perf_counter() - started
        versions = {}
        for name in VERSIONED[library]:
            with contextlib.suppress(metadata.PackageNotFoundError):
                versions[name] = metadata.version(name)
        # One untimed call of each operation on a small file first, so lazy imports and
        # caches filled on first use are reported here rather than charged to one file.
        started = time.perf_counter()
        for operation in operations.values():
            with contextlib.suppress(Exception):
                call(operation, WARM_UP, arguments.timeout)
        warmed = time.perf_counter() - started
        emit(imported=imported, warmed=warmed, versions=versions, python=sys.version.split()[0])
        wanted = set(arguments.operation or ())
        for path in Path(arguments.list).read_text().splitlines():
            emit(start=path)
            for name, operation in operations.items():
                if wanted and name not in wanted and library != "pdfplumber":
                    continue
                times: dict[str, list[float]] = {}
                error = None
                try:
                    for _ in range(arguments.repeats):
                        taken, error = call(operation, path, arguments.timeout)
                        for phase, taken_seconds in taken.items():
                            times.setdefault(phase or name, []).append(taken_seconds)
                except Timeout:
                    emit(path=path, operation=name, timeout=True)
                    continue
                for phase, runs in times.items():
                    emit(path=path, operation=phase, times=runs, error=error)
                if arguments.profile:
                    profile(arguments.profile, library, side, name, operation, path)
            emit(done=path)
    return 0


def profile(
    directory: str, library: str, side: str, name: str, operation: Operation, path: str
) -> None:
    profiler = cProfile.Profile()
    with contextlib.suppress(Exception):
        profiler.runcall(operation, path)
    Path(directory).mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(f"{directory}/{library}-{side}-{name}-{Path(path).name}.prof")


# ---------------------------------------------------------------------------------------------
# The parent: run both sides, compare


def pins(library: str) -> list[str]:
    suite = SUITES[LIBRARIES[library].suite]
    found = [*suite.reference, *(item for item in suite.requires if "==" in item)]
    return [option for pin in dict.fromkeys(found) for option in ("--with", pin)]


def run_side(library: str, side: str, files: list[str], arguments: argparse.Namespace) -> dict:
    """The side's import time and versions, and per (operation, file) the timings."""
    found: dict[str, Any] = {"timings": {}, "excluded": {}}
    remaining = list(files)
    reference_python = SUITES[LIBRARIES[library].suite].reference_python
    workspace = [
        *("uv", "run", "--locked", "--all-packages", "--group", "test"),
        *LIBRARIES[library].uv_options,
    ]
    if reference_python is None:
        environment = [*workspace, *pins(library)]
    elif side == "real":
        # The real library cannot run on the workspace's Python (run_upstream_suite.py says
        # why); it runs on its own, with none of the workspace's packages, and the facade
        # without the real library.
        reference = ("uv", "run", "--python", reference_python, "--isolated", "--no-project")
        environment = [*reference, *pins(library)]
    else:
        environment = workspace
    command = [
        *environment,
        *("python", __file__, "--worker", library, side),
        *("--repeats", str(arguments.repeats), "--timeout", str(arguments.timeout)),
        *(option for name in arguments.operation or () for option in ("--operation", name)),
        *(("--profile", arguments.profile) if arguments.profile else ()),
    ]
    with tempfile.TemporaryDirectory() as scratch:
        while remaining:
            listing, out = Path(scratch, "files"), Path(scratch, "out.jsonl")
            listing.write_text("\n".join(remaining))
            out.write_text("")
            process = subprocess.Popen(  # noqa: S603
                [*command, "--list", str(listing), "--out", str(out)],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL if not arguments.verbose else None,
            )
            hung = watch(process, out, arguments)
            current = None
            for line in out.read_text().splitlines():
                record = json.loads(line)
                if "imported" in record:
                    found.setdefault("imported", record["imported"])
                    found.setdefault("warmed", record["warmed"])
                    found["versions"] = record["versions"]
                    found["python"] = record["python"]
                elif "start" in record:
                    current = record["start"]
                elif "done" in record:
                    remaining.remove(record["done"])
                    current = None
                elif record.get("timeout"):
                    found["excluded"][record["path"]] = "timeout"
                else:
                    key = f"{record['operation']}\t{record['path']}"
                    found["timings"][key] = (statistics.median(record["times"]), record["error"])
            if current is None and remaining:
                if "imported" not in found:
                    raise SystemExit(f"{library} {side}: the worker failed before any file")
                current = remaining[0]
            if current is not None:
                found["excluded"][current] = "hangs" if hung else f"exit {process.returncode}"
                remaining.remove(current)
    return found


def watch(process: subprocess.Popen[bytes], out: Path, arguments: argparse.Namespace) -> bool:
    """Wait for the worker; kill it when it writes nothing for far longer than --timeout."""
    quiet = arguments.timeout * arguments.repeats * 4 + 120
    size, changed = 0, time.monotonic()
    while process.poll() is None:
        time.sleep(0.5)
        now = out.stat().st_size
        if now != size:
            size, changed = now, time.monotonic()
        elif time.monotonic() - changed > quiet:
            process.kill()
            process.wait()
            return True
    return False


def machine() -> str:
    cpu = platform.processor()
    with contextlib.suppress(Exception):
        if sys.platform == "darwin":
            cpu = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        else:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    return f"{platform.platform()}, {cpu}, {os.cpu_count()} CPUs"


def seconds(value: float) -> str:
    if value >= 1:
        return f"{value:.2f} s"
    return f"{value * 1000:.1f} ms" if value >= 0.001 else f"{value * 1e6:.0f} us"


def report(results: dict[str, dict[str, dict]], arguments: argparse.Namespace) -> str:
    lines = [f"Machine: {machine()}"]
    lines += [f"Medians of {arguments.repeats} runs per call; totals in seconds.", ""]
    lines += ["First calls: each operation once on simple1.pdf, untimed in the totals.", ""]
    lines += ["| library | real versions | Python | import real | import facade |"]
    lines[-1] += " first calls real | first calls facade |"
    lines += ["|---|---|---|---|---|---|---|"]
    for library, sides in results.items():
        real, facade = sides["real"], sides["facade"]
        versions = ", ".join(f"{name} {version}" for name, version in real["versions"].items())
        python = real["python"]
        if facade["python"] != python:
            python = f"real {python}, facade {facade['python']}"
        lines.append(
            f"| {library} | {versions} | {python} | {seconds(real['imported'])} "
            f"| {seconds(facade['imported'])} | {seconds(real['warmed'])} "
            f"| {seconds(facade['warmed'])} |"
        )
    lines += ["", "| library | operation | files | real s | facade s | facade/real |"]
    lines[-1] += " facade faster on | slower on | excluded |"
    lines += ["|---|---|---|---|---|---|---|---|---|"]
    worst_lines: list[str] = []
    for library, sides in results.items():
        real, facade = sides["real"], sides["facade"]
        excluded = real["excluded"] | facade["excluded"]
        for operation in LIBRARIES[library].operations:
            rows = []
            for key, (expected, error) in real["timings"].items():
                name, path = key.split("\t")
                if name != operation or path in excluded or key not in facade["timings"]:
                    continue
                actual, facade_error = facade["timings"][key]
                rows.append((path, expected, actual, error, facade_error))
            if not rows:
                continue
            total_real = sum(row[1] for row in rows)
            total_facade = sum(row[2] for row in rows)
            faster = sum(row[2] < row[1] for row in rows)
            lines.append(
                f"| {library} | {operation} | {len(rows)} | {total_real:.2f} "
                f"| {total_facade:.2f} | {total_facade / total_real:.2f}x | {faster} "
                f"| {len(rows) - faster} | {len(excluded)} |"
            )
            # Calls under NOISE on both sides are left out of the worst files, where timer
            # noise would rank them; they still count in the totals.
            ranked = [row for row in rows if max(row[1], row[2]) >= NOISE]
            ranked.sort(key=lambda row: row[2] / max(row[1], 1e-9), reverse=True)
            worst_lines.append(f"- {library} {operation}:")
            for path, expected, actual, error, facade_error in ranked[:WORST]:
                raised = f" (raises {error} / {facade_error})" if error or facade_error else ""
                worst_lines.append(
                    f"  - {Path(path).name}: {actual / max(expected, 1e-9):.1f}x, real "
                    f"{seconds(expected)}, facade {seconds(actual)}{raised}"
                )
    title = (
        f"Worst {WORST} files by facade/real per operation (calls of {NOISE * 1000:g} ms or more):"
    )
    lines += ["", title, *worst_lines]
    excluded_lines = [
        f"- {library} {side}: {Path(path).name} {why}"
        for library, sides in results.items()
        for side, found in sides.items()
        for path, why in found["excluded"].items()
    ]
    if excluded_lines:
        lines += ["", "Excluded (from both sides):", *excluded_lines]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", action="append", choices=sorted(LIBRARIES))
    parser.add_argument("--operation", action="append", help="pdfplumber always runs all")
    parser.add_argument("--files", nargs="+", default=[], help="file names or paths")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300, help="seconds per call")
    parser.add_argument("--profile", help="directory for one cProfile dump per call")
    parser.add_argument("--json", help="write every median here")
    parser.add_argument("--verbose", action="store_true", help="show the workers' stderr")
    parser.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)
    parser.add_argument("--list", help=argparse.SUPPRESS)
    parser.add_argument("--out", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.worker:
        return worker(arguments)
    os.chdir(ROOT)
    results: dict[str, dict[str, dict]] = {}
    for library in arguments.library or LIBRARIES:
        files = discover(library, arguments.files)
        if not files:
            continue
        print(f"{library}: {len(files)} files", file=sys.stderr)
        results[library] = {}
        for side in ("real", "facade"):
            started = time.monotonic()
            results[library][side] = run_side(library, side, files, arguments)
            print(f"  {side}: {time.monotonic() - started:.0f}s", file=sys.stderr)
    if arguments.json:
        Path(arguments.json).write_text(json.dumps(results, indent=1))
    print(report(results, arguments))
    return 0


if __name__ == "__main__":
    sys.exit(main())
