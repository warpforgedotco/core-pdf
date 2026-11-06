"""Check that the pdfplumber facade reads pages as pdfplumber does, on the upstream PDFs.

uv run --all-packages --with "pdfplumber==0.11.10" python scripts/check_pdfplumber_output.py

Each file is opened by the real pdfplumber and by the facade, and the first MAX_PAGES pages
of each are compared check by check:

chars: text, fontname, size, x0, x1, top, bottom, matrix, stroking and non-stroking colors.
rects, lines, curves: every key pdfplumber gives the object.
images: bbox, srcsize, name, bits, colorspace, imagemask, mcid, tag.
words: extract_words().
text: extract_text().
layout: extract_text(layout=True).
tables: extract_tables().

Floats are equal when they agree to within FLOAT_TOLERANCE (absolute), so the last bits of
float arithmetic do not count but a misplaced glyph does. A check that raises must raise
an exception of the same class name in both libraries, and opening the file likewise.

Every difference fails the check unless DEPARTURES lists it, with the reason the facade does
not follow pdfplumber there. A listed difference that no longer differs fails it too, so the
list stays true. The first difference of each file is printed.

Pass file paths as arguments to check only those files, and --all to print every
difference rather than the first of each file.
"""

import collections
import glob
import logging
import math
import os
import sys
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

# Every PDF in the upstream checkouts but pypdf's downloads, which CI does not fetch for this
# check. Symbolic links are left out: each points at a file covered under its own name, or at
# none.
PATTERN = "tests/upstream/**/*.pdf"
DOWNLOADS = "tests/upstream/pypdf/tests/pdf_cache/"
MAX_PAGES = 30
FLOAT_TOLERANCE = 1e-3
CHAR_KEYS = (
    "text",
    "fontname",
    "size",
    "x0",
    "x1",
    "top",
    "bottom",
    "matrix",
    "stroking_color",
    "non_stroking_color",
)
IMAGE_KEYS = (
    "x0",
    "top",
    "x1",
    "bottom",
    "srcsize",
    "name",
    "bits",
    "colorspace",
    "imagemask",
    "mcid",
    "tag",
)
# (file name, check) -> why the facade does not follow pdfplumber there; it covers every
# page of that check.
DEPARTURES: dict[tuple[str, str], str] = {}


def plain(value: Any) -> Any:
    """Turn pdfminer values (literals, references, streams) into comparable plain data.

    Lists stay lists and tuples stay tuples, since pdfplumber users compare them with ==.
    """
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [plain(item) for item in value]
    if isinstance(value, tuple):
        return tuple(plain(item) for item in value)
    if value is None or isinstance(value, bool | int | float | str | bytes):
        return value
    name = getattr(value, "name", None)
    if isinstance(name, str | bytes):
        return f"/{name if isinstance(name, str) else name.decode('latin-1')}"
    if hasattr(value, "rawdata") or hasattr(value, "attrs"):
        return "<stream>"
    if hasattr(value, "objid"):
        return f"<ref {value.objid}>"
    return f"<{type(value).__name__}>"


def same(left: Any, right: Any) -> bool:
    if isinstance(left, float | int) and isinstance(right, float | int):
        if isinstance(left, bool) or isinstance(right, bool):
            return left is right
        return math.isclose(left, right, abs_tol=FLOAT_TOLERANCE)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, list | tuple) and type(left) is type(right):
        return len(left) == len(right) and all(same(x, y) for x, y in zip(left, right, strict=True))
    return type(left) is type(right) and left == right


def difference(left: Any, right: Any, where: str = "") -> str:
    """Describe the first place where two plain values differ."""
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return f"{where} keys: pdfplumber {sorted(left)}, facade {sorted(right)}"
        key = next(k for k in left if not same(left[k], right[k]))
        return difference(left[key], right[key], f"{where}.{key}")
    if isinstance(left, list | tuple) and type(left) is type(right):
        for index, (x, y) in enumerate(zip(left, right, strict=False)):
            if not same(x, y):
                return difference(x, y, f"{where}[{index}]")
        return f"{where} length: pdfplumber {len(left)}, facade {len(right)}"
    return f"{where}: pdfplumber {left!r}, facade {right!r}"


def pick(objects: list[dict[str, Any]], keys: tuple[str, ...] | None) -> list[Any]:
    if keys is None:
        return [{key: plain(value) for key, value in obj.items()} for obj in objects]
    return [{key: plain(obj.get(key)) for key in keys} for obj in objects]


def images(page: Any) -> list[Any]:
    # pdfminer.six names an inline image by the id() of its stream object, which no other
    # process can reproduce, so a name of digits on both sides counts as the same.
    found = pick(page.images, IMAGE_KEYS)
    for image in found:
        if isinstance(image["name"], str) and image["name"].isdigit():
            image["name"] = "<inline image>"
    return found


CHECKS: dict[str, Callable[[Any], Any]] = {
    "chars": lambda page: pick(page.chars, CHAR_KEYS),
    "rects": lambda page: pick(page.rects, None),
    "lines": lambda page: pick(page.lines, None),
    "curves": lambda page: pick(page.curves, None),
    "images": images,
    "words": lambda page: plain(page.extract_words()),
    "text": lambda page: page.extract_text(),
    "layout": lambda page: page.extract_text(layout=True),
    "tables": lambda page: plain(page.extract_tables()),
}


def outcome(function: Callable[[], Any]) -> Any:
    try:
        return ("ok", function())
    except Exception as error:  # noqa: BLE001
        return ("error", f"{type(error).__name__}: {error}")


def read(library: Any, path: str) -> dict[str, Any]:
    """Every check's outcome for each of the first MAX_PAGES pages, keyed "page N check"."""
    found: dict[str, Any] = {}
    opened = outcome(lambda: library.open(path))
    if opened[0] == "error":
        found["open"] = opened
        return found
    pdf = opened[1]
    pages = outcome(lambda: pdf.pages)
    found["pages"] = ("ok", len(pages[1])) if pages[0] == "ok" else pages
    if pages[0] == "ok":
        for number, page in enumerate(pages[1][:MAX_PAGES], 1):
            for name, check in CHECKS.items():
                found[f"page {number} {name}"] = outcome(lambda c=check, p=page: c(p))
            page.close()
    # close() reads the pages too, so it raises again where reading them raised.
    found["close"] = outcome(pdf.close)
    return found


def compare(path: str) -> list[tuple[str, str, str]]:
    """(check, status, detail) for every check of one file."""
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    import pdfplumber as real  # noqa: PLC0415  # ty: ignore[unresolved-import]

    from core_pdf_compat import pdfplumber as facade  # noqa: PLC0415

    expected = read(real, path)
    actual = read(facade, path)
    differences: dict[str, list[str]] = collections.defaultdict(list)
    results = []
    for key in sorted(expected.keys() | actual.keys()):
        check = key.rsplit(" ", 1)[-1]
        left, right = expected.get(key), actual.get(key)
        if left is None or right is None:
            library = "pdfplumber" if right is None else "the facade"
            differences[check].append(f"{key}: only {library} reaches it")
        elif same(plain(left), plain(right)):
            results.append((check, "same", key))
        else:
            differences[check].append(f"{key}{difference(plain(left), plain(right))}"[:400])
    # A departure covers every page of its check, and must still differ on one of them.
    name = Path(path).name
    results += [
        (check, "LISTED BUT SAME", f"{check}: listed as a departure")
        for listed_name, check in DEPARTURES
        if listed_name == name and check not in differences
    ]
    for check, found in differences.items():
        status = "differ, listed departure" if (name, check) in DEPARTURES else "DIFFER"
        results += [(check, status, detail) for detail in found]
    return results


def main() -> int:
    every = "--all" in sys.argv
    arguments = [argument for argument in sys.argv[1:] if argument != "--all"]
    paths = arguments or sorted(
        path
        for path in glob.glob(PATTERN, recursive=True)
        if not path.startswith(DOWNLOADS) and not os.path.islink(path)
    )
    counts: collections.Counter[str] = collections.Counter()
    problems: list[str] = []
    with ProcessPoolExecutor() as pool:
        for path, results in zip(paths, pool.map(compare, paths), strict=True):
            found = []
            for check, status, detail in results:
                counts[f"{check}: {status}"] += 1
                if status.isupper():
                    found.append(f"{path}: {detail}")
            problems.extend(found if every else found[:1])
    print(f"{len(paths)} files, first {MAX_PAGES} pages each, floats within {FLOAT_TOLERANCE}")
    for key, count in sorted(counts.items()):
        print(f"{count:>6}  {key}")
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
