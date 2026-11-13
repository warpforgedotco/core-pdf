# SPDX-License-Identifier: AGPL-3.0-only

"""Move core_pdf.impl's flat modules into subpackages and rewrite every reference to them.

MOVES maps each old module to its new dotted name. The script:

1. moves each file with ``git mv`` and creates the new packages' empty ``__init__.py``;
2. rewrites ``from core_pdf.impl import <module>`` lines into imports from the new package,
   keeping the local name (``from core_pdf.impl.extract import capture as extract_capture``);
3. rewrites every dotted name (``core_pdf.impl.fonts_cmap`` -> ``core_pdf.impl.fonts.cmap``)
   and every path (``impl/fonts_cmap.py`` -> ``impl/fonts/cmap.py``) in the repository's
   tracked text files, submodules excluded.

It is idempotent: a second run finds nothing to move or rewrite. A branch made before the
restructure can be brought onto it by running the script on the branch and committing the
result before rebasing or merging.

``--check`` changes nothing and fails if any old name or path is left, or if a ruff
per-file-ignores path in pyproject.toml does not exist.

uv run --no-project python scripts/restructure_impl.py [--check]
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
IMPL = "src/core_pdf/impl"
PREFIX = "core_pdf.impl."

# Packages whose modules keep their name with the package prefix stripped:
# fonts_cmap -> fonts.cmap.
PREFIXED_PACKAGES = (
    "capture",
    "document",
    "extract",
    "fonts",
    "graphics",
    "layout",
    "recovery",
    "render",
)

# Every other move, by hand.
RENAMED = {
    # base: shared helpers that import only the floor packages and each other.
    "types": "base.types",
    "exceptions": "base.exceptions",
    "caches": "base.caches",
    "geometry": "base.geometry",
    "array_views": "base.arrays",
    "pdf_values": "base.values",
    "text": "base.text",
    "collector": "base.collector",
    "page_selection": "base.pages",
    "spatial": "base.spatial",
    "glyphs": "base.glyphs",
    "runs": "base.runs",
    "document_records": "base.raw_records",
    # output: the structured result and its serializers.
    "output_model": "output.model",
    "output_serialize": "output.serialize",
    "raw_media": "output.media",
    # document: names that would repeat the package.
    "document_document": "document.pdf_document",
    "document_page": "document.pdf_page",
    "object_serialize": "document.object_writer",
}


def flat_modules() -> list[str]:
    """The flat modules under impl/ now, or (after the move) the ones MOVES ever named."""
    return sorted(
        path.stem for path in (REPOSITORY_ROOT / IMPL).glob("*.py") if path.stem != "__init__"
    )


def build_moves(names: list[str]) -> dict[str, str]:
    moves: dict[str, str] = {}
    for name in names:
        if name in RENAMED:
            moves[name] = RENAMED[name]
            continue
        package, _, rest = name.partition("_")
        if package in PREFIXED_PACKAGES and rest:
            moves[name] = f"{package}.{rest}"
            continue
        raise SystemExit(f"no destination for core_pdf.impl.{name}; add it to RENAMED")
    return moves


def known_moves() -> dict[str, str]:
    """Every move this script knows, whether or not the module still exists: what --check
    and the rewrite look for."""
    names = set(flat_modules()) | set(RENAMED)
    # Prefixed modules already moved: recover their old names from the new files.
    for package in PREFIXED_PACKAGES:
        directory = REPOSITORY_ROOT / IMPL / package
        if directory.is_dir():
            names.update(
                f"{package}_{path.stem}"
                for path in directory.glob("*.py")
                if path.stem != "__init__"
            )
    return build_moves(sorted(names - {"__init__"}))


def tracked_text_files() -> list[Path]:
    output = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPOSITORY_ROOT, capture_output=True, check=True
    ).stdout
    paths = []
    for raw in output.split(b"\0"):
        if not raw:
            continue
        path = REPOSITORY_ROOT / raw.decode()
        if path.is_file() and path.suffix in {
            ".py",
            ".md",
            ".toml",
            ".yml",
            ".yaml",
            ".txt",
            ".cfg",
            ".tsv",
            ".sh",
            ".pyx",
        }:
            paths.append(path)
    # This script names the old modules on purpose.
    return [path for path in paths if path != Path(__file__).resolve()]


def move_files(moves: dict[str, str]) -> int:
    moved = 0
    for old, new in moves.items():
        source = REPOSITORY_ROOT / IMPL / f"{old}.py"
        if not source.exists():
            continue
        target = REPOSITORY_ROOT / IMPL / (new.replace(".", "/") + ".py")
        target.parent.mkdir(parents=True, exist_ok=True)
        init = target.parent / "__init__.py"
        if not init.exists():
            init.write_text("# SPDX-License-Identifier: AGPL-3.0-only\n")
            subprocess.run(["git", "add", str(init)], cwd=REPOSITORY_ROOT, check=True)
        subprocess.run(["git", "mv", str(source), str(target)], cwd=REPOSITORY_ROOT, check=True)
        moved += 1
    return moved


def compile_rewrites(
    moves: dict[str, str],
) -> tuple[re.Pattern[str], re.Pattern[str], re.Pattern[str]]:
    names = "|".join(sorted(map(re.escape, moves), key=len, reverse=True))
    dotted = re.compile(rf"\bcore_pdf\.impl\.({names})(?!\w)")
    path = re.compile(rf"\bimpl/({names})\.py\b")
    from_impl = re.compile(
        r"^(?P<indent>[ \t]*)(?P<lazy>lazy )?from core_pdf\.impl import (?P<names>[\w ,]+?)"
        r"(?P<comment>[ \t]*#.*)?$",
        re.MULTILINE,
    )
    return dotted, path, from_impl


def rewrite_text(text: str, moves: dict[str, str]) -> str:
    dotted, path, from_impl = compile_rewrites(moves)

    def split_from(match: re.Match[str]) -> str:
        imported = [name.strip() for name in match.group("names").split(",")]
        if not all(name in moves for name in imported):
            return match.group(0)
        lines = []
        for index, name in enumerate(imported):
            package, _, leaf = moves[name].rpartition(".")
            alias = "" if leaf == name else f" as {name}"
            comment = (match.group("comment") or "") if index == 0 else ""
            lines.append(
                f"{match.group('indent')}{match.group('lazy') or ''}"
                f"from core_pdf.impl.{package} import {leaf}{alias}{comment}"
            )
        return "\n".join(lines)

    text = from_impl.sub(split_from, text)
    text = dotted.sub(lambda match: PREFIX + moves[match.group(1)], text)
    return path.sub(lambda match: "impl/" + moves[match.group(1)].replace(".", "/") + ".py", text)


def leftovers(moves: dict[str, str]) -> list[str]:
    dotted, path, from_impl = compile_rewrites(moves)
    found = []
    for file in tracked_text_files():
        text = file.read_text(encoding="utf-8", errors="replace")
        for pattern in (dotted, path):
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                found.append(f"{file.relative_to(REPOSITORY_ROOT)}:{line}: {match.group(0)}")
        for match in from_impl.finditer(text):
            if any(name.strip() in moves for name in match.group("names").split(",")):
                line = text.count("\n", 0, match.start()) + 1
                found.append(
                    f"{file.relative_to(REPOSITORY_ROOT)}:{line}: {match.group(0).strip()}"
                )
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as handle:
        ignores = tomllib.load(handle)["tool"]["ruff"]["lint"].get("per-file-ignores", {})
    found.extend(
        f"pyproject.toml: per-file-ignores path does not exist: {pattern}"
        for pattern in ignores
        if not any(character in pattern for character in "*?[")
        and not (REPOSITORY_ROOT / pattern).exists()
    )
    return found


def main(argv: list[str]) -> int:
    check = argv[1:] == ["--check"]
    if argv[1:] and not check:
        print(__doc__)
        return 2
    if check:
        # After the restructure only impl/__init__.py stays flat. A module this script knows
        # comes from a branch made before it; any other belongs in a subpackage.
        flat = flat_modules()
        known = [
            name for name in flat if name in RENAMED or name.partition("_")[0] in PREFIXED_PACKAGES
        ]
        new = [name for name in flat if name not in known]
        if known:
            print(f"not moved yet, run this script: {', '.join(known)}")
        if new:
            print(f"flat modules under core_pdf.impl, put them in a subpackage: {', '.join(new)}")
        if flat:
            return 1
    moves = known_moves()
    if check:
        found = leftovers(moves)
        print("\n".join(found) if found else f"no references to {len(moves)} old impl modules")
        return 1 if found else 0
    moved = move_files(build_moves(flat_modules()))
    rewritten = 0
    for file in tracked_text_files():
        text = file.read_text(encoding="utf-8")
        new = rewrite_text(text, moves)
        if new != text:
            file.write_text(new, encoding="utf-8")
            rewritten += 1
    print(f"moved {moved} modules, rewrote {rewritten} files")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
