# Ported from pdfplumber's cli.py, Copyright (c) 2015, Jeremy Singer-Vine, under the
# MIT License.
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from itertools import chain
from typing import Any

from . import PDF

if len(sys.argv) == 1:
    sys.argv.append("--help")


def parse_page_spec(value: str) -> list[int]:
    if "-" in value:
        start, end = map(int, value.split("-"))
        return list(range(start, end + 1))
    return [int(value)]


def parse_args(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser("pdfplumber")
    parser.add_argument("infile", nargs="?", type=str)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--structure",
        help="Write the structure tree as JSON.  "
        "All other arguments except --pages, --laparams, and --indent will be ignored",
        action="store_true",
    )
    group.add_argument(
        "--structure-text",
        help="Write the structure tree as JSON including text contents.  "
        "All other arguments except --pages, --laparams, and --indent will be ignored",
        action="store_true",
    )
    parser.add_argument("--format", choices=["csv", "json", "text"], default="csv")
    parser.add_argument("--types", nargs="+")
    parser.add_argument(
        "--include-attrs", nargs="+", help="Include *only* these object attributes in output."
    )
    parser.add_argument(
        "--exclude-attrs", nargs="+", help="Exclude these object attributes from output."
    )
    parser.add_argument("--laparams", type=json.loads)
    parser.add_argument("--precision", type=int)
    parser.add_argument("--pages", nargs="+", type=parse_page_spec)
    parser.add_argument("--indent", type=int, help="Indent level for JSON pretty-printing.")
    parsed = parser.parse_args(arguments)
    if parsed.pages is not None:
        parsed.pages = list(chain(*parsed.pages))
    return parsed


def add_text_to_mcids(pdf: PDF, data: list[dict[str, Any]]) -> None:
    page_contents: defaultdict[int, defaultdict[int, str]] = defaultdict(lambda: defaultdict(str))
    for page in pdf.pages:
        text_contents = page_contents[page.page_number]
        for char in page.chars:
            mcid = char.get("mcid")
            if mcid is not None:
                text_contents[mcid] += char["text"]
    elements = deque(data)
    while elements:
        element = elements.popleft()
        elements.extend(element.get("children", ()))
        page_number = element.get("page_number")
        if page_number is not None and "mcids" in element:
            contents = page_contents[page_number]
            element["text"] = [contents[mcid] for mcid in element["mcids"]]


def main(arguments: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if arguments is None else arguments)
    with PDF.open(args.infile, pages=args.pages, laparams=args.laparams) as pdf:
        if args.structure:
            print(json.dumps(pdf.structure_tree, indent=args.indent))
        elif args.structure_text:
            tree = pdf.structure_tree
            add_text_to_mcids(pdf, tree)
            print(json.dumps(tree, indent=args.indent, ensure_ascii=False))
        elif args.format == "text":
            for page in pdf.pages:
                print(page.extract_text(layout=True))
        else:
            export = pdf.to_csv if args.format == "csv" else pdf.to_json
            options = {} if args.format == "csv" else {"indent": args.indent}
            export(
                sys.stdout,
                args.types,
                precision=args.precision,
                include_attrs=args.include_attrs,
                exclude_attrs=args.exclude_attrs,
                **options,
            )


if __name__ == "__main__":
    main()
