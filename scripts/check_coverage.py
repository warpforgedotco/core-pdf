from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any


def check_coverage(
    report: dict[str, Any], baseline: dict[str, Any], expected_files: set[str]
) -> list[str]:
    if report.get("meta", {}).get("branch_coverage") is not True:
        raise ValueError("branch coverage must be enabled")
    files = report.get("files", {})
    if set(files) != expected_files:
        missing = sorted(expected_files - files.keys())
        extra = sorted(files.keys() - expected_files)
        raise ValueError(f"incomplete coverage inventory: missing={missing}, extra={extra}")
    totals = report["totals"]
    for key in ("covered_lines", "num_statements", "covered_branches", "num_branches"):
        if totals[key] != sum(file["summary"][key] for file in files.values()):
            raise ValueError(f"inconsistent coverage total: {key}")
    messages = []
    for label, covered, total in (
        ("statements", "covered_lines", "num_statements"),
        ("branches", "covered_branches", "num_branches"),
    ):
        numerator, denominator = totals[covered], totals[total]
        if not 0 <= numerator <= denominator or denominator <= 0:
            raise ValueError(f"invalid {label} coverage counts")
        floor = baseline[label]
        messages.append(
            f"Workspace {label}: {numerator}/{denominator} ({numerator / denominator:.2%})"
        )
        if numerator * floor["total"] < floor["covered"] * denominator:
            raise ValueError(f"{label} coverage regressed: {messages[-1]}")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, nargs="?", default=Path("htmlcov/coverage.json"))
    parser.add_argument("--baseline", type=Path, default=Path("docs/coverage-baseline.json"))
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    config = tomllib.loads(Path("pyproject.toml").read_text())["tool"]["coverage"]["run"]
    if config["source"] != baseline["sources"] or config["omit"] != ["*/_vendor/*"]:
        raise ValueError("coverage source roots or exclusions changed")
    expected = {
        str(path)
        for root in baseline["sources"]
        for path in Path(root).rglob("*.py")
        if "_vendor" not in path.parts
    }
    for message in check_coverage(json.loads(args.report.read_text()), baseline, expected):
        print(message)


if __name__ == "__main__":
    main()
