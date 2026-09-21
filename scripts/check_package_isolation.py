# SPDX-License-Identifier: AGPL-3.0-only
"""Prove a workspace member is installed without the tiers above it.

Usage: ``python scripts/check_package_isolation.py <distribution>``

Run after ``uv sync --locked --package <distribution> --group test``. Every
``core_*`` import name that belongs to another workspace member must be absent
unless that member's ``pyproject.toml`` declares it as a dependency.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import tomllib
from pathlib import Path

internal_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
internal_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def internal_workspace_members() -> dict[str, Path]:
    with (internal_REPOSITORY_ROOT / "pyproject.toml").open("rb") as handle:
        root = tomllib.load(handle)
    members = {root["project"]["name"]: internal_REPOSITORY_ROOT}
    for member in root["tool"]["uv"]["workspace"]["members"]:
        path = internal_REPOSITORY_ROOT / member
        with (path / "pyproject.toml").open("rb") as handle:
            members[tomllib.load(handle)["project"]["name"]] = path
    return members


def internal_import_name(distribution: str) -> str:
    return distribution.replace("-", "_")


def internal_declared_dependencies(path: Path) -> set[str]:
    with (path / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    declared: set[str] = set()
    for requirement in project.get("dependencies", ()):
        match = internal_REQUIREMENT_NAME.match(requirement)
        if match is not None:
            declared.add(match.group(1).lower().replace("_", "-"))
    return declared


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    target = argv[1]
    members = internal_workspace_members()
    if target not in members:
        print(f"unknown workspace member: {target}", file=sys.stderr)
        return 2
    allowed = {target} | internal_declared_dependencies(members[target])
    # Transitive workspace dependencies are legitimately present.
    pending = list(allowed - {target})
    while pending:
        name = pending.pop()
        if name in members:
            for dependency in internal_declared_dependencies(members[name]):
                if dependency not in allowed:
                    allowed.add(dependency)
                    pending.append(dependency)
    failures: list[str] = []
    for name in sorted(members):
        import_name = internal_import_name(name)
        present = importlib.util.find_spec(import_name) is not None
        if name in allowed and not present:
            failures.append(f"{import_name} should be importable for {target} but is missing")
        if name not in allowed and present:
            failures.append(f"{import_name} is importable but {target} does not depend on {name}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"{target}: only its declared workspace dependencies are installed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
