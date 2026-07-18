from __future__ import annotations

import argparse
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from shutil import which

from core_jpeg import JpegError, decode_jpx_image
from core_jpeg.impl.codecs.jpx.output import clamp_sample

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT_CANDIDATES = (
    REPO_ROOT / "openjpeg-data",
    REPO_ROOT / "openjpeg-reference" / "data",
)
DEFAULT_OPENJPEG_BIN_CANDIDATES = (
    Path("/tmp/core-jpeg-openjpeg-build/bin/opj_decompress"),
    REPO_ROOT / "openjpeg-reference" / "build" / "bin" / "opj_decompress",
)
CASE_STATUSES = ("pass", "bug", "known_unsupported")


@dataclass(frozen=True, slots=True)
class PgxImage:
    width: int
    height: int
    precision: int
    is_signed: bool
    data: bytes


@dataclass(frozen=True, slots=True)
class OpenJPEGCase:
    path: str
    status: str
    reason: str | None = None


def discover_openjpeg_input_fixtures(data_root: Path) -> list[str]:
    input_path = data_root / "input"
    return sorted(
        str(path.relative_to(data_root))
        for path in input_path.rglob("*")
        if path.suffix.lower() in {".jp2", ".j2k", ".j2c"}
    )


def resolve_data_root(data_root: str | Path | None) -> Path:
    if data_root is not None:
        path = Path(data_root)
        if not path.exists():
            raise FileNotFoundError(f"openjpeg-data root does not exist: {path}")
        return path
    for candidate in DEFAULT_DATA_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "openjpeg-data root not found; expected ./openjpeg-data or ./openjpeg-reference/data"
    )


def resolve_opj_decompress(bin_path: str | Path | None) -> Path:
    if bin_path is not None:
        path = Path(bin_path)
        if path.is_dir():
            path = path / "opj_decompress"
        if not path.exists():
            raise FileNotFoundError(f"opj_decompress not found: {path}")
        return path
    for candidate in DEFAULT_OPENJPEG_BIN_CANDIDATES:
        if candidate.exists():
            return candidate
    path_env = which("opj_decompress")
    if path_env:
        return Path(path_env)
    raise FileNotFoundError(
        "opj_decompress not found; expected /tmp/core-jpeg-openjpeg-build/bin, "
        "./openjpeg-reference/build/bin, or PATH"
    )


def read_pgx(path: Path) -> PgxImage:
    with path.open("rb") as handle:
        header = handle.readline().decode("ascii").strip().split()
        if len(header) != 6 or header[0] != "PG":
            raise ValueError(f"unsupported PGX header in {path}")
        endian, signed, precision, width, height = header[1:]
        if endian not in {"ML", "LM"}:
            raise ValueError(f"unsupported PGX endian marker {endian}")
        precision_int = int(precision)
        width_int = int(width)
        height_int = int(height)
        raw_data = handle.read()

    byte_count = (precision_int + 7) // 8
    expected_len = width_int * height_int * byte_count
    if byte_count <= 0 or len(raw_data) != expected_len:
        raise ValueError(f"unexpected PGX data length in {path}")

    byte_order = "big" if endian == "ML" else "little"
    max_value = 1 << precision_int
    sign_bit = 1 << (precision_int - 1)
    data = bytearray(width_int * height_int)
    for index in range(width_int * height_int):
        offset = index * byte_count
        value = int.from_bytes(raw_data[offset : offset + byte_count], byte_order)
        if signed == "-" and value & sign_bit:
            value -= max_value
        data[index] = clamp_sample(value, precision_int)

    return PgxImage(
        width=width_int,
        height=height_int,
        precision=precision_int,
        is_signed=signed == "-",
        data=bytes(data),
    )


def decode_with_openjpeg(
    opj_decompress: Path,
    source: Path,
    output_stem: Path,
) -> list[PgxImage]:
    output = output_stem.with_suffix(".pgx")
    subprocess.run(
        [str(opj_decompress), "-i", str(source), "-o", str(output)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    component_files = sorted(output.parent.glob(f"{output.stem}_*.pgx"))
    if not component_files and output.exists():
        component_files = [output]
    return [read_pgx(path) for path in component_files]


def components_match(decoded, oracle: list[PgxImage]) -> bool:
    if not oracle:
        return False
    if decoded.width != oracle[0].width or decoded.height != oracle[0].height:
        return False
    if len(decoded.components) < len(oracle):
        return False
    for actual, expected in zip(decoded.components, oracle, strict=False):
        if actual.width != expected.width:
            return False
        if actual.height != expected.height:
            return False
        if actual.precision != expected.precision:
            return False
        if actual.is_signed != expected.is_signed:
            return False
        if actual.data != expected.data:
            return False
    return True


def classify_openjpeg_case(
    data_root: Path,
    opj_decompress: Path,
    relative_path: str,
) -> OpenJPEGCase:
    source = data_root / relative_path
    with tempfile.TemporaryDirectory(prefix="core-jpeg-openjpeg-") as tmpdir:
        try:
            oracle = decode_with_openjpeg(
                opj_decompress,
                source,
                Path(tmpdir) / "oracle",
            )
        except subprocess.CalledProcessError as exc:
            reason = exc.stderr.decode("utf-8", "ignore").strip() if exc.stderr else ""
            return OpenJPEGCase(
                path=relative_path,
                status="known_unsupported",
                reason=reason or "OpenJPEG failed to decode the fixture",
            )
        except OSError as exc:
            return OpenJPEGCase(
                path=relative_path,
                status="known_unsupported",
                reason=str(exc),
            )

    try:
        decoded = decode_jpx_image(source.read_bytes(), apply_embedded_color=False)
    except JpegError as exc:
        return OpenJPEGCase(
            path=relative_path,
            status="bug",
            reason=f"core_jpeg failed to decode: {exc}",
        )
    except Exception as exc:  # pragma: no cover - defensive tooling path
        return OpenJPEGCase(
            path=relative_path,
            status="bug",
            reason=f"unexpected exception: {type(exc).__name__}: {exc}",
        )

    if not components_match(decoded, oracle):
        return OpenJPEGCase(
            path=relative_path,
            status="bug",
            reason="component output mismatch",
        )
    return OpenJPEGCase(path=relative_path, status="pass")


def print_case(case: OpenJPEGCase) -> None:
    if case.reason:
        print(f"{case.status}\t{case.path}\t{case.reason}")
    else:
        print(f"{case.status}\t{case.path}")


def run_conformance(
    data_root: Path,
    opj_decompress: Path,
    *,
    limit: int | None = None,
    fail_fast: bool = False,
    verbose: bool = False,
) -> int:
    fixtures = discover_openjpeg_input_fixtures(data_root)
    if limit is not None:
        fixtures = fixtures[:limit]
    counts: Counter[str] = Counter()
    for index, relative_path in enumerate(fixtures, start=1):
        case = classify_openjpeg_case(data_root, opj_decompress, relative_path)
        counts[case.status] += 1
        if verbose or case.status != "pass":
            print_case(case)
        if index % 25 == 0:
            print(
                f"progress\t{index}/{len(fixtures)}\tpass={counts['pass']}\t"
                f"bug={counts['bug']}\tknown_unsupported={counts['known_unsupported']}"
            )
        if fail_fast and case.status == "bug":
            break

    print(
        f"summary\tpass={counts['pass']}\tbug={counts['bug']}\t"
        f"known_unsupported={counts['known_unsupported']}\ttotal={sum(counts.values())}"
    )
    return 1 if counts["bug"] else 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="openjpeg-conformance")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to the openjpeg-data checkout",
    )
    parser.add_argument(
        "--openjpeg-bin",
        default=None,
        help="Path to opj_decompress or its containing directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N fixtures",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first bug case",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print pass cases as well as failures",
    )
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    opj_decompress = resolve_opj_decompress(args.openjpeg_bin)
    return run_conformance(
        data_root,
        opj_decompress,
        limit=args.limit,
        fail_fast=args.fail_fast,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    raise SystemExit(main())
