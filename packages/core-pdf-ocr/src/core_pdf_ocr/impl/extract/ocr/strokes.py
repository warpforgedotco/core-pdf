# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import string
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, NoReturn, Self, TypeAlias

from core_pdf.impl.capture.records import CapturedDrawing, CapturedPath
from core_pdf.impl.model.geometry import (
    bbox_area,
    bbox_intersection_area,
    bbox_union,
    points_bbox,
    rect_tuple,
)
from core_pdf.impl.types import Rectangle

frozen_setattr = object.__setattr__


GlyphSignature: TypeAlias = tuple[tuple[tuple[bool, tuple[tuple[int, int], ...]], ...], ...]
GlyphTopology: TypeAlias = tuple[tuple[tuple[bool, int], ...], ...]

STROKED_TEXT_SEED_MIN_CONFIDENCE = 85.0
STROKED_TEXT_MAX_TOKEN_CHARACTERS = 12
STROKED_TEXT_GLYPH_OVERLAP_TOLERANCE = 0.06
STROKED_TEXT_SIGNATURE_QUANTIZATION = 16
STROKED_TEXT_SIGNATURE_MAX_COORDINATE_DISTANCE = 1
STROKED_TEXT_SIGNATURE_MAX_MEAN_DISTANCE = 0.50
STROKED_TEXT_SINGLE_GLYPH_MIN_ASPECT_RATIO = 0.25
STROKED_TEXT_SEED_RUN_MIN_HEIGHT = 1.0
STROKED_TEXT_SEED_RUN_MAX_HEIGHT = 8.0
STROKED_TEXT_SEED_RUN_MAX_WIDTH = 64.0
STROKED_TEXT_ALLOWED_CHARACTERS = frozenset(string.ascii_letters + string.digits + "+-./_")


class StrokedTextSeed:
    __slots__ = ("text", "bbox", "confidence", "sequence")

    text: str
    bbox: Rectangle
    confidence: float
    sequence: int

    __fields__: ClassVar[tuple[str, ...]] = ("text", "bbox", "confidence", "sequence")
    __match_args__ = ("text", "bbox", "confidence", "sequence")

    def __init__(self, text: str, bbox: Rectangle, confidence: float, sequence: int) -> None:
        frozen_setattr(self, "text", text)
        frozen_setattr(self, "bbox", bbox)
        frozen_setattr(self, "confidence", confidence)
        frozen_setattr(self, "sequence", sequence)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"text={self.text!r}, "
            f"bbox={self.bbox!r}, "
            f"confidence={self.confidence!r}, "
            f"sequence={self.sequence!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.text == other.text
            and self.bbox == other.bbox
            and self.confidence == other.confidence
            and self.sequence == other.sequence
        )

    def __hash__(self) -> int:
        return hash((self.text, self.bbox, self.confidence, self.sequence))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        text = changes.pop("text", self.text)
        bbox = changes.pop("bbox", self.bbox)
        confidence = changes.pop("confidence", self.confidence)
        sequence = changes.pop("sequence", self.sequence)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(text, bbox, confidence, sequence)


class StrokedTextObservation:
    __slots__ = ("text", "bbox", "first_drawing", "last_drawing", "confidence")

    text: str
    bbox: Rectangle
    first_drawing: int
    last_drawing: int
    confidence: float

    __fields__: ClassVar[tuple[str, ...]] = (
        "text",
        "bbox",
        "first_drawing",
        "last_drawing",
        "confidence",
    )
    __match_args__ = ("text", "bbox", "first_drawing", "last_drawing", "confidence")

    def __init__(
        self,
        text: str,
        bbox: Rectangle,
        first_drawing: int,
        last_drawing: int,
        confidence: float = 96.0,
    ) -> None:
        frozen_setattr(self, "text", text)
        frozen_setattr(self, "bbox", bbox)
        frozen_setattr(self, "first_drawing", first_drawing)
        frozen_setattr(self, "last_drawing", last_drawing)
        frozen_setattr(self, "confidence", confidence)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"text={self.text!r}, "
            f"bbox={self.bbox!r}, "
            f"first_drawing={self.first_drawing!r}, "
            f"last_drawing={self.last_drawing!r}, "
            f"confidence={self.confidence!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.text == other.text
            and self.bbox == other.bbox
            and self.first_drawing == other.first_drawing
            and self.last_drawing == other.last_drawing
            and self.confidence == other.confidence
        )

    def __hash__(self) -> int:
        return hash((self.text, self.bbox, self.first_drawing, self.last_drawing, self.confidence))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        text = changes.pop("text", self.text)
        bbox = changes.pop("bbox", self.bbox)
        first_drawing = changes.pop("first_drawing", self.first_drawing)
        last_drawing = changes.pop("last_drawing", self.last_drawing)
        confidence = changes.pop("confidence", self.confidence)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(text, bbox, first_drawing, last_drawing, confidence)


class StrokedTextDecode:
    __slots__ = (
        "observations",
        "eligible_seeds",
        "aligned_seeds",
        "accepted_seeds",
        "initial_signatures",
        "learned_signatures",
        "approximate_signatures",
        "alphabet",
        "candidate_runs",
        "decoded_candidate_runs",
        "candidate_glyphs",
        "decoded_candidate_glyphs",
    )

    observations: tuple[StrokedTextObservation, ...]
    eligible_seeds: int
    aligned_seeds: int
    accepted_seeds: int
    initial_signatures: int
    learned_signatures: int
    approximate_signatures: int
    alphabet: tuple[tuple[GlyphSignature, str], ...]
    candidate_runs: int
    decoded_candidate_runs: int
    candidate_glyphs: int
    decoded_candidate_glyphs: int

    __fields__: ClassVar[tuple[str, ...]] = (
        "observations",
        "eligible_seeds",
        "aligned_seeds",
        "accepted_seeds",
        "initial_signatures",
        "learned_signatures",
        "approximate_signatures",
        "alphabet",
        "candidate_runs",
        "decoded_candidate_runs",
        "candidate_glyphs",
        "decoded_candidate_glyphs",
    )
    __match_args__ = (
        "observations",
        "eligible_seeds",
        "aligned_seeds",
        "accepted_seeds",
        "initial_signatures",
        "learned_signatures",
        "approximate_signatures",
        "alphabet",
        "candidate_runs",
        "decoded_candidate_runs",
        "candidate_glyphs",
        "decoded_candidate_glyphs",
    )

    def __init__(
        self,
        observations: tuple[StrokedTextObservation, ...] = (),
        eligible_seeds: int = 0,
        aligned_seeds: int = 0,
        accepted_seeds: int = 0,
        initial_signatures: int = 0,
        learned_signatures: int = 0,
        approximate_signatures: int = 0,
        alphabet: tuple[tuple[GlyphSignature, str], ...] = (),
        candidate_runs: int = 0,
        decoded_candidate_runs: int = 0,
        candidate_glyphs: int = 0,
        decoded_candidate_glyphs: int = 0,
    ) -> None:
        frozen_setattr(self, "observations", observations)
        frozen_setattr(self, "eligible_seeds", eligible_seeds)
        frozen_setattr(self, "aligned_seeds", aligned_seeds)
        frozen_setattr(self, "accepted_seeds", accepted_seeds)
        frozen_setattr(self, "initial_signatures", initial_signatures)
        frozen_setattr(self, "learned_signatures", learned_signatures)
        frozen_setattr(self, "approximate_signatures", approximate_signatures)
        frozen_setattr(self, "alphabet", alphabet)
        frozen_setattr(self, "candidate_runs", candidate_runs)
        frozen_setattr(self, "decoded_candidate_runs", decoded_candidate_runs)
        frozen_setattr(self, "candidate_glyphs", candidate_glyphs)
        frozen_setattr(self, "decoded_candidate_glyphs", decoded_candidate_glyphs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"observations={self.observations!r}, "
            f"eligible_seeds={self.eligible_seeds!r}, "
            f"aligned_seeds={self.aligned_seeds!r}, "
            f"accepted_seeds={self.accepted_seeds!r}, "
            f"initial_signatures={self.initial_signatures!r}, "
            f"learned_signatures={self.learned_signatures!r}, "
            f"approximate_signatures={self.approximate_signatures!r}, "
            f"alphabet={self.alphabet!r}, "
            f"candidate_runs={self.candidate_runs!r}, "
            f"decoded_candidate_runs={self.decoded_candidate_runs!r}, "
            f"candidate_glyphs={self.candidate_glyphs!r}, "
            f"decoded_candidate_glyphs={self.decoded_candidate_glyphs!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.observations == other.observations
            and self.eligible_seeds == other.eligible_seeds
            and self.aligned_seeds == other.aligned_seeds
            and self.accepted_seeds == other.accepted_seeds
            and self.initial_signatures == other.initial_signatures
            and self.learned_signatures == other.learned_signatures
            and self.approximate_signatures == other.approximate_signatures
            and self.alphabet == other.alphabet
            and self.candidate_runs == other.candidate_runs
            and self.decoded_candidate_runs == other.decoded_candidate_runs
            and self.candidate_glyphs == other.candidate_glyphs
            and self.decoded_candidate_glyphs == other.decoded_candidate_glyphs
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.observations,
                self.eligible_seeds,
                self.aligned_seeds,
                self.accepted_seeds,
                self.initial_signatures,
                self.learned_signatures,
                self.approximate_signatures,
                self.alphabet,
                self.candidate_runs,
                self.decoded_candidate_runs,
                self.candidate_glyphs,
                self.decoded_candidate_glyphs,
            )
        )

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        observations = changes.pop("observations", self.observations)
        eligible_seeds = changes.pop("eligible_seeds", self.eligible_seeds)
        aligned_seeds = changes.pop("aligned_seeds", self.aligned_seeds)
        accepted_seeds = changes.pop("accepted_seeds", self.accepted_seeds)
        initial_signatures = changes.pop("initial_signatures", self.initial_signatures)
        learned_signatures = changes.pop("learned_signatures", self.learned_signatures)
        approximate_signatures = changes.pop("approximate_signatures", self.approximate_signatures)
        alphabet = changes.pop("alphabet", self.alphabet)
        candidate_runs = changes.pop("candidate_runs", self.candidate_runs)
        decoded_candidate_runs = changes.pop("decoded_candidate_runs", self.decoded_candidate_runs)
        candidate_glyphs = changes.pop("candidate_glyphs", self.candidate_glyphs)
        decoded_candidate_glyphs = changes.pop(
            "decoded_candidate_glyphs", self.decoded_candidate_glyphs
        )
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(
            observations,
            eligible_seeds,
            aligned_seeds,
            accepted_seeds,
            initial_signatures,
            learned_signatures,
            approximate_signatures,
            alphabet,
            candidate_runs,
            decoded_candidate_runs,
            candidate_glyphs,
            decoded_candidate_glyphs,
        )

    @property
    def candidate_run_coverage(self) -> float:
        return self.decoded_candidate_runs / max(1, self.candidate_runs)

    @property
    def candidate_glyph_coverage(self) -> float:
        return self.decoded_candidate_glyphs / max(1, self.candidate_glyphs)


class StrokedTextRun:
    __slots__ = ("bbox", "drawing_indexes", "glyph_count")

    bbox: Rectangle
    drawing_indexes: tuple[int, ...]
    glyph_count: int

    __fields__: ClassVar[tuple[str, ...]] = ("bbox", "drawing_indexes", "glyph_count")
    __match_args__ = ("bbox", "drawing_indexes", "glyph_count")

    def __init__(self, bbox: Rectangle, drawing_indexes: tuple[int, ...], glyph_count: int) -> None:
        frozen_setattr(self, "bbox", bbox)
        frozen_setattr(self, "drawing_indexes", drawing_indexes)
        frozen_setattr(self, "glyph_count", glyph_count)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"bbox={self.bbox!r}, "
            f"drawing_indexes={self.drawing_indexes!r}, "
            f"glyph_count={self.glyph_count!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.bbox == other.bbox
            and self.drawing_indexes == other.drawing_indexes
            and self.glyph_count == other.glyph_count
        )

    def __hash__(self) -> int:
        return hash((self.bbox, self.drawing_indexes, self.glyph_count))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        bbox = changes.pop("bbox", self.bbox)
        drawing_indexes = changes.pop("drawing_indexes", self.drawing_indexes)
        glyph_count = changes.pop("glyph_count", self.glyph_count)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(bbox, drawing_indexes, glyph_count)


class PathRecord:
    __slots__ = ("index", "path", "bbox")

    index: int
    path: CapturedPath
    bbox: Rectangle

    __fields__: ClassVar[tuple[str, ...]] = ("index", "path", "bbox")
    __match_args__ = ("index", "path", "bbox")

    def __init__(self, index: int, path: CapturedPath, bbox: Rectangle) -> None:
        frozen_setattr(self, "index", index)
        frozen_setattr(self, "path", path)
        frozen_setattr(self, "bbox", bbox)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"index={self.index!r}, "
            f"path={self.path!r}, "
            f"bbox={self.bbox!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.index == other.index and self.path == other.path and self.bbox == other.bbox

    def __hash__(self) -> int:
        return hash((self.index, self.path, self.bbox))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        index = changes.pop("index", self.index)
        path = changes.pop("path", self.path)
        bbox = changes.pop("bbox", self.bbox)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(index, path, bbox)


Glyph: TypeAlias = tuple[PathRecord, ...]


class StrokedTextProfile:
    __slots__ = ("records", "run_profiles", "seed_runs")

    records: tuple[PathRecord, ...]
    run_profiles: tuple[StrokedTextRunProfile, ...]
    seed_runs: tuple[StrokedTextRun, ...]

    __fields__: ClassVar[tuple[str, ...]] = ("records", "run_profiles", "seed_runs")
    __match_args__ = ("records", "run_profiles", "seed_runs")

    def __init__(
        self,
        records: tuple[PathRecord, ...] = (),
        run_profiles: tuple[StrokedTextRunProfile, ...] = (),
        seed_runs: tuple[StrokedTextRun, ...] = (),
    ) -> None:
        frozen_setattr(self, "records", records)
        frozen_setattr(self, "run_profiles", run_profiles)
        frozen_setattr(self, "seed_runs", seed_runs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"records={self.records!r}, "
            f"run_profiles={self.run_profiles!r}, "
            f"seed_runs={self.seed_runs!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.records == other.records
            and self.run_profiles == other.run_profiles
            and self.seed_runs == other.seed_runs
        )

    def __hash__(self) -> int:
        return hash((self.records, self.run_profiles, self.seed_runs))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        records = changes.pop("records", self.records)
        run_profiles = changes.pop("run_profiles", self.run_profiles)
        seed_runs = changes.pop("seed_runs", self.seed_runs)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(records, run_profiles, seed_runs)


class SeedSample:
    __slots__ = ("seed", "text", "signatures")

    seed: StrokedTextSeed
    text: str
    signatures: tuple[GlyphSignature, ...]

    __fields__: ClassVar[tuple[str, ...]] = ("seed", "text", "signatures")
    __match_args__ = ("seed", "text", "signatures")

    def __init__(
        self,
        seed: StrokedTextSeed,
        text: str,
        signatures: tuple[GlyphSignature, ...],
    ) -> None:
        frozen_setattr(self, "seed", seed)
        frozen_setattr(self, "text", text)
        frozen_setattr(self, "signatures", signatures)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"seed={self.seed!r}, "
            f"text={self.text!r}, "
            f"signatures={self.signatures!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.seed == other.seed
            and self.text == other.text
            and self.signatures == other.signatures
        )

    def __hash__(self) -> int:
        return hash((self.seed, self.text, self.signatures))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        seed = changes.pop("seed", self.seed)
        text = changes.pop("text", self.text)
        signatures = changes.pop("signatures", self.signatures)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(seed, text, signatures)


class StrokedTextRunProfile:
    __slots__ = ("glyphs", "signatures", "bbox", "first_drawing", "last_drawing", "seed_run")

    glyphs: tuple[Glyph, ...]
    signatures: tuple[GlyphSignature | None, ...]
    bbox: Rectangle
    first_drawing: int
    last_drawing: int
    seed_run: StrokedTextRun | None

    __fields__: ClassVar[tuple[str, ...]] = (
        "glyphs",
        "signatures",
        "bbox",
        "first_drawing",
        "last_drawing",
        "seed_run",
    )
    __match_args__ = ("glyphs", "signatures", "bbox", "first_drawing", "last_drawing", "seed_run")

    def __init__(
        self,
        glyphs: tuple[Glyph, ...],
        signatures: tuple[GlyphSignature | None, ...],
        bbox: Rectangle,
        first_drawing: int,
        last_drawing: int,
        seed_run: StrokedTextRun | None,
    ) -> None:
        frozen_setattr(self, "glyphs", glyphs)
        frozen_setattr(self, "signatures", signatures)
        frozen_setattr(self, "bbox", bbox)
        frozen_setattr(self, "first_drawing", first_drawing)
        frozen_setattr(self, "last_drawing", last_drawing)
        frozen_setattr(self, "seed_run", seed_run)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"glyphs={self.glyphs!r}, "
            f"signatures={self.signatures!r}, "
            f"bbox={self.bbox!r}, "
            f"first_drawing={self.first_drawing!r}, "
            f"last_drawing={self.last_drawing!r}, "
            f"seed_run={self.seed_run!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.glyphs == other.glyphs
            and self.signatures == other.signatures
            and self.bbox == other.bbox
            and self.first_drawing == other.first_drawing
            and self.last_drawing == other.last_drawing
            and self.seed_run == other.seed_run
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.glyphs,
                self.signatures,
                self.bbox,
                self.first_drawing,
                self.last_drawing,
                self.seed_run,
            )
        )

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}")

    def __getstate__(self) -> list[Any]:
        return [getattr(self, name) for name in self.__fields__]

    def __setstate__(self, state: list[Any]) -> None:
        for name, value in zip(self.__fields__, state, strict=True):
            frozen_setattr(self, name, value)

    def __replace__(self, /, **changes: Any) -> Self:
        glyphs = changes.pop("glyphs", self.glyphs)
        signatures = changes.pop("signatures", self.signatures)
        bbox = changes.pop("bbox", self.bbox)
        first_drawing = changes.pop("first_drawing", self.first_drawing)
        last_drawing = changes.pop("last_drawing", self.last_drawing)
        seed_run = changes.pop("seed_run", self.seed_run)
        if changes:
            raise TypeError(f"__replace__() got unexpected keyword arguments {sorted(changes)!r}")
        return self.__class__(glyphs, signatures, bbox, first_drawing, last_drawing, seed_run)


def path_records(
    drawings: tuple[CapturedDrawing, ...], drawing_indexes: Iterable[int]
) -> tuple[PathRecord, ...]:
    records: list[PathRecord] = []
    for index in drawing_indexes:
        if not 0 <= index < len(drawings):
            continue
        drawing = drawings[index]
        if drawing.path is None:
            continue
        bbox = rect_tuple(drawing.rect)
        if bbox is None:
            continue
        records.append(PathRecord(index, drawing.path, bbox))
    return tuple(records)


def group_overlapping_x(
    records: Iterable[PathRecord],
) -> tuple[Glyph, ...]:
    groups: list[list[PathRecord]] = []
    bounds: list[list[float]] = []
    for record in records:
        x0, _, x1, _ = record.bbox
        if (
            groups
            and x0 <= bounds[-1][1] + STROKED_TEXT_GLYPH_OVERLAP_TOLERANCE
            and x1 >= bounds[-1][0] - STROKED_TEXT_GLYPH_OVERLAP_TOLERANCE
        ):
            groups[-1].append(record)
            bounds[-1][0] = min(bounds[-1][0], x0)
            bounds[-1][1] = max(bounds[-1][1], x1)
        else:
            groups.append([record])
            bounds.append([x0, x1])
    return tuple(tuple(group) for group in groups)


def glyph_signature(
    glyph: Glyph,
) -> GlyphSignature | None:
    points = tuple(
        point for record in glyph for subpath in record.path.subpaths for point in subpath.points
    )
    if not points:
        return None
    x0, y0, x1, y1 = points_bbox(points) or (0.0, 0.0, 0.0, 0.0)
    x_scale = max(0.02, x1 - x0)
    y_scale = max(0.02, y1 - y0)
    quantization = STROKED_TEXT_SIGNATURE_QUANTIZATION
    return tuple(
        tuple(
            (
                bool(subpath.closed),
                tuple(
                    (
                        round((point[0] - x0) / x_scale * quantization),
                        round((point[1] - y0) / y_scale * quantization),
                    )
                    for point in subpath.points
                ),
            )
            for subpath in record.path.subpaths
        )
        for record in glyph
    )


def seed_text(seed: StrokedTextSeed) -> str | None:
    text = seed.text.strip()
    if (
        seed.confidence < STROKED_TEXT_SEED_MIN_CONFIDENCE
        or not 2 <= len(text) <= STROKED_TEXT_MAX_TOKEN_CHARACTERS
        or any(character.isspace() for character in text)
        or any(character not in STROKED_TEXT_ALLOWED_CHARACTERS for character in text)
    ):
        return None
    return text


def seed_run_overlap(
    seed_box: Rectangle,
    run_box: Rectangle,
) -> float:
    intersection = bbox_intersection_area(seed_box, run_box)
    seed_area = max(0.01, bbox_area(seed_box))
    run_area = max(0.01, bbox_area(run_box))
    return intersection / min(seed_area, run_area)


def seed_samples(
    profile: StrokedTextProfile,
    seeds: tuple[StrokedTextSeed, ...],
) -> tuple[tuple[SeedSample, ...], int]:
    samples: list[SeedSample] = []
    eligible = 0
    records = profile.records
    direct_runs = {
        run.first_drawing: run for run in profile.run_profiles if run.seed_run is not None
    }
    records_by_y = tuple(
        sorted(records, key=lambda record: (record.bbox[1] + record.bbox[3]) * 0.5)
    )
    y_centers = tuple((record.bbox[1] + record.bbox[3]) * 0.5 for record in records_by_y)
    for seed in seeds:
        text = seed_text(seed)
        if text is None:
            continue
        eligible += 1
        direct = direct_runs.get(seed.sequence)
        if (
            direct is not None
            and len(direct.signatures) == len(text)
            and all(signature is not None for signature in direct.signatures)
            and seed_run_overlap(seed.bbox, direct.bbox) >= 0.50
        ):
            samples.append(
                SeedSample(
                    seed,
                    text,
                    tuple(signature for signature in direct.signatures if signature is not None),
                )
            )
            continue
        x0, y0, x1, y1 = seed.bbox
        padding = min(1.0, max(0.5, (y1 - y0) * 0.22))
        y_start = bisect_left(y_centers, y0 - padding)
        y_stop = bisect_right(y_centers, y1 + padding)
        hits = tuple(
            sorted(
                (
                    record
                    for record in records_by_y[y_start:y_stop]
                    if x0 - padding <= (record.bbox[0] + record.bbox[2]) * 0.5 <= x1 + padding
                ),
                key=lambda record: record.index,
            )
        )
        glyphs = group_overlapping_x(hits)
        if len(glyphs) != len(text):
            continue
        signatures = tuple(glyph_signature(glyph) for glyph in glyphs)
        if any(signature is None for signature in signatures):
            continue
        samples.append(
            SeedSample(
                seed,
                text,
                tuple(signature for signature in signatures if signature is not None),
            )
        )
    return tuple(samples), eligible


def consensus_mapping(
    samples: tuple[SeedSample, ...],
) -> tuple[dict[GlyphSignature, str], int, int]:
    votes: dict[GlyphSignature, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for sample in samples:
        for character, signature in zip(sample.text, sample.signatures, strict=True):
            votes[signature][character].add(sample.seed.sequence)

    mapping: dict[GlyphSignature, str] = {}
    for signature, character_votes in votes.items():
        ranked = sorted(
            ((len(sequences), character) for character, sequences in character_votes.items()),
            reverse=True,
        )
        winner_count, winner = ranked[0]
        total = sum(count for count, _ in ranked)
        runner_up = ranked[1][0] if len(ranked) > 1 else 0
        if winner_count >= 2 and winner_count / total >= 0.75 and winner_count > runner_up:
            mapping[signature] = winner
    initial = len(mapping)

    accepted_sequences: set[int] = set()
    while True:
        anchored_votes: dict[GlyphSignature, dict[str, set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for sample in samples:
            known = tuple(
                index for index, signature in enumerate(sample.signatures) if signature in mapping
            )
            if not known or any(
                mapping[sample.signatures[index]] != sample.text[index] for index in known
            ):
                continue
            alphanumeric_anchors = sum(sample.text[index].isalnum() for index in known)
            if alphanumeric_anchors < 1 or (len(known) < 2 and len(sample.text) < 3):
                continue
            accepted_sequences.add(sample.seed.sequence)
            for character, signature in zip(sample.text, sample.signatures, strict=True):
                anchored_votes[signature][character].add(sample.seed.sequence)

        additions = 0
        for signature, character_votes in anchored_votes.items():
            if signature in mapping:
                continue
            if len(character_votes) == 1:
                mapping[signature] = next(iter(character_votes))
                additions += 1
        if not additions:
            break
    return mapping, initial, len(accepted_sequences)


def required_bbox(boxes: Iterable[Rectangle]) -> Rectangle:
    bbox = bbox_union(boxes)
    if bbox is None:
        raise ValueError("cannot take the bounding box of an empty group")
    return bbox


def glyph_bbox(glyph: Glyph) -> Rectangle:
    return required_bbox(record.bbox for record in glyph)


def signature_distance(
    left: GlyphSignature,
    right: GlyphSignature,
) -> tuple[int, float] | None:
    if len(left) != len(right):
        return None
    maximum = 0
    total = 0
    coordinates = 0
    for left_drawing, right_drawing in zip(left, right, strict=True):
        if len(left_drawing) != len(right_drawing):
            return None
        for (left_closed, left_points), (right_closed, right_points) in zip(
            left_drawing,
            right_drawing,
            strict=True,
        ):
            if left_closed != right_closed or len(left_points) != len(right_points):
                return None
            for left_point, right_point in zip(left_points, right_points, strict=True):
                for left_value, right_value in zip(left_point, right_point, strict=True):
                    difference = abs(left_value - right_value)
                    maximum = max(maximum, difference)
                    total += difference
                    coordinates += 1
    return maximum, total / max(1, coordinates)


def signature_topology(signature: GlyphSignature) -> GlyphTopology:
    return tuple(
        tuple((closed, len(points)) for closed, points in drawing) for drawing in signature
    )


def expand_mapping(
    runs: tuple[StrokedTextRunProfile, ...],
    mapping: dict[GlyphSignature, str],
) -> int:
    additions = 0
    learned_by_topology: dict[GlyphTopology, list[tuple[GlyphSignature, str]]] = defaultdict(list)
    for learned_signature, character in mapping.items():
        learned_by_topology[signature_topology(learned_signature)].append(
            (learned_signature, character)
        )
    unknown = {
        signature
        for run in runs
        for signature in run.signatures
        if signature is not None and signature not in mapping
    }
    for signature in unknown:
        candidates = {
            character
            for learned_signature, character in learned_by_topology[signature_topology(signature)]
            if (distance := signature_distance(signature, learned_signature)) is not None
            and distance[0] <= STROKED_TEXT_SIGNATURE_MAX_COORDINATE_DISTANCE
            and distance[1] <= STROKED_TEXT_SIGNATURE_MAX_MEAN_DISTANCE
        }
        if len(candidates) == 1:
            mapping[signature] = candidates.pop()
            additions += 1
    return additions


def path_runs(
    records: tuple[PathRecord, ...],
) -> tuple[tuple[Glyph, ...], ...]:
    runs: list[tuple[Glyph, ...]] = []
    run: list[Glyph] = []
    glyph: list[PathRecord] = []
    glyph_x0 = 0.0
    glyph_x1 = 0.0
    line_y0 = 0.0
    line_y1 = 0.0
    previous_index = -2

    def finish_glyph() -> None:
        if glyph:
            run.append(tuple(glyph))
            glyph.clear()

    def finish_run() -> None:
        finish_glyph()
        if run:
            runs.append(tuple(run))
            run.clear()

    for record in records:
        if record.index != previous_index + 1:
            finish_run()
        if not glyph:
            glyph.append(record)
            glyph_x0 = record.bbox[0]
            glyph_x1 = record.bbox[2]
            line_y0 = record.bbox[1]
            line_y1 = record.bbox[3]
            previous_index = record.index
            continue

        line_height = max(0.2, line_y1 - line_y0)
        center_y = (record.bbox[1] + record.bbox[3]) * 0.5
        vertical_match = line_y0 - line_height * 0.30 <= center_y <= line_y1 + line_height * 0.30
        overlaps_glyph = (
            record.bbox[0] <= glyph_x1 + STROKED_TEXT_GLYPH_OVERLAP_TOLERANCE
            and record.bbox[2] >= glyph_x0 - STROKED_TEXT_GLYPH_OVERLAP_TOLERANCE
        )
        if vertical_match and overlaps_glyph:
            glyph.append(record)
            glyph_x0 = min(glyph_x0, record.bbox[0])
            glyph_x1 = max(glyph_x1, record.bbox[2])
            line_y0 = min(line_y0, record.bbox[1])
            line_y1 = max(line_y1, record.bbox[3])
        else:
            gap = record.bbox[0] - glyph_x1
            forward = record.bbox[0] >= glyph_x0 - STROKED_TEXT_GLYPH_OVERLAP_TOLERANCE
            maximum_gap = max(1.5, line_height * 0.65)
            if (
                vertical_match
                and forward
                and -STROKED_TEXT_GLYPH_OVERLAP_TOLERANCE <= gap <= maximum_gap
            ):
                finish_glyph()
                glyph.append(record)
                glyph_x0 = record.bbox[0]
                glyph_x1 = record.bbox[2]
                line_y0 = min(line_y0, record.bbox[1])
                line_y1 = max(line_y1, record.bbox[3])
            else:
                finish_run()
                glyph.append(record)
                glyph_x0 = record.bbox[0]
                glyph_x1 = record.bbox[2]
                line_y0 = record.bbox[1]
                line_y1 = record.bbox[3]
        previous_index = record.index
    finish_run()
    return tuple(runs)


STROKED_TEXT_ISOLATED_MIN_ASPECT_RATIO = 0.12


def stroked_text_isolated_runs(profile: StrokedTextProfile) -> tuple[StrokedTextRun, ...]:
    isolated: list[StrokedTextRun] = []
    for run in profile.run_profiles:
        glyphs = run.glyphs
        if len(glyphs) != 1:
            continue
        box = glyph_bbox(glyphs[0])
        width = box[2] - box[0]
        height = box[3] - box[1]
        if not (
            STROKED_TEXT_SEED_RUN_MIN_HEIGHT <= height <= STROKED_TEXT_SEED_RUN_MAX_HEIGHT
            and width <= STROKED_TEXT_SEED_RUN_MAX_WIDTH
        ):
            continue
        if min(width, height) < max(width, height) * STROKED_TEXT_ISOLATED_MIN_ASPECT_RATIO:
            continue
        isolated.append(
            StrokedTextRun(
                bbox=box,
                drawing_indexes=tuple(record.index for record in glyphs[0]),
                glyph_count=1,
            )
        )
    return tuple(isolated)


def stroked_text_seed_run(
    glyphs: tuple[Glyph, ...],
) -> StrokedTextRun | None:
    if not 2 <= len(glyphs) <= STROKED_TEXT_MAX_TOKEN_CHARACTERS:
        return None
    bbox = required_bbox(glyph_bbox(glyph) for glyph in glyphs)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if not (
        STROKED_TEXT_SEED_RUN_MIN_HEIGHT <= height <= STROKED_TEXT_SEED_RUN_MAX_HEIGHT
        and width <= STROKED_TEXT_SEED_RUN_MAX_WIDTH
    ):
        return None
    return StrokedTextRun(
        bbox=bbox,
        drawing_indexes=tuple(record.index for glyph in glyphs for record in glyph),
        glyph_count=len(glyphs),
    )


def profile_stroked_text(
    drawings: tuple[CapturedDrawing, ...],
    drawing_indexes: Iterable[int],
) -> StrokedTextProfile:
    records = path_records(drawings, drawing_indexes)
    if not records:
        return StrokedTextProfile()
    runs = path_runs(records)
    run_profiles: list[StrokedTextRunProfile] = []
    for run in runs:
        signatures = tuple(glyph_signature(glyph) for glyph in run)
        bbox = required_bbox(glyph_bbox(glyph) for glyph in run)
        run_profiles.append(
            StrokedTextRunProfile(
                glyphs=run,
                signatures=signatures,
                bbox=bbox,
                first_drawing=run[0][0].index,
                last_drawing=run[-1][-1].index,
                seed_run=stroked_text_seed_run(run),
            )
        )
    return StrokedTextProfile(
        records=records,
        run_profiles=tuple(run_profiles),
        seed_runs=tuple(run.seed_run for run in run_profiles if run.seed_run is not None),
    )


def decode_runs(
    runs: tuple[StrokedTextRunProfile, ...],
    mapping: dict[GlyphSignature, str],
) -> tuple[StrokedTextObservation, ...]:
    observations: list[StrokedTextObservation] = []
    for run in runs:
        if not 1 <= len(run.signatures) <= STROKED_TEXT_MAX_TOKEN_CHARACTERS:
            continue
        signatures = run.signatures
        if any(signature is None or signature not in mapping for signature in signatures):
            continue
        text = "".join(mapping[signature] for signature in signatures if signature is not None)
        if not any(character.isalnum() for character in text):
            continue
        bbox = run.bbox
        if len(signatures) == 1 and (bbox[2] - bbox[0]) / max(0.01, bbox[3] - bbox[1]) < (
            STROKED_TEXT_SINGLE_GLYPH_MIN_ASPECT_RATIO
        ):
            continue
        observations.append(
            StrokedTextObservation(
                text=text,
                bbox=bbox,
                first_drawing=run.first_drawing,
                last_drawing=run.last_drawing,
            )
        )
    return tuple(observations)


def decode_with_mapping(
    profile: StrokedTextProfile,
    mapping: dict[GlyphSignature, str],
) -> tuple[tuple[StrokedTextObservation, ...], int, int, int, int, int]:
    approximate = expand_mapping(profile.run_profiles, mapping)
    observations = decode_runs(profile.run_profiles, mapping)
    decoded_spans = {
        (observation.first_drawing, observation.last_drawing) for observation in observations
    }
    candidates = profile.seed_runs
    decoded_candidates = tuple(
        candidate
        for candidate in candidates
        if (candidate.drawing_indexes[0], candidate.drawing_indexes[-1]) in decoded_spans
    )
    return (
        observations,
        approximate,
        len(candidates),
        len(decoded_candidates),
        sum(candidate.glyph_count for candidate in candidates),
        sum(candidate.glyph_count for candidate in decoded_candidates),
    )


def decoded_profile(
    profile: StrokedTextProfile,
    mapping: dict[GlyphSignature, str],
    *,
    eligible_seeds: int = 0,
    aligned_seeds: int = 0,
    accepted_seeds: int = 0,
    initial_signatures: int = 0,
) -> StrokedTextDecode:
    observations, approximate, candidates, decoded_candidates, glyphs, decoded_glyphs = (
        decode_with_mapping(profile, mapping)
    )
    return StrokedTextDecode(
        observations=observations,
        eligible_seeds=eligible_seeds,
        aligned_seeds=aligned_seeds,
        accepted_seeds=accepted_seeds,
        initial_signatures=initial_signatures,
        learned_signatures=len(mapping),
        approximate_signatures=approximate,
        alphabet=tuple(mapping.items()),
        candidate_runs=candidates,
        decoded_candidate_runs=decoded_candidates,
        candidate_glyphs=glyphs,
        decoded_candidate_glyphs=decoded_glyphs,
    )


def decode_stroked_text_profile(
    profile: StrokedTextProfile,
    seeds: tuple[StrokedTextSeed, ...],
) -> StrokedTextDecode:
    if not profile.records or not seeds:
        return StrokedTextDecode()
    samples, eligible = seed_samples(profile, seeds)
    if not samples:
        return StrokedTextDecode(eligible_seeds=eligible)
    mapping, initial, accepted = consensus_mapping(samples)
    if not mapping:
        return StrokedTextDecode(
            eligible_seeds=eligible,
            aligned_seeds=len(samples),
        )
    return decoded_profile(
        profile,
        mapping,
        eligible_seeds=eligible,
        aligned_seeds=len(samples),
        accepted_seeds=accepted,
        initial_signatures=initial,
    )


def decode_stroked_text_profile_with_supplemental_seeds(
    profile: StrokedTextProfile,
    primary_seeds: tuple[StrokedTextSeed, ...],
    supplemental_seeds: tuple[StrokedTextSeed, ...],
) -> StrokedTextDecode:
    if not supplemental_seeds:
        return decode_stroked_text_profile(profile, primary_seeds)
    if not profile.records:
        return StrokedTextDecode()

    primary_samples, primary_eligible = seed_samples(profile, primary_seeds)
    supplemental_samples, supplemental_eligible = seed_samples(profile, supplemental_seeds)
    samples = (*primary_samples, *supplemental_samples)
    eligible = primary_eligible + supplemental_eligible
    if not samples:
        return StrokedTextDecode(eligible_seeds=eligible)

    primary_mapping: dict[GlyphSignature, str] = {}
    if primary_samples:
        primary_mapping, _, _ = consensus_mapping(primary_samples)
    mapping, initial, accepted = consensus_mapping(samples)
    mapping.update(primary_mapping)
    if not mapping:
        return StrokedTextDecode(
            eligible_seeds=eligible,
            aligned_seeds=len(samples),
        )

    return decoded_profile(
        profile,
        mapping,
        eligible_seeds=eligible,
        aligned_seeds=len(samples),
        accepted_seeds=accepted,
        initial_signatures=initial,
    )


def decode_stroked_text_profile_with_alphabet(
    profile: StrokedTextProfile,
    alphabet: Mapping[GlyphSignature, str] | Iterable[tuple[GlyphSignature, str]],
) -> StrokedTextDecode:
    mapping = dict(alphabet)
    if not profile.records or not mapping:
        return StrokedTextDecode()
    return decoded_profile(profile, mapping, initial_signatures=len(mapping))
