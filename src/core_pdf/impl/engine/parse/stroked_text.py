# SPDX-License-Identifier: AGPL-3.0-only
"""Recover repeated single-line text from flattened vector paths.

Some CAD exporters convert every character to a handful of stroked paths and
discard the PDF text objects.  Raster OCR can seed a page-local alphabet for
those paths: repeated, conflict-free shapes are learned first, then anchored
OCR words safely teach the remaining shapes.  Fully recognized path runs can
thereafter be emitted without another OCR pass.
"""

from __future__ import annotations

import string
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, TypeAlias

from core_pdf.impl.engine.model.geometry import bbox_area, bbox_intersection_area, rect_tuple
from core_pdf.impl.types import Rectangle

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


@dataclass(frozen=True, slots=True)
class StrokedTextSeed:
    """One OCR token that may label a sequence of vector glyphs."""

    text: str
    bbox: Rectangle
    confidence: float
    sequence: int


@dataclass(frozen=True, slots=True)
class StrokedTextObservation:
    """Text decoded directly from a consecutive vector-path run."""

    text: str
    bbox: Rectangle
    first_drawing: int
    last_drawing: int
    confidence: float = 96.0


@dataclass(frozen=True, slots=True)
class StrokedTextDecode:
    """Page-local decoder output and compact diagnostics."""

    observations: tuple[StrokedTextObservation, ...] = ()
    eligible_seeds: int = 0
    aligned_seeds: int = 0
    accepted_seeds: int = 0
    initial_signatures: int = 0
    learned_signatures: int = 0
    approximate_signatures: int = 0
    alphabet: tuple[tuple[GlyphSignature, str], ...] = ()
    candidate_runs: int = 0
    decoded_candidate_runs: int = 0
    candidate_glyphs: int = 0
    decoded_candidate_glyphs: int = 0

    @property
    def candidate_run_coverage(self) -> float:
        return self.decoded_candidate_runs / max(1, self.candidate_runs)

    @property
    def candidate_glyph_coverage(self) -> float:
        return self.decoded_candidate_glyphs / max(1, self.candidate_glyphs)


@dataclass(frozen=True, slots=True)
class StrokedTextRun:
    """One compact horizontal path run suitable for seeding raster OCR."""

    bbox: Rectangle
    drawing_indexes: tuple[int, ...]
    glyph_count: int


@dataclass(frozen=True, slots=True)
class internal_PathRecord:
    index: int
    drawing: Any
    bbox: Rectangle


internal_Glyph: TypeAlias = tuple[internal_PathRecord, ...]


@dataclass(frozen=True, slots=True)
class StrokedTextProfile:
    """Reusable segmentation and signatures for one flattened vector font layer."""

    records: tuple[internal_PathRecord, ...] = ()
    runs: tuple[tuple[internal_Glyph, ...], ...] = ()
    run_profiles: tuple[internal_StrokedTextRunProfile, ...] = ()
    seed_runs: tuple[StrokedTextRun, ...] = ()
    signatures: tuple[tuple[tuple[int, ...], GlyphSignature | None], ...] = ()

    def signature_cache(self) -> dict[tuple[int, ...], GlyphSignature | None]:
        return dict(self.signatures)


@dataclass(frozen=True, slots=True)
class internal_SeedSample:
    seed: StrokedTextSeed
    text: str
    signatures: tuple[GlyphSignature, ...]


@dataclass(frozen=True, slots=True)
class internal_StrokedTextRunProfile:
    """Geometry and signatures that remain invariant across OCR seed sets."""

    glyphs: tuple[internal_Glyph, ...]
    signatures: tuple[GlyphSignature | None, ...]
    bbox: Rectangle
    first_drawing: int
    last_drawing: int
    seed_run: StrokedTextRun | None


def internal_path_records(
    drawings: tuple[Any, ...], drawing_indexes: Iterable[int]
) -> tuple[internal_PathRecord, ...]:
    records: list[internal_PathRecord] = []
    for index in drawing_indexes:
        if not 0 <= index < len(drawings):
            continue
        drawing = drawings[index]
        if getattr(drawing, "path", None) is None:
            continue
        bbox = rect_tuple(getattr(drawing, "rect", None))
        if bbox is None:
            continue
        records.append(internal_PathRecord(index, drawing, bbox))
    return tuple(records)


def internal_group_overlapping_x(
    records: Iterable[internal_PathRecord],
) -> tuple[internal_Glyph, ...]:
    """Group consecutive path components whose horizontal extents overlap."""
    groups: list[list[internal_PathRecord]] = []
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


def internal_glyph_signature(
    glyph: internal_Glyph,
    cache: dict[tuple[int, ...], GlyphSignature | None] | None = None,
) -> GlyphSignature | None:
    """Return a translation- and scale-independent polyline signature."""
    cache_key = tuple(record.index for record in glyph)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    points = tuple(
        point
        for record in glyph
        for subpath in record.drawing.path.subpaths
        for point in subpath.points
    )
    if not points:
        if cache is not None:
            cache[cache_key] = None
        return None
    x0 = min(point[0] for point in points)
    y0 = min(point[1] for point in points)
    x1 = max(point[0] for point in points)
    y1 = max(point[1] for point in points)
    x_scale = max(0.02, x1 - x0)
    y_scale = max(0.02, y1 - y0)
    quantization = STROKED_TEXT_SIGNATURE_QUANTIZATION
    signature = tuple(
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
            for subpath in record.drawing.path.subpaths
        )
        for record in glyph
    )
    if cache is not None:
        cache[cache_key] = signature
    return signature


def internal_seed_text(seed: StrokedTextSeed) -> str | None:
    text = seed.text.strip()
    if (
        seed.confidence < STROKED_TEXT_SEED_MIN_CONFIDENCE
        or not 2 <= len(text) <= STROKED_TEXT_MAX_TOKEN_CHARACTERS
        or any(character.isspace() for character in text)
        or any(character not in STROKED_TEXT_ALLOWED_CHARACTERS for character in text)
    ):
        return None
    return text


def internal_seed_run_overlap(
    seed_box: Rectangle,
    run_box: Rectangle,
) -> float:
    """Return containment-style overlap for a remapped packed OCR token."""
    intersection = bbox_intersection_area(seed_box, run_box)
    seed_area = max(0.01, bbox_area(seed_box))
    run_area = max(0.01, bbox_area(run_box))
    return intersection / min(seed_area, run_area)


def internal_seed_samples(
    profile: StrokedTextProfile,
    seeds: tuple[StrokedTextSeed, ...],
    signature_cache: dict[tuple[int, ...], GlyphSignature | None] | None = None,
) -> tuple[tuple[internal_SeedSample, ...], int]:
    samples: list[internal_SeedSample] = []
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
        text = internal_seed_text(seed)
        if text is None:
            continue
        eligible += 1
        direct = direct_runs.get(seed.sequence)
        if (
            direct is not None
            and len(direct.signatures) == len(text)
            and all(signature is not None for signature in direct.signatures)
            and internal_seed_run_overlap(seed.bbox, direct.bbox) >= 0.50
        ):
            samples.append(
                internal_SeedSample(
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
        glyphs = internal_group_overlapping_x(hits)
        if len(glyphs) != len(text):
            continue
        signatures = tuple(internal_glyph_signature(glyph, signature_cache) for glyph in glyphs)
        if any(signature is None for signature in signatures):
            continue
        samples.append(
            internal_SeedSample(
                seed,
                text,
                tuple(signature for signature in signatures if signature is not None),
            )
        )
    return tuple(samples), eligible


def internal_consensus_mapping(
    samples: tuple[internal_SeedSample, ...],
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

    # A consensus alphabet validates otherwise unique glyphs inside anchored
    # words.  A single anchor is sufficient only for tokens of at least three
    # characters, and it must be alphanumeric rather than punctuation.
    accepted_sequences: set[int] = set()
    for _ in range(8):
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
            ranked = sorted(
                ((len(sequences), character) for character, sequences in character_votes.items()),
                reverse=True,
            )
            winner_count, winner = ranked[0]
            if winner_count == sum(count for count, _ in ranked):
                mapping[signature] = winner
                additions += 1
        if not additions:
            break
    return mapping, initial, len(accepted_sequences)


def internal_glyph_bbox(glyph: internal_Glyph) -> Rectangle:
    return (
        min(record.bbox[0] for record in glyph),
        min(record.bbox[1] for record in glyph),
        max(record.bbox[2] for record in glyph),
        max(record.bbox[3] for record in glyph),
    )


def internal_signature_distance(
    left: GlyphSignature,
    right: GlyphSignature,
) -> tuple[int, float] | None:
    """Compare signatures only when their path topology is identical."""
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


def internal_signature_topology(signature: GlyphSignature) -> GlyphTopology:
    return tuple(
        tuple((closed, len(points)) for closed, points in drawing) for drawing in signature
    )


def internal_expand_mapping(
    runs: tuple[internal_StrokedTextRunProfile, ...],
    mapping: dict[GlyphSignature, str],
) -> int:
    """Map small quantization variants only when their character is unique."""
    additions = 0
    learned_by_topology: dict[GlyphTopology, list[tuple[GlyphSignature, str]]] = defaultdict(list)
    for learned_signature, character in mapping.items():
        learned_by_topology[internal_signature_topology(learned_signature)].append(
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
            for learned_signature, character in learned_by_topology[
                internal_signature_topology(signature)
            ]
            if (distance := internal_signature_distance(signature, learned_signature)) is not None
            and distance[0] <= STROKED_TEXT_SIGNATURE_MAX_COORDINATE_DISTANCE
            and distance[1] <= STROKED_TEXT_SIGNATURE_MAX_MEAN_DISTANCE
        }
        if len(candidates) == 1:
            mapping[signature] = candidates.pop()
            additions += 1
    return additions


def internal_path_runs(
    records: tuple[internal_PathRecord, ...],
) -> tuple[tuple[internal_Glyph, ...], ...]:
    """Segment capture-order paths into horizontal glyph runs."""
    runs: list[tuple[internal_Glyph, ...]] = []
    run: list[internal_Glyph] = []
    glyph: list[internal_PathRecord] = []
    glyph_x0 = 0.0
    glyph_x1 = 0.0
    # Running vertical extent of every record appended to the current run;
    # records are only ever added, so the min/max accumulate monotonically.
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
    """Return single-glyph runs the seed packer skips (pin numbers, lone digits).

    Seed packing requires at least two glyphs per run, so isolated labels are
    never rasterized for OCR.  Collect the glyph-sized singles here so a
    supplemental montage can show them to OCR; near-degenerate boxes (wire
    stubs, junction dashes) stay excluded via the aspect gate.
    """
    isolated: list[StrokedTextRun] = []
    for glyphs in profile.runs:
        if len(glyphs) != 1:
            continue
        box = internal_glyph_bbox(glyphs[0])
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


def internal_stroked_text_seed_run(
    glyphs: tuple[internal_Glyph, ...],
) -> StrokedTextRun | None:
    if not 2 <= len(glyphs) <= STROKED_TEXT_MAX_TOKEN_CHARACTERS:
        return None
    boxes = tuple(internal_glyph_bbox(glyph) for glyph in glyphs)
    bbox = (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )
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
    drawings: tuple[Any, ...],
    drawing_indexes: Iterable[int],
) -> StrokedTextProfile:
    """Segment and fingerprint one stroke-text layer once for all later stages."""
    records = internal_path_records(drawings, drawing_indexes)
    if not records:
        return StrokedTextProfile()
    runs = internal_path_runs(records)
    signature_cache: dict[tuple[int, ...], GlyphSignature | None] = {}
    run_profiles: list[internal_StrokedTextRunProfile] = []
    for run in runs:
        signatures = tuple(internal_glyph_signature(glyph, signature_cache) for glyph in run)
        boxes = tuple(internal_glyph_bbox(glyph) for glyph in run)
        bbox = (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )
        run_profiles.append(
            internal_StrokedTextRunProfile(
                glyphs=run,
                signatures=signatures,
                bbox=bbox,
                first_drawing=run[0][0].index,
                last_drawing=run[-1][-1].index,
                seed_run=internal_stroked_text_seed_run(run),
            )
        )
    return StrokedTextProfile(
        records=records,
        runs=runs,
        run_profiles=tuple(run_profiles),
        seed_runs=tuple(run.seed_run for run in run_profiles if run.seed_run is not None),
        signatures=tuple(signature_cache.items()),
    )


def internal_decode_runs(
    runs: tuple[internal_StrokedTextRunProfile, ...],
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
            # Isolated vertical wires can share the normalized topology of a
            # single-line-font ``I``. Real standalone glyphs retain measurable
            # horizontal ink even when the character itself is narrow.
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


def internal_decode_with_mapping(
    profile: StrokedTextProfile,
    mapping: dict[GlyphSignature, str],
) -> tuple[tuple[StrokedTextObservation, ...], int, int, int, int, int]:
    approximate = internal_expand_mapping(profile.run_profiles, mapping)
    observations = internal_decode_runs(profile.run_profiles, mapping)
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


def decode_stroked_text_profile(
    profile: StrokedTextProfile,
    seeds: tuple[StrokedTextSeed, ...],
) -> StrokedTextDecode:
    """Learn and decode a page-local font from an existing structural profile."""
    if not profile.records or not seeds:
        return StrokedTextDecode()
    signature_cache = profile.signature_cache()
    samples, eligible = internal_seed_samples(profile, seeds, signature_cache)
    if not samples:
        return StrokedTextDecode(eligible_seeds=eligible)
    mapping, initial, accepted = internal_consensus_mapping(samples)
    if not mapping:
        return StrokedTextDecode(
            eligible_seeds=eligible,
            aligned_seeds=len(samples),
        )
    learned = len(mapping)
    alphabet = tuple(mapping.items())
    observations, approximate, candidates, decoded_candidates, glyphs, decoded_glyphs = (
        internal_decode_with_mapping(profile, mapping)
    )
    return StrokedTextDecode(
        observations=observations,
        eligible_seeds=eligible,
        aligned_seeds=len(samples),
        accepted_seeds=accepted,
        initial_signatures=initial,
        learned_signatures=learned,
        approximate_signatures=approximate,
        alphabet=alphabet,
        candidate_runs=candidates,
        decoded_candidate_runs=decoded_candidates,
        candidate_glyphs=glyphs,
        decoded_candidate_glyphs=decoded_glyphs,
    )


def decode_stroked_text_profile_with_supplemental_seeds(
    profile: StrokedTextProfile,
    primary_seeds: tuple[StrokedTextSeed, ...],
    supplemental_seeds: tuple[StrokedTextSeed, ...],
) -> StrokedTextDecode:
    """Extend a primary alphabet monotonically and decode the profile once.

    Character-level OCR can expose complete packed cells that the word iterator
    omitted.  Its consensus may add signatures, but never replaces a mapping
    supported by the primary word observations.
    """
    if not supplemental_seeds:
        return decode_stroked_text_profile(profile, primary_seeds)
    if not profile.records:
        return StrokedTextDecode()

    signature_cache = profile.signature_cache()
    primary_samples, primary_eligible = internal_seed_samples(
        profile,
        primary_seeds,
        signature_cache,
    )
    supplemental_samples, supplemental_eligible = internal_seed_samples(
        profile,
        supplemental_seeds,
        signature_cache,
    )
    samples = (*primary_samples, *supplemental_samples)
    eligible = primary_eligible + supplemental_eligible
    if not samples:
        return StrokedTextDecode(eligible_seeds=eligible)

    primary_mapping: dict[GlyphSignature, str] = {}
    if primary_samples:
        primary_mapping, _, _ = internal_consensus_mapping(primary_samples)
    mapping, initial, accepted = internal_consensus_mapping(samples)
    # Primary word recognition wins conflicts. Supplemental symbols can only
    # make the exact alphabet larger, so structural coverage cannot regress.
    mapping.update(primary_mapping)
    if not mapping:
        return StrokedTextDecode(
            eligible_seeds=eligible,
            aligned_seeds=len(samples),
        )

    learned = len(mapping)
    alphabet = tuple(mapping.items())
    observations, approximate, candidates, decoded_candidates, glyphs, decoded_glyphs = (
        internal_decode_with_mapping(profile, mapping)
    )
    return StrokedTextDecode(
        observations=observations,
        eligible_seeds=eligible,
        aligned_seeds=len(samples),
        accepted_seeds=accepted,
        initial_signatures=initial,
        learned_signatures=learned,
        approximate_signatures=approximate,
        alphabet=alphabet,
        candidate_runs=candidates,
        decoded_candidate_runs=decoded_candidates,
        candidate_glyphs=glyphs,
        decoded_candidate_glyphs=decoded_glyphs,
    )


def decode_stroked_text_profile_with_alphabet(
    profile: StrokedTextProfile,
    alphabet: Mapping[GlyphSignature, str] | Iterable[tuple[GlyphSignature, str]],
) -> StrokedTextDecode:
    """Apply an exact document alphabet to an existing structural profile."""
    mapping = dict(alphabet)
    if not profile.records or not mapping:
        return StrokedTextDecode()
    exact_alphabet = tuple(mapping.items())
    learned = len(mapping)
    observations, approximate, candidates, decoded_candidates, glyphs, decoded_glyphs = (
        internal_decode_with_mapping(profile, mapping)
    )
    return StrokedTextDecode(
        observations=observations,
        initial_signatures=learned,
        learned_signatures=learned,
        approximate_signatures=approximate,
        alphabet=exact_alphabet,
        candidate_runs=candidates,
        decoded_candidate_runs=decoded_candidates,
        candidate_glyphs=glyphs,
        decoded_candidate_glyphs=decoded_glyphs,
    )
