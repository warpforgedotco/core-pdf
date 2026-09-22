from copy import replace

import pytest

from core_pdf.impl._impl.capture.records import CapturedDrawing, CapturedPath, CapturedSubpath
from core_pdf_ocr.impl.extract.ocr import strokes

SHAPES = {
    "A": ((0, 4), (1, 0), (2, 4)),
    "B": ((0, 0), (2, 0), (2, 4), (0, 4)),
    "C": ((2, 0), (0, 0), (0, 4), (2, 4)),
}


def internal_profile(*words: str) -> strokes.StrokedTextProfile:
    drawings = tuple(
        CapturedDrawing(
            len(word) * row + column,
            None,
            None,
            kind="stroke",
            path=CapturedPath(
                [CapturedSubpath([(x + column * 3, y + row * 10) for x, y in SHAPES[character]])]
            ),
        )
        for row, word in enumerate(words)
        for column, character in enumerate(word)
    )
    return strokes.profile_stroked_text(drawings, range(len(drawings)))


def internal_seed(
    profile: strokes.StrokedTextProfile, index: int, text: str
) -> strokes.StrokedTextSeed:
    run = profile.run_profiles[index]
    return strokes.StrokedTextSeed(text, run.bbox, 95, run.first_drawing)


@pytest.mark.parametrize(
    ("text", "confidence"),
    [("A", 95), ("A B", 95), ("A@", 95), ("A" * 13, 95), ("AB", 84.9), ("", 95)],
)
def test_invalid_ocr_tokens_are_ineligible(text: str, confidence: float) -> None:
    profile = internal_profile("AB")
    seed = replace(internal_seed(profile, 0, text), confidence=confidence)
    decoded = strokes.decode_stroked_text_profile(profile, (seed,))
    assert decoded.eligible_seeds == 0
    assert decoded.observations == ()


def test_two_independent_votes_decode_unseeded_repetitions() -> None:
    profile = internal_profile("AB", "AB", "BA")
    seeds = tuple(internal_seed(profile, index, " AB ") for index in (0, 1))
    decoded = strokes.decode_stroked_text_profile(profile, seeds)
    assert [o.text for o in decoded.observations] == ["AB", "AB", "BA"]
    assert decoded.eligible_seeds == decoded.aligned_seeds == decoded.accepted_seeds == 2
    assert decoded.initial_signatures == decoded.learned_signatures == 2
    assert decoded.candidate_runs == decoded.decoded_candidate_runs == 3
    assert decoded.candidate_glyphs == decoded.decoded_candidate_glyphs == 6
    assert decoded.candidate_run_coverage == decoded.candidate_glyph_coverage == 1
    assert [o.bbox for o in decoded.observations] == [(0, 0, 5, 4), (0, 10, 5, 14), (0, 20, 5, 24)]
    assert [(o.first_drawing, o.last_drawing) for o in decoded.observations] == [
        (0, 1),
        (2, 3),
        (4, 5),
    ]


def test_duplicate_seed_sequence_cannot_supply_independent_votes() -> None:
    profile = internal_profile("AB")
    seed = internal_seed(profile, 0, "AB")
    decoded = strokes.decode_stroked_text_profile(profile, (seed, seed, seed))
    assert decoded.aligned_seeds == 3
    assert decoded.alphabet == ()
    assert decoded.observations == ()


@pytest.mark.parametrize(
    ("labels", "expected"), [(("AB", "AB", "AB", "AC"), "AB"), (("AB", "AB", "AC", "AC"), None)]
)
def test_consensus_requires_three_quarters_and_rejects_ties(
    labels: tuple[str, ...], expected: str | None
) -> None:
    profile = internal_profile(*("AB" for _ in labels))
    seeds = tuple(internal_seed(profile, i, label) for i, label in enumerate(labels))
    decoded = strokes.decode_stroked_text_profile(profile, seeds)
    assert [o.text for o in decoded.observations] == ([expected] * len(labels) if expected else [])


def test_anchored_word_teaches_unique_glyph_without_overwriting_consensus() -> None:
    profile = internal_profile("AB", "AB", "ABC", "CA")
    seeds = tuple(internal_seed(profile, i, text) for i, text in enumerate(("AB", "AB", "ABC")))
    decoded = strokes.decode_stroked_text_profile(profile, seeds)
    assert decoded.initial_signatures == 2
    assert decoded.learned_signatures == 3
    assert [o.text for o in decoded.observations] == ["AB", "AB", "ABC", "CA"]


def test_supplemental_conflicting_labels_never_replace_primary_alphabet() -> None:
    profile = internal_profile("AB", "AB", "ABC", "ABC", "BA")
    primary = (internal_seed(profile, 0, "AB"), internal_seed(profile, 1, "AB"))
    supplemental = (internal_seed(profile, 2, "AXC"), internal_seed(profile, 3, "AXC"))
    decoded = strokes.decode_stroked_text_profile_with_supplemental_seeds(
        profile, primary, supplemental
    )
    assert [o.text for o in decoded.observations] == ["AB", "AB", "ABC", "ABC", "BA"]
    assert decoded.learned_signatures == 3
    assert decoded.eligible_seeds == decoded.aligned_seeds == 4
    assert strokes.decode_stroked_text_profile_with_supplemental_seeds(profile, primary, ()) == (
        strokes.decode_stroked_text_profile(profile, primary)
    )


def test_document_alphabet_decodes_without_mutating_caller_mapping() -> None:
    profile = internal_profile("AB", "AB")
    learned = strokes.decode_stroked_text_profile(
        profile,
        (
            internal_seed(profile, 0, "AB"),
            internal_seed(profile, 1, "AB"),
        ),
    )
    alphabet = dict(learned.alphabet)
    target = internal_profile("BA")
    decoded = strokes.decode_stroked_text_profile_with_alphabet(target, alphabet.items())
    assert [o.text for o in decoded.observations] == ["BA"]
    assert decoded.eligible_seeds == decoded.aligned_seeds == decoded.accepted_seeds == 0
    assert decoded.initial_signatures == 2
    assert alphabet == dict(learned.alphabet)


def test_seed_alignment_can_use_geometry_when_sequence_is_unavailable() -> None:
    profile = internal_profile("AB", "AB")
    seeds = tuple(replace(internal_seed(profile, i, "AB"), sequence=999 + i) for i in range(2))
    decoded = strokes.decode_stroked_text_profile(profile, seeds)
    assert [o.text for o in decoded.observations] == ["AB", "AB"]
    miss = replace(seeds[0], bbox=(100, 100, 110, 110))
    decoded = strokes.decode_stroked_text_profile(profile, (miss,))
    assert decoded.eligible_seeds == 1
    assert decoded.aligned_seeds == 0


@pytest.mark.parametrize("empty_profile", [True, False])
def test_missing_evidence_returns_empty_decodes(empty_profile: bool) -> None:
    profile = strokes.StrokedTextProfile() if empty_profile else internal_profile("AB")
    assert strokes.decode_stroked_text_profile(profile, ()) == strokes.StrokedTextDecode()
    assert (
        strokes.decode_stroked_text_profile_with_alphabet(profile, {})
        == strokes.StrokedTextDecode()
    )
    assert strokes.StrokedTextDecode().candidate_run_coverage == 0
    assert strokes.StrokedTextDecode().candidate_glyph_coverage == 0


def test_profile_skips_missing_paths_bounds_and_invalid_indexes() -> None:
    empty = CapturedDrawing(0, None, None)
    pathless = replace(empty, path=CapturedPath())
    assert (
        strokes.profile_stroked_text((empty, pathless), (-1, 0, 1, 99))
        == strokes.StrokedTextProfile()
    )
    with pytest.raises(ValueError, match="empty group"):
        strokes.internal_required_bbox(())


def test_isolated_glyphs_are_separate_from_multi_glyph_seed_runs() -> None:
    profile = internal_profile("A", "AB", "C")
    isolated = strokes.stroked_text_isolated_runs(profile)
    assert [(r.glyph_count, r.drawing_indexes) for r in isolated] == [(1, (0,)), (1, (3,))]
    assert [(r.glyph_count, r.drawing_indexes) for r in profile.seed_runs] == [(2, (1, 2))]


def test_conflicting_anchored_words_do_not_teach_an_ambiguous_glyph() -> None:
    profile = internal_profile("AB", "AB", "ABC", "ABC")
    seeds = tuple(
        internal_seed(profile, i, word) for i, word in enumerate(("AB", "AB", "ABC", "ABX"))
    )
    decoded = strokes.decode_stroked_text_profile(profile, seeds)
    assert decoded.learned_signatures == 2
    assert [o.text for o in decoded.observations] == ["AB", "AB"]


def internal_single(
    points: list[tuple[float, float]], *, closed: bool = False
) -> strokes.StrokedTextProfile:
    drawing = CapturedDrawing(
        0, None, None, path=CapturedPath([CapturedSubpath(points, closed=closed)])
    )
    return strokes.profile_stroked_text((drawing,), (0,))


def internal_signature(profile: strokes.StrokedTextProfile) -> strokes.GlyphSignature:
    signature = profile.run_profiles[0].signatures[0]
    assert signature is not None
    return signature


def test_approximate_mapping_accepts_small_unique_variants_without_mutating_input() -> None:
    exact = internal_single([(0, 4), (1, 0), (2, 4)])
    variant = internal_single([(0, 4), (1.12, 0), (2, 4)])
    mapping = {internal_signature(exact): "A"}
    decoded = strokes.decode_stroked_text_profile_with_alphabet(variant, mapping)
    assert [o.text for o in decoded.observations] == ["A"]
    assert decoded.approximate_signatures == 1
    assert decoded.learned_signatures == 2
    assert len(mapping) == 1


def test_approximate_mapping_rejects_multiple_labels_and_excessive_distance() -> None:
    left = internal_single([(0, 4), (0.88, 0), (2, 4)])
    center = internal_single([(0, 4), (1, 0), (2, 4)])
    right = internal_single([(0, 4), (1.12, 0), (2, 4)])
    ambiguous = {internal_signature(left): "A", internal_signature(right): "X"}
    assert strokes.decode_stroked_text_profile_with_alphabet(center, ambiguous).observations == ()
    far = internal_single([(0, 4), (1.5, 0), (2, 4)])
    assert (
        strokes.decode_stroked_text_profile_with_alphabet(
            far, {internal_signature(center): "A"}
        ).observations
        == ()
    )


@pytest.mark.parametrize(
    ("points", "closed"), [([(0, 0), (1, 1)], False), ([(0, 4), (1, 0), (2, 4)], True)]
)
def test_signature_comparison_requires_same_topology(
    points: list[tuple[float, float]], closed: bool
) -> None:
    original = internal_signature(internal_single([(0, 4), (1, 0), (2, 4)]))
    different = internal_signature(internal_single(points, closed=closed))
    assert strokes.internal_signature_distance(original, different) is None
    assert strokes.internal_signature_distance(original, ()) is None
    assert strokes.internal_signature_distance(original, ((),)) is None
    assert strokes.internal_signature_distance((), ()) == (0, 0)


@pytest.mark.parametrize(
    ("width", "height", "isolated"),
    [(0.1, 4, False), (2, 9, False), (2, 0.5, False), (65, 4, False), (2, 4, True)],
)
def test_isolated_run_requires_glyph_like_dimensions(
    width: float, height: float, isolated: bool
) -> None:
    profile = internal_single([(0, 0), (width, height)])
    assert bool(strokes.stroked_text_isolated_runs(profile)) is isolated
    decoded = strokes.decode_stroked_text_profile_with_alphabet(
        profile, {internal_signature(profile): "I"}
    )
    if width / height < 0.25:
        assert decoded.observations == ()


def test_punctuation_only_alphabet_does_not_emit_text_observations() -> None:
    profile = internal_profile("AB")
    mapping = {
        signature: "+" for signature in profile.run_profiles[0].signatures if signature is not None
    }
    assert strokes.decode_stroked_text_profile_with_alphabet(profile, mapping).observations == ()


def test_oversized_token_is_not_a_seed_or_decoded_run() -> None:
    profile = internal_profile("A" * 13)
    assert profile.seed_runs == ()
    mapping = {
        signature: "A" for signature in profile.run_profiles[0].signatures if signature is not None
    }
    assert strokes.decode_stroked_text_profile_with_alphabet(profile, mapping).observations == ()


def test_overlapping_path_components_form_one_glyph() -> None:
    drawings = tuple(
        CapturedDrawing(
            i,
            None,
            None,
            path=CapturedPath([CapturedSubpath([(float(x), float(y)) for x, y in points])]),
        )
        for i, points in enumerate(
            ([(0, 4), (1, 0), (2, 4)], [(0.5, 2), (1.5, 2)], [(3, 0), (5, 0), (5, 4)])
        )
    )
    profile = strokes.profile_stroked_text(drawings, (0, 1, 2))
    assert profile.seed_runs[0].glyph_count == 2
    assert len(profile.run_profiles[0].glyphs[0]) == 2
    records = strokes.internal_group_overlapping_x(profile.records)
    assert [len(glyph) for glyph in records] == [2, 1]
    separated = strokes.profile_stroked_text(drawings, (0, 2))
    assert len(separated.run_profiles) == 2


def test_supplemental_seeds_report_missing_or_unaligned_evidence() -> None:
    profile = internal_profile("AB")
    seed = internal_seed(profile, 0, "AB")
    assert (
        strokes.decode_stroked_text_profile_with_supplemental_seeds(
            strokes.StrokedTextProfile(), (), (seed,)
        )
        == strokes.StrokedTextDecode()
    )
    no_alignment = replace(seed, sequence=999, bbox=(100, 100, 110, 110))
    decoded = strokes.decode_stroked_text_profile_with_supplemental_seeds(
        profile, (), (no_alignment,)
    )
    assert decoded.eligible_seeds == 1
    assert decoded.aligned_seeds == 0
    decoded = strokes.decode_stroked_text_profile_with_supplemental_seeds(profile, (), (seed,))
    assert decoded.aligned_seeds == 1
    assert decoded.alphabet == ()


def test_empty_geometry_with_explicit_bounds_cannot_teach_a_glyph() -> None:
    drawings = tuple(
        CapturedDrawing(i, None, None, path=CapturedPath(), bbox=(i * 3, 0, i * 3 + 2, 4))
        for i in range(2)
    )
    profile = strokes.profile_stroked_text(drawings, (0, 1))
    decoded = strokes.decode_stroked_text_profile(profile, (internal_seed(profile, 0, "AB"),))
    assert decoded.eligible_seeds == 1
    assert decoded.aligned_seeds == 0
    assert decoded.observations == ()


def test_seed_run_rejects_excessive_dimensions() -> None:
    drawings = tuple(
        CapturedDrawing(
            i, None, None, path=CapturedPath([CapturedSubpath([(i * 3, 0), (i * 3 + 2, 9)])])
        )
        for i in range(2)
    )
    profile = strokes.profile_stroked_text(drawings, (0, 1))
    assert profile.seed_runs == ()


def test_anchored_learning_reaches_all_supported_glyphs_in_long_chain() -> None:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUV"
    signatures: dict[str, strokes.GlyphSignature] = {
        char: (((False, ((0, 0), (16, 16)) * (i + 1)),),) for i, char in enumerate(alphabet)
    }
    words = ("AB", "AB", *(alphabet[i : i + 3] for i in range(1, 20, 2)))
    samples = tuple(
        strokes.internal_SeedSample(
            strokes.StrokedTextSeed(word, (0, 0, 5, 4), 95, 1 if i == 1 else 0),
            word,
            tuple(signatures[char] for char in word),
        )
        for i, word in enumerate(words)
    )
    mapping, initial, accepted = strokes.internal_consensus_mapping(samples)
    assert initial == 2
    assert mapping == {signature: char for char, signature in signatures.items()}
    assert accepted == 2
