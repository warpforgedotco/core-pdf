from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

from core_pdf.impl.engine.spec.s_07_content.capture import CapturedPath, CapturedSubpath
from core_pdf.impl.engine.stroked_text import (
    StrokedTextSeed,
    decode_stroked_text,
    decode_stroked_text_profile,
    decode_stroked_text_profile_with_alphabet,
    decode_stroked_text_profile_with_supplemental_seeds,
    decode_stroked_text_with_alphabet,
    internal_signature_distance,
    profile_stroked_text,
    stroked_text_seed_runs,
)

GLYPHS = {
    "A": ((0.0, 0.0), (0.5, 1.0), (1.0, 0.0)),
    "B": ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)),
    "C": ((1.0, 1.0), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0)),
}


def vector_word(
    text: str,
    *,
    y: float,
    scale: float = 1.0,
    shapes: Mapping[str, tuple[tuple[float, float], ...]] = GLYPHS,
) -> tuple[tuple[SimpleNamespace, ...], tuple[float, float, float, float]]:
    drawings: list[SimpleNamespace] = []
    x = 0.0
    for character in text:
        points = [(x + px * scale, y + py * scale) for px, py in shapes[character]]
        path = CapturedPath([CapturedSubpath(points)])
        drawings.append(SimpleNamespace(path=path, rect=path.bbox()))
        x += scale * 1.5
    return tuple(drawings), (0.0, y, x - scale * 0.5, y + scale)


def test_page_local_glyph_decoder_bootstraps_unique_shapes() -> None:
    first, first_box = vector_word("AB", y=0.0)
    second, second_box = vector_word("AB", y=10.0, scale=2.0)
    anchored, anchored_box = vector_word("ABC", y=20.0, scale=1.5)
    variant = GLYPHS | {"A": ((0.0, 0.0), (0.45, 1.0), (1.0, 0.0))}
    unseeded, _ = vector_word("ABC", y=30.0, scale=0.75, shapes=variant)
    drawings = (*first, *second, *anchored, *unseeded)

    decoded = decode_stroked_text(
        drawings,
        range(len(drawings)),
        (
            StrokedTextSeed("AB", first_box, 99.0, 0),
            StrokedTextSeed("AB", second_box, 99.0, 1),
            StrokedTextSeed("ABC", anchored_box, 99.0, 2),
        ),
    )

    assert decoded.initial_signatures == 2
    assert decoded.learned_signatures == 3
    assert decoded.approximate_signatures == 1
    assert decoded.aligned_seeds == 3
    assert tuple(observation.text for observation in decoded.observations) == (
        "AB",
        "AB",
        "ABC",
        "ABC",
    )


def test_page_local_glyph_decoder_rejects_ambiguous_signatures() -> None:
    identical = {"A": GLYPHS["A"], "B": GLYPHS["A"]}
    words_and_boxes = tuple(
        vector_word(text, y=float(index * 10), shapes=identical)
        for index, text in enumerate(("AA", "AA", "BB", "BB"))
    )
    drawings = tuple(drawing for word, ignored_box in words_and_boxes for drawing in word)
    seeds = tuple(
        StrokedTextSeed(text, box, 99.0, index)
        for index, (text, (ignored_word, box)) in enumerate(
            zip(("AA", "AA", "BB", "BB"), words_and_boxes, strict=True)
        )
    )

    decoded = decode_stroked_text(drawings, range(len(drawings)), seeds)

    assert decoded.aligned_seeds == 4
    assert decoded.learned_signatures == 0
    assert decoded.observations == ()


def test_page_local_glyph_decoder_accepts_repeated_85_percent_seeds() -> None:
    first, first_box = vector_word("AB", y=0.0)
    second, second_box = vector_word("AB", y=10.0, scale=2.0)
    drawings = (*first, *second)

    accepted = decode_stroked_text(
        drawings,
        range(len(drawings)),
        (
            StrokedTextSeed("AB", first_box, 85.0, 0),
            StrokedTextSeed("AB", second_box, 85.0, 1),
        ),
    )
    rejected = decode_stroked_text(
        drawings,
        range(len(drawings)),
        (
            StrokedTextSeed("AB", first_box, 84.9, 0),
            StrokedTextSeed("AB", second_box, 84.9, 1),
        ),
    )

    assert tuple(observation.text for observation in accepted.observations) == ("AB", "AB")
    assert rejected.observations == ()


def test_approximate_glyph_mapping_accepts_bounded_half_unit_mean_error() -> None:
    base = {"D": ((0.0, 0.0), (0.2, 0.2), (0.4, 0.4), (0.6, 0.6), (0.8, 0.8), (1.0, 1.0))}
    variant = {
        "D": ((0.0, 0.0), (0.25, 0.25), (0.45, 0.45), (0.65, 0.65), (0.85, 0.85), (1.0, 1.0))
    }
    first, first_box = vector_word("DD", y=0.0, shapes=base)
    second, second_box = vector_word("DD", y=10.0, scale=2.0, shapes=base)
    shifted, _ = vector_word("DD", y=20.0, shapes=variant)
    drawings = (*first, *second, *shifted)

    profile = profile_stroked_text(drawings, range(len(drawings)))
    signatures = dict(profile.signatures)
    first_signature = signatures[(0,)]
    shifted_signature = signatures[(4,)]
    assert first_signature is not None
    assert shifted_signature is not None
    assert internal_signature_distance(first_signature, shifted_signature) == (1, 0.5)

    decoded = decode_stroked_text_profile(
        profile,
        (
            StrokedTextSeed("DD", first_box, 99.0, 0),
            StrokedTextSeed("DD", second_box, 99.0, 1),
        ),
    )

    assert decoded.approximate_signatures == 1
    assert tuple(observation.text for observation in decoded.observations) == ("DD", "DD", "DD")


def test_page_local_glyph_decoder_rejects_degenerate_single_strokes() -> None:
    shapes = {"I": ((0.0, 0.0), (0.0, 1.0))}
    first, first_box = vector_word("II", y=0.0, shapes=shapes)
    second, second_box = vector_word("II", y=10.0, shapes=shapes)
    standalone, _ = vector_word("I", y=20.0, shapes=shapes)
    drawings = (*first, *second, *standalone)

    decoded = decode_stroked_text(
        drawings,
        range(len(drawings)),
        (
            StrokedTextSeed("II", first_box, 99.0, 0),
            StrokedTextSeed("II", second_box, 99.0, 1),
        ),
    )

    assert tuple(observation.text for observation in decoded.observations) == ("II", "II")


def test_seed_runs_retain_only_compact_multi_glyph_words() -> None:
    word, word_box = vector_word("ABC", y=10.0, scale=2.0)
    single, _ = vector_word("A", y=20.0)
    oversized, _ = vector_word("AB", y=30.0, scale=10.0)
    drawings = (*word, *single, *oversized)

    runs = stroked_text_seed_runs(drawings, range(len(drawings)))

    assert len(runs) == 1
    assert runs[0].bbox == word_box
    assert runs[0].drawing_indexes == (0, 1, 2)
    assert runs[0].glyph_count == 3


def test_document_alphabet_decodes_scaled_glyphs_without_ocr_seeds() -> None:
    first, first_box = vector_word("AB", y=0.0)
    second, second_box = vector_word("AB", y=10.0, scale=2.0)
    anchored, anchored_box = vector_word("ABC", y=20.0, scale=1.5)
    source = (*first, *second, *anchored)
    learned = decode_stroked_text(
        source,
        range(len(source)),
        (
            StrokedTextSeed("AB", first_box, 99.0, 0),
            StrokedTextSeed("AB", second_box, 99.0, 1),
            StrokedTextSeed("ABC", anchored_box, 99.0, 2),
        ),
    )
    target, _ = vector_word("ABC", y=30.0, scale=1.25)

    decoded = decode_stroked_text_with_alphabet(
        target,
        range(len(target)),
        learned.alphabet,
    )

    assert tuple(observation.text for observation in decoded.observations) == ("ABC",)
    assert decoded.candidate_runs == 1
    assert decoded.decoded_candidate_runs == 1
    assert decoded.candidate_run_coverage == 1.0
    assert decoded.candidate_glyph_coverage == 1.0


def test_stroked_text_profile_is_reused_by_seed_and_document_decoders() -> None:
    first, first_box = vector_word("AB", y=0.0)
    second, second_box = vector_word("AB", y=10.0, scale=2.0)
    drawings = (*first, *second)
    profile = profile_stroked_text(drawings, range(len(drawings)))

    learned = decode_stroked_text_profile(
        profile,
        (
            StrokedTextSeed("AB", first_box, 99.0, 0),
            StrokedTextSeed("AB", second_box, 99.0, 1),
        ),
    )
    transferred = decode_stroked_text_profile_with_alphabet(profile, learned.alphabet)

    assert tuple(observation.text for observation in transferred.observations) == ("AB", "AB")


def test_supplemental_cell_symbols_extend_primary_alphabet_in_one_decode() -> None:
    first, first_box = vector_word("AB", y=0.0)
    second, second_box = vector_word("AB", y=10.0, scale=2.0)
    supplemented, supplemented_box = vector_word("ABC", y=20.0, scale=1.5)
    drawings = (*first, *second, *supplemented)
    profile = profile_stroked_text(drawings, range(len(drawings)))

    decoded = decode_stroked_text_profile_with_supplemental_seeds(
        profile,
        (
            StrokedTextSeed("AB", first_box, 99.0, 0),
            StrokedTextSeed("AB", second_box, 99.0, 2),
        ),
        (StrokedTextSeed("ABC", supplemented_box, 95.0, 4),),
    )

    assert decoded.aligned_seeds == 3
    assert decoded.learned_signatures == 3
    assert tuple(observation.text for observation in decoded.observations) == (
        "AB",
        "AB",
        "ABC",
    )
