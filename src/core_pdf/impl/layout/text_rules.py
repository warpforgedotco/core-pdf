# SPDX-License-Identifier: AGPL-3.0-only
"""Pure decision rules for reconstructing text from positioned runs."""

from __future__ import annotations

import gzip
import mmap
import re
import struct
import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from importlib.resources import files
from statistics import median_low

from core_pdf.impl.model.runs import TextRun

WORDLIST_PACKAGE = "core_pdf.impl.layout.data.wordlists"
NORVIG_COUNTS = "norvig_count_1w.txt.gz"
WORDNINJA_WORDS = "wordninja_words.txt.gz"
WORD_RANK_INDEX = "english_word_ranks.bin"
WORD_RANK_MAGIC = b"CPWRANK1"
WORD_RANK_HEADER = struct.Struct("<8sI")
UINT32 = struct.Struct("<I")


@dataclass(frozen=True, slots=True)
class WordFrequency:
    count: int
    rank: int


class WordRankIndex(Mapping[str, int]):
    """Read-only binary-search index backed directly by packaged byte buffer."""

    __slots__ = ("internal_count", "internal_data_start", "internal_mmap")

    def __init__(self, path_or_bytes: str | bytes) -> None:
        if isinstance(path_or_bytes, bytes):
            mapped: mmap.mmap | bytes = path_or_bytes
        else:
            with open(path_or_bytes, "rb") as handle:
                try:
                    mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
                except (OSError, ValueError):
                    mapped = handle.read()
        if len(mapped) < WORD_RANK_HEADER.size:
            if isinstance(mapped, mmap.mmap):
                mapped.close()
            raise ValueError("word-rank index is truncated")
        magic, count = WORD_RANK_HEADER.unpack_from(mapped)
        data_start = WORD_RANK_HEADER.size + (count + 1) * UINT32.size
        if magic != WORD_RANK_MAGIC or data_start > len(mapped):
            if isinstance(mapped, mmap.mmap):
                mapped.close()
            raise ValueError("word-rank index has an unsupported format")
        final_offset = UINT32.unpack_from(
            mapped,
            WORD_RANK_HEADER.size + count * UINT32.size,
        )[0]
        if data_start + final_offset != len(mapped):
            if isinstance(mapped, mmap.mmap):
                mapped.close()
            raise ValueError("word-rank index has invalid offsets")
        self.internal_mmap = mapped
        self.internal_count = count
        self.internal_data_start = data_start

    def internal_offset(self, index: int) -> int:
        return UINT32.unpack_from(
            self.internal_mmap,
            WORD_RANK_HEADER.size + index * UINT32.size,
        )[0]

    def internal_entry(self, index: int) -> tuple[bytes, int]:
        start = self.internal_offset(index)
        stop = self.internal_offset(index + 1)
        absolute_stop = self.internal_data_start + stop
        word = self.internal_mmap[self.internal_data_start + start : absolute_stop - 5]
        rank = UINT32.unpack_from(self.internal_mmap, absolute_stop - UINT32.size)[0]
        return word, rank

    def lookup(self, normalized: str) -> int | None:
        target = normalized.encode("utf-8")
        low = 0
        high = self.internal_count
        while low < high:
            middle = (low + high) // 2
            word, rank = self.internal_entry(middle)
            if word < target:
                low = middle + 1
            elif word > target:
                high = middle
            else:
                return rank
        return None

    def __getitem__(self, word: str) -> int:
        rank = self.lookup(word)
        if rank is None:
            raise KeyError(word)
        return rank

    def __iter__(self) -> Iterator[str]:
        for index in range(self.internal_count):
            yield self.internal_entry(index)[0].decode("utf-8")

    def __len__(self) -> int:
        return self.internal_count


def english_word_frequencies() -> dict[str, WordFrequency]:
    frequencies: dict[str, WordFrequency] = {}
    load_norvig_counts(frequencies)
    load_wordninja_ranks(frequencies)
    return frequencies


def english_word_ranks() -> Mapping[str, int]:
    """Open the packaged rank index without inflating source word lists."""
    import os
    import sys

    # When running under Nuitka compiled / onefile mode, locate file directly in unpacked dist tree
    if "__compiled__" in globals() or "__compiled__" in sys.modules:
        base_dir = os.path.dirname(__file__)
        candidates = [
            os.path.join(base_dir, "data", "wordlists", WORD_RANK_INDEX),
            os.path.join(
                sys.prefix,
                "core_pdf",
                "impl",
                "layout",
                "data",
                "wordlists",
                WORD_RANK_INDEX,
            ),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                try:
                    return WordRankIndex(candidate)
                except Exception:
                    # Path exists but is not a usable index (truncated, wrong layout,
                    # unreadable). Fall through to the next candidate.
                    pass

    res = files(WORDLIST_PACKAGE).joinpath(WORD_RANK_INDEX)
    try:
        return WordRankIndex(res.read_bytes())
    except Exception:
        # Not readable as package data -- e.g. inside a zipimport or a compiled
        # bundle. Fall through to the as_file() path below.
        pass
    from importlib.resources import as_file

    try:
        with as_file(res) as path_obj:
            return WordRankIndex(path_obj.read_bytes())
    except Exception:
        return {word: freq.rank for word, freq in english_word_frequencies().items()}


def load_norvig_counts(frequencies: dict[str, WordFrequency]) -> None:
    res = files(WORDLIST_PACKAGE).joinpath(NORVIG_COUNTS)
    try:
        raw_bytes = res.read_bytes()
        lines = gzip.decompress(raw_bytes).decode("utf-8").splitlines()
    except (TypeError, ValueError, OSError, AttributeError):
        from importlib.resources import as_file

        with as_file(res) as path_obj, gzip.open(str(path_obj), "rt", encoding="utf-8") as handle:
            lines = handle.readlines()

    for rank, line in enumerate(lines, start=1):
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        word, raw_count = parts
        word = word.casefold()
        if not word or not word.isalpha():
            continue
        try:
            count = int(raw_count)
        except ValueError:
            continue
        frequencies[word] = WordFrequency(count, rank)


def load_wordninja_ranks(frequencies: dict[str, WordFrequency]) -> None:
    res = files(WORDLIST_PACKAGE).joinpath(WORDNINJA_WORDS)
    try:
        raw_bytes = res.read_bytes()
        lines = gzip.decompress(raw_bytes).decode("utf-8").splitlines()
    except (TypeError, ValueError, OSError, AttributeError):
        from importlib.resources import as_file

        with as_file(res) as path_obj, gzip.open(str(path_obj), "rt", encoding="utf-8") as handle:
            lines = handle.readlines()

    for rank, line in enumerate(lines, start=1):
        word = line.strip().casefold()
        if not word or not word.isalpha() or word in frequencies:
            continue
        frequencies[word] = WordFrequency(0, rank)


def word_rank(word: str) -> int | None:
    normalized = word.casefold()
    if not normalized or not normalized.isalpha():
        return None
    ranks = english_word_ranks()
    if isinstance(ranks, WordRankIndex):
        return ranks.lookup(normalized)
    return ranks.get(normalized)


FOOTER_RE = re.compile(r"^\s*page\s*\d+\s*$", re.IGNORECASE)
LEADER_START_CHARS = "._~-–—"
SCRIPT_DIGITS = frozenset("⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉")
INLINE_MARKERS = frozenset({"™", "℠", "®", "©"})
FORMULA_MARKERS = frozenset("∂∑√∞∈θΦω")


def runs_are_left_to_right(runs: list[TextRun]) -> bool:
    if len(runs) < 2:
        return True

    previous = runs[0]
    prev_x0 = previous.x0
    prev_x1 = previous.x1
    prev_height = previous.height
    prev_order = previous.order

    for idx in range(1, len(runs)):
        run = runs[idx]
        x0 = run.x0
        if x0 < prev_x0:
            return False
        if x0 == prev_x0 and run.order < prev_order:
            return False
        x1 = run.x1
        height = run.height
        overlap = (prev_x1 if prev_x1 < x1 else x1) - (prev_x0 if prev_x0 > x0 else x0)
        if overlap > (prev_height if prev_height < height else height) * 0.25:
            return False
        prev_x0 = x0
        prev_x1 = x1
        prev_height = height
        prev_order = run.order
    return True


def runs_are_right_to_left(runs: list[TextRun]) -> bool:
    if len(runs) < 2:
        return False
    ordered = runs
    for run in runs:
        if not run.has_text:
            ordered = [text_run for text_run in runs if text_run.has_text]
            if len(ordered) < 2:
                return False
            break

    stream_sorted = sorted(ordered, key=lambda r: (r.order, r.stream_order))
    decreases = 0
    increases = 0
    prev_x0 = stream_sorted[0].x0
    for idx in range(1, len(stream_sorted)):
        x0 = stream_sorted[idx].x0
        if x0 < prev_x0:
            decreases += 1
        elif x0 > prev_x0:
            increases += 1
        prev_x0 = x0
    return decreases > increases


def has_interleaved_horizontal_overlap(runs: list[TextRun]) -> bool:
    previous: TextRun | None = None
    prev_x0 = 0.0
    prev_x1 = 0.0
    prev_space_width = 0.0
    for idx in range(len(runs)):
        run = runs[idx]
        if not run.has_text:
            continue
        x0 = run.x0
        x1 = run.x1
        space_width = run.space_width
        if previous is not None:
            overlap = (prev_x1 if prev_x1 < x1 else x1) - (prev_x0 if prev_x0 > x0 else x0)
            min_width = min(prev_x1 - prev_x0, x1 - x0)
            threshold = max(2.5, min_width * 0.45, max(prev_space_width, space_width) * 0.8)
            if overlap > threshold:
                return True
        previous = run
        prev_x0 = x0
        prev_x1 = x1
        prev_space_width = space_width
    return False


def positive_run_gaps(runs: list[TextRun]) -> list[float]:
    """Positive horizontal gaps between consecutive text-bearing runs."""
    gaps: list[float] = []
    previous: TextRun | None = None
    for run in runs:
        if not run.has_text:
            continue
        if previous is not None:
            gap = run.x0 - previous.x1
            if gap > 0.0:
                gaps.append(gap)
        previous = run
    return gaps


def is_tracked_glyph_run_line(non_space_runs: list[TextRun]) -> bool:
    if len(non_space_runs) < 6:
        return False

    single_glyph_runs = sum(1 for r in non_space_runs if len(r.stripped_text) == 1)
    if single_glyph_runs < len(non_space_runs) * 0.72:
        return False

    text_chars = sum(len(r.stripped_text) for r in non_space_runs)
    if text_chars > len(non_space_runs) * 1.5:
        return False

    positive_gaps = positive_run_gaps(non_space_runs)
    if len(positive_gaps) < len(non_space_runs) * 0.45:
        return False

    positive_gaps.sort()
    typical_gap = positive_gaps[len(positive_gaps) // 2]
    typical_height = max(r.height for r in non_space_runs)
    return typical_gap <= max(typical_height * 1.2, 24.0)


def explicit_spaces_should_control_glyph_gaps(
    non_space_runs: list[TextRun],
    *,
    explicit_space_count: int,
) -> bool:
    if explicit_space_count < 2 or len(non_space_runs) < 8:
        return False
    single_glyph_runs = sum(1 for run in non_space_runs if len(run.stripped_text) == 1)
    if single_glyph_runs < len(non_space_runs) * 0.72:
        return False
    text_chars = sum(len(run.stripped_text) for run in non_space_runs)
    return not text_chars > len(non_space_runs) * 1.5


def tracked_glyph_word_gap_threshold(runs: list[TextRun]) -> float | None:
    gaps = positive_run_gaps(runs)
    if len(gaps) < 5:
        return None
    gaps.sort()
    median_gap = gaps[len(gaps) // 2]
    upper_gap = gaps[int(len(gaps) * 0.8)]
    if upper_gap <= median_gap * 1.8:
        return None
    typical_height = max((r.height for r in runs if r.has_text), default=0.0)
    return max(median_gap * 2.2, typical_height * 0.22, 1.5)


def column_gap_threshold_for_runs(runs: list[TextRun]) -> float:
    non_space_runs = [run for run in runs if run.has_text]
    heights = [run.height for run in non_space_runs if run.height > 0.0]
    spaces = [run.space_width for run in non_space_runs if run.space_width > 0.0]
    typical_height = median_low(heights) if heights else 0.0
    typical_space = median_low(spaces) if spaces else 0.0

    gaps = positive_run_gaps(non_space_runs)
    typical_gap = median_low(gaps) if gaps else 0.0
    return max(
        typical_gap * 4.0,
        typical_space * 5.0,
        typical_height * 1.55,
        12.0,
    )


def estimated_char_width_for_suspect_line(sorted_runs: list[TextRun]) -> float | None:
    non_space_runs = [
        run
        for run in sorted_runs
        if run.visible and run.has_text and run.stripped_text and run.stripped_text.isalpha()
    ]
    if len(non_space_runs) < 5:
        return None
    suspect_runs = 0
    for run in non_space_runs:
        text_len = len(run.stripped_text)
        width = run.x1 - run.x0
        if width <= 0.0:
            suspect_runs += 1
            continue
        if abs((width / max(1, text_len)) - run.space_width) <= max(1.0, run.space_width * 0.05):
            suspect_runs += 1
    if suspect_runs < max(3, len(non_space_runs) // 3):
        return None

    ratios: list[float] = []
    previous: TextRun | None = None
    for run in non_space_runs:
        if previous is not None:
            prev_len = len(previous.stripped_text)
            if prev_len > 0:
                delta = run.x0 - previous.x0
                ratio = delta / prev_len
                space_width = max(1.0, min(previous.space_width, run.space_width))
                if space_width * 0.18 <= ratio <= space_width * 0.95:
                    ratios.append(ratio)
        previous = run
    if len(ratios) < 3:
        return None
    ratios.sort()
    median_ratio = ratios[len(ratios) // 2]
    typical_space = median_low([run.space_width for run in non_space_runs if run.space_width > 0.0])
    if typical_space <= 0.0:
        return median_ratio
    return min(median_ratio, typical_space * 0.48)


def should_use_estimated_word_spacing(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    if previous == "T" and current in {"he", "hes", "hese", "his"}:
        return True
    return not (not previous[-1].isalpha() or not current[0].isalpha())


def repair_table_split_word_boundaries(text: str) -> str:
    """Join dictionary-backed fragments split by table glyph spacing."""
    tokens = text.split(" ")
    if len(tokens) < 2:
        return text
    index = 0
    while index + 1 < len(tokens):
        left = tokens[index]
        right = tokens[index + 1]
        if table_split_word_join_is_plausible(left, right):
            tokens[index : index + 2] = [left + right]
            if index > 0:
                index -= 1
            continue
        index += 1
    repaired = " ".join(tokens)
    return re.sub(r"\s+([.,;:)\]])", r"\1", repaired)


def table_split_word_join_is_plausible(left: str, right: str) -> bool:
    if not left.isalpha() or not right.isalpha():
        return False
    joined = left + right
    joined_rank = word_rank(joined)
    if joined_rank is None or len(joined) < 3:
        return False
    left_rank = word_rank(left)
    right_rank = word_rank(right)
    if left_rank is None or right_rank is None:
        return True
    if joined_rank < min(left_rank, right_rank):
        return True
    return joined_rank <= 10_000 and max(left_rank, right_rank) >= 20_000


def trailing_alpha_token(text: str) -> str:
    index = len(text)
    while index > 0 and text[index - 1].isalpha():
        index -= 1
    return text[index:]


def leading_alpha_token(text: str) -> str:
    index = 0
    limit = len(text)
    while index < limit and text[index].isalpha():
        index += 1
    return text[:index]


def is_high_frequency_boundary_word(text: str) -> bool:
    rank = word_rank(text.strip())
    return rank is not None and rank <= 250


def should_insert_phrase_continuation_space(previous: str, current: str) -> bool:
    tail = trailing_alpha_token(previous.strip())
    head = leading_alpha_token(current.strip())
    if len(head) < 3 or not tail:
        return False
    return len(tail) >= 3 or is_high_frequency_boundary_word(tail)


def should_join_plausible_split_word(
    previous: str,
    current: str,
    *,
    x_gap: float,
    height: float,
    space_width: float,
    prev_visible: bool,
    visible: bool,
    allow_short_prefix: bool = False,
) -> bool:
    if not (prev_visible and visible):
        return False
    prev = previous.strip()
    text = current.strip()
    if not prev or not text:
        return False
    tail = trailing_alpha_token(prev)
    head = leading_alpha_token(text)
    if len(tail) < (1 if allow_short_prefix else 3) or not head:
        return False
    if not (tail[-1].islower() and head[0].islower()):
        return False
    if x_gap < -max(0.5, min(space_width, height) * 0.2):
        return False
    if x_gap > max(space_width * 1.45, height * 0.45, 4.5):
        return False
    if allow_short_prefix and len(tail) < 3:
        # A producer may split one word into several short text-showing
        # operators (``V`` + ``o`` + ``l`` + ``u`` + ``m`` + ``e``). There is
        # no useful dictionary candidate until the final fragment arrives, so
        # keep only tight, lowercase joins in this opt-in path.
        return x_gap <= max(1.8, min(space_width, height) * 0.25)
    joined = f"{tail}{head}"
    joined_rank = word_rank(joined)
    if joined_rank is None or joined_rank > 150_000:
        return False
    tail_rank = word_rank(tail)
    head_rank = word_rank(head)
    if tail_rank is None:
        return True
    if head_rank is None and joined_rank <= 75_000 and len(tail) <= 5:
        return True
    if head_rank is not None and joined_rank < min(tail_rank, head_rank):
        return True
    if len(head) <= 3 and joined_rank <= max(tail_rank * 8, 150_000):
        return True
    return bool(head_rank is None and joined_rank < tail_rank)


def digit_fragments_are_tightly_joined(
    previous: str,
    current: str,
    *,
    x_gap: float,
    height: float,
    space_width: float,
) -> bool:
    prev = previous.strip()
    text = current.strip()
    if not prev or not text or not prev[-1].isdigit() or not text[0].isdigit():
        return False
    return x_gap <= max(0.25, min(space_width, height) * 0.1)


def should_insert_tight_word_space(
    *,
    prev_text: str,
    text: str,
    x_gap: float,
    height: float,
    space_width: float,
) -> bool:
    if not prev_text or not text:
        return False
    prev = prev_text.strip()
    current = text.strip()
    if len(prev) <= 1 or len(current) <= 1:
        return False
    prev_last = prev[-1]
    current_first = current[0]
    if not prev_last.isalpha() or not current_first.isalpha():
        return False

    max_overlap = max(0.25, min(space_width, height) * 0.05)
    if x_gap < -max_overlap:
        return False

    if is_high_frequency_boundary_word(prev) or is_high_frequency_boundary_word(current):
        return True
    return bool(prev.isupper() and current_first.islower() and len(prev) <= 8)


def should_insert_hidden_ocr_overlap_space(
    *,
    prev_text: str,
    text: str,
    x_gap: float,
    height: float,
    space_width: float,
    prev_visible: bool,
    visible: bool,
) -> bool:
    if prev_visible or visible:
        return False
    prev = prev_text.strip()
    current = text.strip()
    if len(prev) <= 1 or len(current) <= 1:
        return False
    prev_last = prev[-1]
    current_first = current[0]
    if not prev_last.isalnum() or not current_first.isalnum():
        return False
    max_overlap = max(1.25, min(space_width, height) * 0.35)
    if x_gap < -max_overlap:
        return False
    if prev_last.isdigit() and current_first.isdigit():
        return False
    if prev.isupper() and current.isupper():
        return True
    if prev_last.isdigit() and current_first.isalpha():
        return True
    if prev_last.isalpha() and current_first.isdigit():
        return True
    return prev_last.islower() != current_first.islower()


def split_glued_numeric_label_boundaries(text: str) -> str:
    if not text:
        return text
    output: list[str] = []
    for index, ch in enumerate(text):
        if index > 0 and should_split_glued_numeric_label(text, index):
            output.append(" ")
        output.append(ch)
    return "".join(output)


def should_split_glued_numeric_label(text: str, index: int) -> bool:
    ch = text[index]
    prev = text[index - 1]
    if ch.isalpha() and prev.isdigit():
        left = text[max(0, index - 8) : index]
        return any(c in left for c in "./,:") and ch.isupper()
    if ch.isdigit() and prev.isalpha():
        left = text[max(0, index - 4) : index].casefold()
        return (
            left.endswith((" m", " m.", " g", " g.", " l", " l."))
            and len(text) > index + 1
            and text[index + 1].isspace()
        )
    return bool(ch.isdigit() and prev == ":")


def strip_private_use_chars(text: str) -> str:
    cleaned = "".join(ch for ch in text if not is_private_use_or_control(ch) and ch not in "»«•·●")
    if "...." in cleaned:
        while "...." in cleaned:
            cleaned = cleaned.replace("....", "..")
    if "----" in cleaned:
        while "----" in cleaned:
            cleaned = cleaned.replace("----", "--")
    return cleaned


def collapse_repeated_spaces(text: str) -> str:
    if "  " not in text:
        return text
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def baseline_midpoint(
    baseline: tuple[float, float, float, float],
    rotation_angle: int,
) -> float:
    if rotation_angle in (90, 270):
        return (baseline[0] + baseline[2]) * 0.5
    return (baseline[1] + baseline[3]) * 0.5


def formula_like_runs(runs: list[TextRun]) -> bool:
    text = "".join(run.text for run in runs if run.has_text)
    return bool(FORMULA_MARKERS.intersection(text))


def reorder_stacked_formula_numerators(runs: list[TextRun]) -> list[TextRun]:
    """Place a vertically stacked numeric numerator before its denominator.

    PDF math producers commonly paint a fraction's numerator and denominator
    as independent glyphs at the same x position.  The ordinary horizontal
    sort consequently emits the denominator first.  Keep the correction
    limited to a small digit/formula overlap so ordinary subscripts and table
    values retain their existing order.
    """
    reordered = list(runs)
    for index in range(1, len(reordered)):
        numerator = reordered[index]
        numerator_text = numerator.stripped_text
        if not (
            (numerator_text.isdigit() and len(numerator_text) <= 2) or numerator_text in {"t", "s"}
        ):
            continue
        denominator_index = next(
            (
                candidate_index
                for candidate_index in range(index - 1, -1, -1)
                if stacked_formula_denominator(
                    reordered[candidate_index],
                    numerator,
                    following=reordered[index + 1] if index + 1 < len(reordered) else None,
                )
            ),
            None,
        )
        if denominator_index is None:
            continue
        denominator = reordered[denominator_index]
        reordered[denominator_index:index] = [numerator, *reordered[denominator_index:index]]
        reordered[index] = denominator
    return reordered


def stacked_formula_denominator(
    denominator: TextRun,
    numerator: TextRun,
    *,
    following: TextRun | None = None,
) -> bool:
    if not denominator.has_text or not numerator.has_text:
        return False
    if denominator.stripped_text.isdigit():
        return False
    if denominator.stripped_text[:1] not in "√GT":
        return False
    numerator_text = numerator.stripped_text
    if denominator.stripped_text[:1] == "T" and numerator_text in {"t", "s"}:
        if following is None or following.stripped_text[:1] not in ",;:)]}]:":
            return False
    elif not numerator_text.isdigit():
        return False
    if denominator.rotation_angle != 0 or numerator.rotation_angle != 0:
        return False
    denominator_height = denominator.height
    numerator_height = numerator.height
    if denominator_height <= 0.0 or numerator_height <= 0.0:
        return False
    if numerator_height < denominator_height * 0.7 or numerator_height > denominator_height * 1.3:
        return False
    overlap = min(denominator.x1, numerator.x1) - max(denominator.x0, numerator.x0)
    if overlap <= 0.0:
        return False
    vertical_gap = numerator.y0 - denominator.y0
    return vertical_gap >= max(2.0, denominator_height * 0.18)


def script_digit_text(text: str) -> bool:
    if len(text) == 1:
        return text in SCRIPT_DIGITS
    stripped = text.strip()
    return bool(stripped) and all(ch in SCRIPT_DIGITS for ch in stripped)


def inline_marker_text(text: str) -> bool:
    if len(text) == 1:
        return text in INLINE_MARKERS
    return text.strip() in INLINE_MARKERS


def compact_unit_suffix_should_join(
    previous: str,
    current: str,
    *,
    x_gap: float,
    height: float,
    space_width: float,
) -> bool:
    prev = previous.strip()
    text = current.strip()
    if text != "V" or not prev or not any(ch.isdigit() for ch in prev):
        return False
    if prev[-1:] not in {"k", "K", "m", "M"}:
        return False
    return x_gap <= max(0.5, min(space_width, height) * 0.15)


def chemical_subscript_prefix_text(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1:].isalpha() and stripped[-1:].isupper()


def is_private_use_or_control(ch: str) -> bool:
    codepoint = ord(ch)
    if ch in "\t\n\r":
        return False
    if codepoint == 0xFFFD or codepoint == 0x00AD:
        return True
    if 0xE000 <= codepoint <= 0xF8FF:
        return True
    category = unicodedata.category(ch)
    return category in {"Cc", "Cf", "Cs", "Co", "Cn"}


def is_structural_list_marker_run(run: TextRun) -> bool:
    return run.stripped_text == "\u25cf"


def is_decorative_leader(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 3 or stripped[0] not in LEADER_START_CHARS:
        return False
    return all(ch in LEADER_START_CHARS or ch.isspace() for ch in stripped)


def is_tiny_page_footer(text: str) -> bool:
    return bool(FOOTER_RE.match(text))


def trailing_tiny_page_label_run_indexes(sorted_runs: list[TextRun]) -> set[int]:
    text_indexes = [index for index, run in enumerate(sorted_runs) if run.stripped_text]
    if len(text_indexes) < 3:
        return set()
    digit_index = text_indexes[-1]
    page_index = text_indexes[-2]
    previous_index = text_indexes[-3]
    digit_run = sorted_runs[digit_index]
    page_run = sorted_runs[page_index]
    previous_run = sorted_runs[previous_index]
    if page_run.stripped_text.casefold() != "page":
        return set()
    if not digit_run.stripped_text.isdigit():
        return set()
    page_width = page_run.x1 - page_run.x0
    digit_width = digit_run.x1 - digit_run.x0
    pair_gap = page_run.x0 - previous_run.x1
    if page_width <= 0.0 or digit_width <= 0.0:
        return set()
    significant_font = max(
        (
            run.font_size
            for run in sorted_runs
            if run.stripped_text
            and run.stripped_text.casefold() != "page"
            and not run.stripped_text.isdigit()
        ),
        default=0.0,
    )
    compact_pair = (
        page_width <= 16.0
        and digit_width <= 14.0
        and pair_gap >= max(32.0, previous_run.space_width * 8.0)
    )
    if not compact_pair and (
        page_run.font_size > 6.5
        or digit_run.font_size > 6.5
        or page_width > 20.0
        or digit_width > 16.0
    ):
        return set()
    if (
        significant_font > 0.0
        and not compact_pair
        and max(page_run.font_size, digit_run.font_size) >= significant_font * 0.7
    ):
        return set()
    if digit_run.x0 < page_run.x0:
        return set()
    return {page_index, digit_index}
