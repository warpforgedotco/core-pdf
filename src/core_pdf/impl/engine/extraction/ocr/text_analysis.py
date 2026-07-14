# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol

from core_pdf.impl.engine.extraction.ocr import rendering as ocr_rendering
from core_pdf.impl.engine.extraction.ocr.types import (
    OcrRow,
    ocr_float_value,
)
from core_pdf.impl.engine.layout.word_frequencies import word_rank

if TYPE_CHECKING:
    from core_pdf.impl.engine.extraction.ocr.candidates import OcrCandidate


TEXT_TOKEN_RE = re.compile(r"\w+")
NONSPACE_TOKEN_RE = re.compile(r"\S+")
UNINTERPRETABLE_TEXT_RE = re.compile(
    "[\ue000-\uf8ff\ufffd\x00-\x08\x0b\x0c\x0e-\x1f\x7f\xad]"
)
OCR_ARTIFACT_EDGE_CHARS = "‘’“”`~_=|¦¬^°•·.,;:!?"
OCR_ARTIFACT_CHARS = frozenset("~_=|¦¬^°•·`“”‘’")
OCR_FORMULA_CHARS = frozenset("()*+/=")
OCR_TECHNICAL_FORMULA_CHARS = frozenset("[]{}|∑∂√∞∈≠≈≊≤≥×−±′″ˆ¯ˉ˜̄θΘωΩφΦℓ")
OCR_STRONG_TECHNICAL_FORMULA_CHARS = frozenset("∑∂√∞∈≠≈≊≤≥×−±′″ˆ¯ˉ˜̄θΘωΩφΦℓ")
OCR_SCHEMATIC_REPAIR_MIN_SUPPORT_TOKENS = 80
OCR_SCHEMATIC_REPAIR_MIN_SUPPORT_TARGETS = 12
FORMULA_CONTROL_TRANSLATION = str.maketrans({"\x02": "[", "\x03": "]"})
_SCHEMATIC_SUPPORT_COMMON_TOKENS = frozenset(
    {
        "agnd",
        "clk",
        "dgnd",
        "gnd",
        "nc",
        "pgnd",
        "pwr",
        "rst",
        "vcc",
        "vdd",
        "vin",
        "vout",
        "vss",
    }
)
_SCHEMATIC_REFERENCE_RE = re.compile(
    r"^(?:ref|net|sw|tp|jp|rv|ic|[rcldqupyjswxtbfmnk])\d+(?:\.\d+)?$",
    re.IGNORECASE,
)
_SCHEMATIC_VALUE_RE = re.compile(
    r"^[+-]?\d+(?:\.\d+)?(?:k|m|g|u|n|p|f|v|h|r|ohm|kohm|mohm|uf|nf|pf|mh|gh)?$",
    re.IGNORECASE,
)
_SCHEMATIC_SUPPORT_EDGE_CHARS = OCR_ARTIFACT_EDGE_CHARS + "()[]{}<>"


class FragmentaryRegionCandidate(Protocol):
    bbox: tuple[float, float, float, float] | None
    image_width: int | None
    image_height: int | None


def extracted_text_token_count(text: str) -> int:
    count = 0
    in_token = False
    for ch in text:
        if ch.isalnum() or ch == "_":
            if not in_token:
                count += 1
                in_token = True
        else:
            in_token = False
    return count


def normalized_text_tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TEXT_TOKEN_RE.finditer(text)]


def support_text_overlap_score(text: str, support_text: str) -> float | None:
    """Return lexical agreement with an independent page-text observation.

    The score is deliberately based on sets and is only a weak signal: OCR can
    correct a damaged native layer, and native text can be incomplete.  A
    caller should therefore use it as a tie-breaker, never as a hard filter.
    """
    candidate_tokens = normalized_text_tokens(text)
    support_tokens = set(normalized_text_tokens(support_text))
    if len(candidate_tokens) < 24 or len(support_tokens) < 12:
        return None
    overlap = sum(token in support_tokens for token in candidate_tokens)
    return overlap / len(candidate_tokens)


def overlapping_ocr_word_penalty(
    rows: list[OcrRow] | tuple[OcrRow, ...],
) -> float:
    """Estimate duplicate OCR output from overlapping word geometry.

    OCR engines occasionally emit the same word more than once when a page
    contains a text layer over a raster image.  Text-only duplicate detection
    cannot distinguish repeated prose from this case; overlapping boxes can.
    """
    words: list[tuple[str, float, float, float, float]] = []
    for row in rows:
        token = str(row.get("text", "")).strip().casefold()
        if not token:
            continue
        try:
            bbox = row.get("page_bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x0, y0, x1, y1 = (ocr_float_value(value) for value in bbox)
            else:
                x0 = ocr_float_value(row["left"])
                y0 = ocr_float_value(row["top"])
                x1 = x0 + ocr_float_value(row["width"])
                y1 = y0 + ocr_float_value(row["height"])
        except KeyError, TypeError, ValueError:
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        words.append((token, x0, y0, x1, y1))
    if len(words) < 8:
        return 0.0
    duplicate_count = 0
    for index, (token, x0, y0, x1, y1) in enumerate(words):
        area = (x1 - x0) * (y1 - y0)
        for other_token, ox0, oy0, ox1, oy1 in words[index + 1 :]:
            if token != other_token:
                continue
            intersection = max(0.0, min(x1, ox1) - max(x0, ox0)) * max(
                0.0, min(y1, oy1) - max(y0, oy0)
            )
            other_area = (ox1 - ox0) * (oy1 - oy0)
            if intersection / max(1.0, min(area, other_area)) >= 0.60:
                duplicate_count += 1
                break
    return duplicate_count / len(words)


def repair_formula_control_delimiters(text: str) -> str:
    if "\x02" not in text and "\x03" not in text:
        return text
    return text.translate(FORMULA_CONTROL_TRANSLATION)


def text_ocr_quality_score(text: str) -> float:
    alnum = 0
    punctuation = 0
    lines = 0
    short_lines = 0
    words = 0
    one_char_words = 0
    current_word_len = 0

    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            lines += 1
            if len(stripped) <= 2:
                short_lines += 1

    for ch in text:
        if ch.isalnum() or ch == "_":
            alnum += 1
            current_word_len += 1
        else:
            if current_word_len:
                words += 1
                if current_word_len == 1:
                    one_char_words += 1
                current_word_len = 0
            if not ch.isspace():
                punctuation += 1
    if current_word_len:
        words += 1
        if current_word_len == 1:
            one_char_words += 1

    punctuation_ratio = punctuation / max(1, alnum + punctuation)
    one_char_ratio = one_char_words / max(1, words)
    short_line_ratio = short_lines / max(1, lines)
    return punctuation_ratio + one_char_ratio * 0.5 + short_line_ratio * 0.2


def rendered_ocr_fragmentation_score(text: str) -> float:
    tokens = [token.strip(OCR_ARTIFACT_EDGE_CHARS) for token in text.split()]
    tokens = [token for token in tokens if token]
    if not tokens:
        return 0.0
    penalty = 0.0
    for line in text.splitlines():
        line_tokens = [
            token.strip(OCR_ARTIFACT_EDGE_CHARS)
            for token in line.split()
            if token.strip(OCR_ARTIFACT_EDGE_CHARS)
        ]
        for index, token in enumerate(line_tokens):
            alpha_count = sum(1 for ch in token if ch.isalpha())
            if alpha_count == 0:
                continue
            if "_" in token and alpha_count >= 2:
                penalty += 1.5
            if any(ch in OCR_ARTIFACT_CHARS for ch in token) and alpha_count >= 2:
                penalty += 0.75
            if len(token) == 1 and token.isalpha():
                previous_alpha = (
                    index > 0
                    and len(line_tokens[index - 1]) >= 2
                    and any(ch.isalpha() for ch in line_tokens[index - 1])
                )
                next_alpha = (
                    index + 1 < len(line_tokens)
                    and len(line_tokens[index + 1]) >= 2
                    and any(ch.isalpha() for ch in line_tokens[index + 1])
                )
                if previous_alpha or next_alpha:
                    penalty += 1.0
            elif (
                len(token) == 2
                and token.isalpha()
                and len(line_tokens) >= 5
                and token.casefold() not in {"of", "to", "in", "on", "by", "as", "is"}
            ):
                previous_alpha = index > 0 and any(
                    ch.isalpha() for ch in line_tokens[index - 1]
                )
                next_alpha = index + 1 < len(line_tokens) and any(
                    ch.isalpha() for ch in line_tokens[index + 1]
                )
                if previous_alpha and next_alpha:
                    penalty += 0.35
    return penalty / len(tokens)


def alphabetic_gibberish_line_score(line: str) -> float:
    raw_tokens = [match.group(0) for match in TEXT_TOKEN_RE.finditer(line)]
    if not raw_tokens:
        return 0.0
    normalized_tokens = [token.casefold() for token in raw_tokens]
    alpha_tokens = [
        (raw, normalized)
        for raw, normalized in zip(raw_tokens, normalized_tokens, strict=True)
        if any(ch.isalpha() for ch in raw) and not any(ch.isdigit() for ch in raw)
    ]
    if len(alpha_tokens) < 4:
        return 0.0
    digit_tokens = sum(1 for token in raw_tokens if any(ch.isdigit() for ch in token))
    if digit_tokens / len(raw_tokens) >= 0.30:
        return 0.0
    if line_has_readable_technical_notation(line, normalized_tokens):
        return 0.0

    known = 0
    label_like = 0
    suspicious = 0
    short_unknown = 0
    short_alpha = 0
    supported_long_alpha = 0
    alpha_token_values: list[str] = []
    for raw, normalized in alpha_tokens:
        alpha_token_values.append(normalized)
        if len(normalized) <= 3:
            short_alpha += 1
        elif normalized.isalpha() and (rank := word_rank(normalized)) is not None:
            if rank <= 100_000:
                supported_long_alpha += 1
        rank = word_rank(normalized) if normalized.isalpha() else None
        if rank is not None and rank <= 100_000:
            known += 1
            continue
        raw_alpha = "".join(ch for ch in raw if ch.isalpha())
        if len(raw_alpha) >= 3 and raw_alpha.isupper():
            label_like += 1
            continue
        if len(normalized) <= 2:
            short_unknown += 1
            suspicious += 1
            continue
        if alpha_token_looks_ocr_garbled(raw):
            suspicious += 1
            continue
        suspicious += 1

    alpha_count = len(alpha_tokens)
    known_ratio = known / alpha_count
    short_alpha_ratio = short_alpha / alpha_count
    repeated_alpha_ratio = 1.0 - (len(set(alpha_token_values)) / alpha_count)
    supported_long_ratio = supported_long_alpha / alpha_count
    low_information_alpha_run = (
        alpha_count >= 8
        and short_alpha_ratio >= 0.65
        and repeated_alpha_ratio >= 0.20
        and supported_long_ratio < 0.20
    )
    if known_ratio >= 0.40 and not low_information_alpha_run:
        return 0.0
    suspicious_ratio = suspicious / alpha_count
    unknown_ratio = (alpha_count - known - label_like) / alpha_count
    short_unknown_ratio = short_unknown / alpha_count
    if (
        suspicious_ratio < 0.45
        and unknown_ratio < 0.60
        and not low_information_alpha_run
    ):
        return 0.0
    if label_like / alpha_count >= 0.65 and suspicious_ratio < 0.55:
        return 0.0
    score = min(
        1.0,
        suspicious_ratio * 0.70 + unknown_ratio * 0.35 + short_unknown_ratio * 0.25,
    )
    if low_information_alpha_run:
        score = max(
            score,
            min(1.0, 0.50 + short_alpha_ratio * 0.30 + repeated_alpha_ratio * 0.60),
        )
    return score


def alphabetic_gibberish_score(text: str) -> float:
    total_tokens = 0
    weighted_score = 0.0
    for line in text.splitlines():
        tokens = normalized_text_tokens(line)
        if not tokens:
            continue
        total_tokens += len(tokens)
        weighted_score += alphabetic_gibberish_line_score(line) * len(tokens)
    if total_tokens == 0:
        return 0.0
    return weighted_score / total_tokens


def rendered_ocr_line_coverage_score(
    text: str,
    *,
    line_rows: int = 0,
    word_rows: int = 0,
) -> float:
    tokens = extracted_text_token_count(text)
    if tokens == 0:
        return 0.0
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not nonempty_lines:
        return 0.0
    useful_lines = sum(
        1
        for line in nonempty_lines
        if extracted_text_token_count(line) >= 2 or len(line) >= 12
    )
    text_line_score = useful_lines / len(nonempty_lines)
    geometry_rows = max(line_rows, word_rows)
    if geometry_rows <= 0:
        return text_line_score
    expected_rows = max(1.0, min(float(len(nonempty_lines) * 1.5), tokens / 4.0))
    geometry_score = min(1.0, geometry_rows / expected_rows)
    return text_line_score * 0.65 + geometry_score * 0.35


def token_alnum_count(token: str) -> int:
    return sum(1 for ch in token if ch.isalnum() or ch == "_")


def scanned_ocr_artifact_score(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    artifact_weight = 0.0
    for token in tokens:
        alnum = token_alnum_count(token)
        if alnum == 0:
            artifact_weight += 1.0
        elif alnum == 1 and len(token) > 3:
            artifact_weight += 1.0
        elif alnum <= 2 and any(ch in OCR_ARTIFACT_CHARS for ch in token):
            artifact_weight += 0.5
    return artifact_weight / len(tokens)


def full_page_diagram_mixed_noise_score(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        return 0.0
    short_diagram_lines = 0
    readable_label_lines = 0
    for line in lines:
        tokens = [match.group(0) for match in TEXT_TOKEN_RE.finditer(line)]
        if not tokens:
            continue
        token_count = len(tokens)
        has_digit = any(ch.isdigit() for ch in line)
        has_axis_symbol = any(ch in line for ch in "+-=")
        short_upper = sum(
            1
            for token in tokens
            if 1 <= len(token) <= 5 and token.isalpha() and token.isupper()
        )
        common_words = sum(
            1
            for token in tokens
            if len(token) >= 3
            and token.isalpha()
            and (rank := word_rank(token.casefold())) is not None
            and rank <= 100_000
        )
        if token_count <= 3 and (has_digit or has_axis_symbol or short_upper >= 1):
            short_diagram_lines += 1
            continue
        if token_count >= 2 and common_words >= 2:
            readable_label_lines += 1
    if short_diagram_lines < 4 or readable_label_lines < 2:
        return 0.0
    return min(1.0, min(short_diagram_lines, readable_label_lines) / len(lines) * 2.2)


def sparse_text_looks_noisy(text: str) -> bool:
    if UNINTERPRETABLE_TEXT_RE.search(text):
        return True
    alnum = 0
    punctuation = 0
    for ch in text:
        if ch.isalnum():
            alnum += 1
        elif not ch.isspace():
            punctuation += 1
    if alnum == 0:
        return True
    return punctuation / (alnum + punctuation) >= 0.12


def uninterpretable_char_count(text: str) -> int:
    return sum(1 for _ in UNINTERPRETABLE_TEXT_RE.finditer(text))


def line_looks_tabular_numeric(line: str) -> bool:
    digit_count = sum(1 for ch in line if ch.isdigit())
    if digit_count < 2:
        return False
    return digit_count / max(1, len(line)) >= 0.12


def text_has_many_digit_lines(text: str) -> bool:
    nonempty_lines = 0
    digit_lines = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        nonempty_lines += 1
        if line_looks_tabular_numeric(stripped):
            digit_lines += 1
    return (
        nonempty_lines >= 12
        and digit_lines >= 8
        and digit_lines / max(1, nonempty_lines) >= 0.25
    )


def dominant_image_text_layer_looks_weak(text: str) -> bool:
    tokens = extracted_text_token_count(text)
    if tokens < 80:
        return False
    quality = text_ocr_quality_score(text)
    return sparse_text_looks_noisy(text) or quality >= 0.10


def alpha_token_looks_ocr_garbled(token: str) -> bool:
    if not any(ch.isalpha() for ch in token):
        return False
    if any(ch.isdigit() for ch in token):
        return True
    if not token.isalpha():
        return True
    return not (token.islower() or token.isupper() or token.istitle())


def line_has_non_ascii_alnum(line: str) -> bool:
    return any(ch.isalnum() and ord(ch) > 0x7F for ch in line)


def line_has_readable_technical_notation(line: str, tokens: list[str]) -> bool:
    if len(tokens) < 4:
        return False
    alpha_tokens = [token for token in tokens if any(ch.isalpha() for ch in token)]
    if len(alpha_tokens) < 2:
        return False
    raw_alpha_run_count = 0
    mixed_case_runs = 0
    run_has_lower = False
    run_has_upper = False
    run_first_is_upper = False
    run_title_rest_lower = True
    in_alpha_run = False
    for ch in line:
        is_upper = "A" <= ch <= "Z"
        is_lower = "a" <= ch <= "z"
        if is_upper or is_lower:
            if not in_alpha_run:
                in_alpha_run = True
                run_has_lower = is_lower
                run_has_upper = is_upper
                run_first_is_upper = is_upper
                run_title_rest_lower = True
            else:
                run_has_lower = run_has_lower or is_lower
                run_has_upper = run_has_upper or is_upper
                run_title_rest_lower = run_title_rest_lower and is_lower
            continue
        if in_alpha_run:
            raw_alpha_run_count += 1
            if not (
                (run_has_lower and not run_has_upper)
                or (run_has_upper and not run_has_lower)
                or (run_first_is_upper and run_title_rest_lower)
            ):
                mixed_case_runs += 1
            in_alpha_run = False
    if in_alpha_run:
        raw_alpha_run_count += 1
        if not (
            (run_has_lower and not run_has_upper)
            or (run_has_upper and not run_has_lower)
            or (run_first_is_upper and run_title_rest_lower)
        ):
            mixed_case_runs += 1
    if mixed_case_runs >= max(2, int(raw_alpha_run_count * 0.15)):
        return False
    weird_alpha = sum(
        1 for token in alpha_tokens if alpha_token_looks_ocr_garbled(token)
    )
    if sum(
        1
        for token in alpha_tokens
        if len(token) >= 3
        and token.isalpha()
        and (rank := word_rank(token)) is not None
        and rank <= 5_000
    ):
        if not weird_alpha and not line_has_non_ascii_alnum(line):
            return False
    unknown_alpha = sum(
        1
        for token in alpha_tokens
        if len(token) >= 2
        and (
            not token.isalpha() or (rank := word_rank(token)) is None or rank > 100_000
        )
    )
    if line_has_non_ascii_alnum(line) and unknown_alpha:
        return True
    notation_chars = sum(
        1
        for ch in line
        if ch in OCR_FORMULA_CHARS
        or ch in OCR_TECHNICAL_FORMULA_CHARS
        or ch in "\x02\x03-\u2013\u2014_,;"
    )
    non_bracket_notation_chars = sum(
        1
        for ch in line
        if ch in OCR_FORMULA_CHARS
        or ch in OCR_STRONG_TECHNICAL_FORMULA_CHARS
        or ch in "-\u2013\u2014_,;"
    )
    math_notation_chars = sum(
        1 for ch in line if ch in OCR_TECHNICAL_FORMULA_CHARS or ch in "\x02\x03"
    )
    strong_math_notation_chars = sum(
        1 for ch in line if ch in OCR_STRONG_TECHNICAL_FORMULA_CHARS
    )
    mixed_alnum_tokens = sum(
        1
        for token in tokens
        if any(ch.isalpha() for ch in token) and any(ch.isdigit() for ch in token)
    )
    if (
        non_bracket_notation_chars < 2
        and mixed_alnum_tokens < 2
        and strong_math_notation_chars < 2
    ):
        return False
    common_alpha = sum(
        1
        for token in alpha_tokens
        if len(token) >= 3
        and token.isalpha()
        and (rank := word_rank(token)) is not None
        and rank <= 75_000
    )
    formula_alpha = sum(
        1
        for token in alpha_tokens
        if token in {"c", "h", "n", "o", "p", "s", "si", "ch", "nh", "oh", "oc"}
    )
    long_unknown_alpha = sum(
        1
        for token in alpha_tokens
        if len(token) >= 5
        and (
            not token.isalpha() or (rank := word_rank(token)) is None or rank > 100_000
        )
    )
    formula_like_mixed = mixed_alnum_tokens >= 2 and notation_chars >= 3
    if weird_alpha / len(alpha_tokens) > 0.25 and not formula_like_mixed:
        return False
    formula_like_symbols = (
        math_notation_chars >= 3
        and strong_math_notation_chars >= 2
        and notation_chars >= 4
    )
    if formula_like_symbols and len(alpha_tokens) >= 2:
        return True
    if common_alpha >= 2 and non_bracket_notation_chars >= 2:
        return True
    if common_alpha >= 1 and non_bracket_notation_chars >= 3:
        return bool(long_unknown_alpha or formula_alpha >= 2 or mixed_alnum_tokens)
    if formula_like_mixed and long_unknown_alpha >= 2:
        return True
    if long_unknown_alpha >= 2 and notation_chars >= 4 and not weird_alpha:
        return True
    return False


def supplemental_ocr_short_line_looks_tabular(
    tokens: list[str],
    seen_tokens: set[str],
) -> bool:
    if not tokens:
        return False
    if not any(any(ch.isdigit() for ch in token) for token in tokens):
        return False
    return any(token not in seen_tokens for token in tokens)


def supplemental_ocr_line_looks_fragmentary(
    tokens: list[str],
    line: str = "",
) -> bool:
    if len(tokens) < 3:
        return False
    if line and line_has_readable_technical_notation(line, tokens):
        return False
    digit_tokens = sum(1 for token in tokens if any(ch.isdigit() for ch in token))
    if digit_tokens / len(tokens) >= 0.35:
        return False
    alpha_tokens = [token for token in tokens if any(ch.isalpha() for ch in token)]
    if len(alpha_tokens) < 3:
        return False
    short_alpha = sum(1 for token in alpha_tokens if len(token) <= 2)
    common_alpha = sum(
        1
        for token in alpha_tokens
        if len(token) >= 3
        and token.isalpha()
        and (rank := word_rank(token)) is not None
        and rank <= 75_000
    )
    strong_common_alpha = sum(
        1
        for token in alpha_tokens
        if len(token) >= 3
        and token.isalpha()
        and (rank := word_rank(token)) is not None
        and rank <= 20_000
    )
    if short_alpha / len(alpha_tokens) >= 0.65 and common_alpha == 0:
        return True
    if len(alpha_tokens) >= 8 and short_alpha / len(alpha_tokens) >= 0.40:
        return strong_common_alpha / len(alpha_tokens) <= 0.15
    if len(alpha_tokens) >= 5 and short_alpha / len(alpha_tokens) >= 0.50:
        return common_alpha / len(alpha_tokens) <= 0.25
    return False


def numeric_token_ratio(text: str) -> float:
    tokens = normalized_text_tokens(text)
    if not tokens:
        return 0.0
    numeric = sum(1 for token in tokens if any(ch.isdigit() for ch in token))
    return numeric / len(tokens)


def ocr_line_short_token_ratio(text: str) -> float:
    tokens = normalized_text_tokens(text)
    if not tokens:
        return 1.0
    return sum(1 for token in tokens if len(token) <= 2) / len(tokens)


def ocr_line_punctuation_ratio(text: str) -> float:
    nonspace = [ch for ch in text if not ch.isspace()]
    if not nonspace:
        return 1.0
    punctuation = sum(1 for ch in nonspace if not ch.isalnum())
    return punctuation / len(nonspace)


def chemical_signal_count(text: str) -> int:
    count = 0
    tokens = normalized_text_tokens(text)
    compact = text.casefold()
    for token in tokens:
        if token in {
            "c",
            "h",
            "n",
            "o",
            "p",
            "s",
            "si",
            "nh",
            "oh",
            "oc",
            "pg",
        }:
            count += 1
            continue
        if (
            len(token) >= 2
            and any(ch.isalpha() for ch in token)
            and any(ch.isdigit() for ch in token)
        ):
            count += 1
            continue
        if token in {"wherein", "alkyl", "alkenyl", "alkynyl", "haloalkyl"}:
            count += 1
    count += compact.count("ch2")
    count += compact.count("och")
    count += compact.count("nh")
    return count


def ocr_confusion_char_count(text: str) -> int:
    return sum(1 for ch in text if ch in {"_", "|", "¢", "©", "®", "™", "!", "?", "¦"})


def canonicalized_ocr_consensus_tokens(text: str) -> tuple[str, ...]:
    tokens = normalized_text_tokens(text)
    canonical: list[str] = []
    for token in tokens:
        normalized = (
            token.replace("l", "1")
            .replace("i", "1")
            .replace("o", "0")
            .replace("s", "5")
            .replace("¢", "c")
        )
        canonical.append(normalized)
    return tuple(canonical)


def alpha_unknown_word_ratio(text: str) -> float | None:
    if numeric_token_ratio(text) >= 0.30:
        return None
    alpha_tokens = [
        token
        for token in normalized_text_tokens(text)
        if len(token) >= 3 and token.isalpha()
    ]
    if len(alpha_tokens) < 40:
        return None
    if ocr_text_has_dense_formula_notation(text):
        return None
    unknown = sum(
        1
        for token in alpha_tokens
        if (rank := word_rank(token)) is None or rank > 100_000
    )
    return unknown / len(alpha_tokens)


def ocr_line_has_formula_notation(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 3:
        return False
    formula_chars = sum(1 for ch in stripped if ch in OCR_FORMULA_CHARS)
    formula_chars += sum(1 for ch in stripped if ch in OCR_TECHNICAL_FORMULA_CHARS)
    formula_chars += sum(1 for ch in stripped if ch in "_!?|")
    if formula_chars < 2:
        return False
    alnum = sum(1 for ch in stripped if ch.isalnum())
    if alnum == 0:
        return False
    uppercase = sum(1 for ch in stripped if ch.isupper())
    digits = sum(1 for ch in stripped if ch.isdigit())
    return uppercase + digits >= 2


def ocr_text_has_dense_formula_notation(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        return False
    formula_lines = sum(1 for line in lines if ocr_line_has_formula_notation(line))
    if formula_lines < 6:
        return False
    formula_chars = sum(1 for ch in text if ch in OCR_FORMULA_CHARS)
    formula_chars += sum(1 for ch in text if ch in OCR_TECHNICAL_FORMULA_CHARS)
    formula_chars += sum(1 for ch in text if ch in "_!?|")
    return formula_chars >= 24


def table_like_ocr_coverage_bonus(
    text: str,
    tokens: int,
    confidence: int | None,
    quality: float,
) -> float:
    if not (450 <= tokens <= 750):
        return 0.0
    if confidence is None or confidence < 80:
        return 0.0
    if quality > 0.30:
        return 0.0
    if not text_has_many_digit_lines(text):
        return 0.0
    return min(18.0, 8.0 + (tokens - 450) * 0.04)


def fragmentary_region_candidate_penalty(
    candidate: FragmentaryRegionCandidate,
    tokens: int,
) -> float:
    bbox = candidate.bbox
    if bbox is None:
        return 0.0
    image_width = candidate.image_width
    image_height = candidate.image_height
    if image_width is None or image_height is None:
        return 0.0
    if image_width <= 0 or image_height <= 0:
        return 0.0
    x0, y0, x1, y1 = bbox
    width = max(0, x1 - x0)
    height = max(0, y1 - y0)
    if width <= 0 or height <= 0:
        return 0.0
    coverage = (width * height) / float(image_width * image_height)
    if coverage >= 0.015 or tokens >= 24:
        return 0.0
    token_penalty = 6.0 + max(0, 24 - tokens) * 1.25
    coverage_penalty = ((0.015 - coverage) / 0.015) * 12.0
    return token_penalty + coverage_penalty


def _normalize_schematic_support_token(token: str) -> str:
    return token.strip(_SCHEMATIC_SUPPORT_EDGE_CHARS).casefold()


def _schematic_support_token_is_target(token: str) -> bool:
    if not token:
        return False
    if token in _SCHEMATIC_SUPPORT_COMMON_TOKENS:
        return True
    if _SCHEMATIC_REFERENCE_RE.fullmatch(token):
        return True
    return bool(_SCHEMATIC_VALUE_RE.fullmatch(token))


def _schematic_support_tokens(text: str) -> frozenset[str]:
    tokens: set[str] = set()
    for match in NONSPACE_TOKEN_RE.finditer(text):
        token = _normalize_schematic_support_token(match.group(0))
        if _schematic_support_token_is_target(token):
            tokens.add(token)
    return frozenset(tokens)


def vector_text_supports_schematic_tiled_ocr(vector_text: str) -> bool:
    if (
        extracted_text_token_count(vector_text)
        < ocr_rendering.OCR_SCHEMATIC_VECTOR_RENDER_TILE_MIN_TOKENS
    ):
        return False
    return len(_schematic_support_tokens(vector_text)) >= (
        OCR_SCHEMATIC_REPAIR_MIN_SUPPORT_TARGETS
    )


def schematic_tiled_ocr_candidate_support_metrics(
    candidate: OcrCandidate,
    support_text: str,
) -> dict[str, float | int]:
    if not candidate.name.startswith("rendered_page_"):
        return {}
    if not candidate.name.endswith("_tiled"):
        return {}
    support_targets = _schematic_support_tokens(support_text)
    if (
        extracted_text_token_count(support_text)
        < ocr_rendering.OCR_SCHEMATIC_VECTOR_RENDER_TILE_MIN_TOKENS
        or len(support_targets) < OCR_SCHEMATIC_REPAIR_MIN_SUPPORT_TARGETS
    ):
        return {}
    tokens = []
    for match in NONSPACE_TOKEN_RE.finditer(candidate.result.text):
        token = _normalize_schematic_support_token(match.group(0))
        if token:
            tokens.append(token)
    if len(tokens) < 40:
        return {}
    support_hits = sum(1 for token in tokens if token in support_targets)
    unique_support_hits = len(set(tokens).intersection(support_targets))
    if unique_support_hits < 12 or support_hits < 35:
        bonus = 0.0
    else:
        token_bonus = min(20.0, len(tokens) * 0.10)
        support_bonus = min(80.0, support_hits * 0.50)
        unique_bonus = min(70.0, unique_support_hits * 1.20)
        bonus = token_bonus + support_bonus + unique_bonus
    return {
        "schematic_support_targets": len(support_targets),
        "schematic_support_hits": support_hits,
        "schematic_unique_support_hits": unique_support_hits,
        "schematic_support_bonus": round(bonus, 4),
    }
