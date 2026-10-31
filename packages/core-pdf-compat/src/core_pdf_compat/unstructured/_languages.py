from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from functools import lru_cache
from importlib import import_module
from typing import Any

from ._elements import Element

type LanguageFallback = Callable[[str], list[str] | None]

ASCII_TEXT = re.compile(r"^[\x00-\x7F]+$")

# langdetect guesses poorly below this many words, so short ASCII text is English.
MINIMUM_DETECTABLE_WORDS = 5


def load_language_libraries() -> tuple[Any, Any]:
    try:
        return import_module("langdetect"), import_module("iso639")
    except ImportError as error:
        raise ImportError(
            "The Unstructured compatibility facade requires langdetect and python-iso639. "
            "Install them with: python -m pip install 'core-pdf-compat[unstructured]'"
        ) from error


LANGDETECT, ISO639 = load_language_libraries()


def check_languages_argument(languages: object) -> None:
    if languages and not isinstance(languages, list):
        raise TypeError(
            "The language parameter must be a list of language codes as strings, ex. ['eng']"
        )


@lru_cache(maxsize=256)
def iso639_language(code: str) -> Any:
    try:
        return ISO639.Language.match(code, strict_case=False)
    except ISO639.LanguageNotFoundError:
        return None


def requested_language(language: str) -> str | None:
    # A name, as in "japanese", is matched whole, and a code by its prefix, as "chi_sim".
    match = iso639_language(language) or iso639_language(language[:3])
    return None if match is None else match.part3


def fallback_languages(value: list[str] | None) -> list[str] | None:
    if not isinstance(value, list):
        return None
    validated: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            continue
        language = item.strip()
        match = "zho" if language == "zho" else requested_language(language[:3])
        if match is not None:
            validated.append(match)
    return validated or None


def detected_languages(text: str) -> list[str] | None:
    # langdetect is random unless seeded, and resets nothing between calls.
    LANGDETECT.DetectorFactory.seed = 0
    try:
        detected = LANGDETECT.detect_langs(text)
    except LANGDETECT.lang_detect_exception.LangDetectException:
        return None
    languages: list[str] = []
    for candidate in detected:
        code = str(candidate.lang)
        language = "zho" if code.startswith("zh") else requested_language(code[:3])
        if language is not None and language not in languages:
            languages.append(language)
    return languages


def text_languages(
    text: str,
    languages: Sequence[str],
    fallback: LanguageFallback | None,
) -> list[str] | None:
    if languages[0] == "" or not text.strip():
        return None
    if ASCII_TEXT.match(text) and len(text.split()) < MINIMUM_DETECTABLE_WORDS:
        return ["eng"] if fallback is None else fallback_languages(fallback(text))
    if "auto" in languages:
        return detected_languages(text)
    return [code for language in languages if (code := requested_language(language))]


def apply_languages(
    elements: list[Element],
    languages: Sequence[str] | None,
    *,
    per_element: bool,
    fallback: LanguageFallback | None,
) -> None:
    requested = languages or ["auto"]
    if requested == [""]:
        return
    document_languages = text_languages(
        " ".join(element.text for element in elements if element.text), requested, fallback
    )
    single_language = document_languages is not None and len(document_languages) == 1
    for element in elements:
        element.metadata["languages"] = (
            document_languages
            if single_language and not per_element
            else text_languages(element.text, requested, fallback)
        )
