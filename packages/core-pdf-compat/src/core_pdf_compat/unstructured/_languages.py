from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from functools import lru_cache
from importlib import import_module
from typing import Any

from ._elements import Element

type LanguageFallback = Callable[[str], list[str] | None]

ASCII_TEXT = re.compile(r"^[\x00-\x7F]+$")

# langdetect guesses poorly below this many words, so detection calls short ASCII text English.
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


# unstructured 0.27.37's Tesseract language names (Apache-2.0), from
# unstructured.partition.utils.constants: a caller's language is looked up here first.
TESSERACT_LANGUAGES_AND_CODES = {
    "afrikaans": "afr",
    "amharic": "amh",
    "arabic": "ara",
    "assamese": "asm",
    "azerbaijani": "aze",
    "azerbaijani - cyrilic": "aze_cyrl",
    "belarusian": "bel",
    "bengali": "ben",
    "tibetan": "bod",
    "bosnian": "bos",
    "breton": "bre",
    "bulgarian": "bul",
    "catalan; Valencian": "cat",
    "cebuano": "ceb",
    "czech": "ces",
    "chinese - simplified": "chi_sim",
    "chinese": "chi_sim",
    "chinese - traditional": "chi_tra",
    "cherokee": "chr",
    "corsican": "cos",
    "welsh": "cym",
    "danish": "dan",
    "danish - fraktur": "dan_frak",
    "german": "deu",
    "german - fraktur (contrib)": "deu_frak",  # "contrib" not removed because it would repeat key
    "dzongkha": "dzo",
    "greek, modern": "ell",
    "greek": "ell",
    "english": "eng",
    "english, middle": "enm",
    "esperanto": "epo",
    "math / equation detection module": "equ",
    "estonian": "est",
    "basque": "eus",
    "faroese": "fao",
    "persian": "fas",
    "filipino (old - tagalog)": "fil",
    "filipino": "fil",
    "finnish": "fin",
    "french": "fra",
    "german - fraktur": "frk",
    "french, middle": "frm",
    "western frisian": "fry",
    "scottish gaelic": "gla",
    "irish": "gle",
    "galician": "glg",
    "greek, ancient": "grc",
    "gujarati": "guj",
    "haitian": "hat",
    "haitian creole": "hat",
    "hebrew": "heb",
    "hindi": "hin",
    "croatian": "hrv",
    "hungarian": "hun",
    "armenian": "hye",
    "inuktitut": "iku",
    "indonesian": "ind",
    "icelandic": "isl",
    "italian": "ita",
    "italian - old": "ita_old",
    "javanese": "jav",
    "japanese": "jpn",
    "kannada": "kan",
    "georgian": "kat",
    "georgian - old": "kat_old",
    "kazakh": "kaz",
    "central khmer": "khm",
    "kirghiz": "kir",
    "kyrgyz": "kir",
    "kurmanji (kurdish - latin script)": "kmr",
    "korean": "kor",
    "korean (vertical)": "kor_vert",
    "kurdish (arabic script)": "kur",
    "lao": "lao",
    "latin": "lat",
    "latvian": "lav",
    "lithuanian": "lit",
    "luxembourgish": "ltz",
    "malayalam": "mal",
    "marathi": "mar",
    "macedonian": "mkd",
    "maltese": "mlt",
    "mongolian": "mon",
    "maori": "mri",
    "malay": "msa",
    "burmese": "mya",
    "nepali": "nep",
    "dutch": "nld",
    "flemish": "nld",
    "norwegian": "nor",
    "occitan": "oci",
    "oriya": "ori",
    "orientation and script detection module": "osd",
    "panjabi": "pan",
    "punjabi": "pan",
    "polish": "pol",
    "portuguese": "por",
    "pushto": "pus",
    "pashto": "pus",
    "quechua": "que",
    "romanian": "ron",
    "moldavian": "ron",
    "moldovan": "ron",
    "russian": "rus",
    "sanskrit": "san",
    "sinhala": "sin",
    "sinhalese": "sin",
    "slovak": "slk",
    "slovak - fraktur": "slk_frak",
    "slovenian": "slv",
    "sindhi": "snd",
    "spanish": "spa",
    "castilian": "spa",
    "spanish - old": "spa_old",
    "castilian - old": "spa_old",
    "albanian": "sqi",
    "serbian": "srp",
    "serbian - latin": "srp_latn",
    "sundanese": "sun",
    "swahili": "swa",
    "swedish": "swe",
    "syriac": "syr",
    "tamil": "tam",
    "tatar": "tat",
    "telugu": "tel",
    "tajik": "tgk",
    "tagalog": "tgl",
    "thai": "tha",
    "tigrinya": "tir",
    "tonga": "ton",
    "turkish": "tur",
    "uighur": "uig",
    "uyghur": "uig",
    "ukrainian": "ukr",
    "urdu": "urd",
    "uzbek": "uzb",
    "uzbek - cyrilic": "uzb_cyrl",
    "vietnamese": "vie",
    "yiddish": "yid",
    "yoruba": "yor",
}


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


def named_language(language: str) -> str | None:
    # As unstructured reads a language the caller names: a Tesseract name becomes its code,
    # as "english, middle" becomes "enm", and the first three letters are matched.
    code = TESSERACT_LANGUAGES_AND_CODES.get(language.lower(), language)
    match = iso639_language(code[:3])
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
    text: str, languages: Sequence[str], fallback: LanguageFallback | None
) -> list[str] | None:
    if languages[0] == "" or not text.strip():
        return None
    if "auto" in languages:
        # Since 0.27.37 only detection defaults short ASCII text to English: languages the
        # caller names apply to short text too, and the fallback is not called.
        if ASCII_TEXT.match(text) and len(text.split()) < MINIMUM_DETECTABLE_WORDS:
            return ["eng"] if fallback is None else fallback_languages(fallback(text))
        return detected_languages(text)
    return [code for language in languages if (code := named_language(language))]


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
