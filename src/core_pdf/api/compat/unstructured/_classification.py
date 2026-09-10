"""Required NLP pipeline and Unstructured semantic classification."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from importlib import import_module
from typing import Any

internal_BULLET_CHARS = "\x95•‣⁃ㅤ⁌⁍∙○●◘◦☙❥❧⦾⦿-–\uf0b7*·"

internal_BULLET = re.compile(
    rf"^\s*[{re.escape(internal_BULLET_CHARS)}](?![{re.escape(internal_BULLET_CHARS)}])"
)

internal_NUMBERED = re.compile(r"^\s*\d+(?:\.|\))\s+.+")

internal_EMAIL = re.compile(r"[a-z0-9.\-+_]+@[a-z0-9.\-+_]+\.[a-z]+", re.IGNORECASE)

internal_ADDRESS = re.compile(
    r"^(?:[A-Z][a-z.\-]{1,15} ?){1,5},\s?"
    r"(?:AL|AK|AS|AZ|AR|CA|CO|CT|DE|DC|FM|FL|GA|GU|HI|ID|IL|IN|IA|KS|KY|LA|ME|MH|"
    r"MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|MP|OH|OK|OR|PW|PA|PR|RI|SC|SD|"
    r"TN|TX|UT|VT|VI|VA|WA|WV|WI|WY)(?:, |\s)?\d{5}(?:-\d{4})?\b",
    re.IGNORECASE,
)

internal_POS_VERB_TAGS = frozenset({"VB", "VBG", "VBD", "VBN", "VBP", "VBZ"})


def internal_load_nlp() -> Any:
    """Load the required English pipeline without installing dependencies at runtime."""
    try:
        model = import_module("en_core_web_sm")
        return model.load()
    except (ImportError, OSError) as error:
        raise ImportError(
            "The Unstructured compatibility facade requires spaCy and en_core_web_sm. "
            "Install them with: python -m pip install 'core-pdf[unstructured]' "
            "'https://github.com/explosion/spacy-models/releases/download/"
            "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'"
        ) from error


internal_NLP = internal_load_nlp()

# Load the required model before importing elements. Failed facade imports must
# not leave a cached element module that can be imported without the model.
from ._elements import (  # noqa: E402
    Address,
    Element,
    EmailAddress,
    Footer,
    Header,
    ListItem,
    NarrativeText,
    Title,
    UncategorizedText,
)


@lru_cache(maxsize=4096)
def internal_nlp_features(
    text: str,
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    nlp = internal_NLP
    document = nlp(text[: nlp.max_length])
    tokens = tuple((token.text, token.tag_) for token in document)
    sentences = tuple(sentence.text for sentence in document.sents)
    return tokens, sentences


def internal_sentence_count(sentences: tuple[str, ...], minimum_words: int) -> int:
    return sum(
        len(
            "".join(
                character
                for character in sentence
                if not unicodedata.category(character).startswith("P")
            ).split()
        )
        >= minimum_words
        for sentence in sentences
    )


def internal_element_class(
    text: str,
    bbox: tuple[float, float, float, float],
    page_height: float,
) -> type[Element]:
    height_percentage = 1.0 - (bbox[1] + bbox[3]) / (2.0 * page_height) if page_height else 0.5
    if height_percentage < 0.07:
        return Header
    if height_percentage > 0.93:
        return Footer
    if internal_BULLET.match(text) or internal_NUMBERED.match(text):
        return ListItem
    if internal_EMAIL.match(text.strip()):
        return EmailAddress
    if internal_ADDRESS.search(text):
        return Address
    alphabetic = sum(character.isalpha() for character in text)
    non_space = sum(not character.isspace() for character in text)
    alpha_ratio = alphabetic / max(non_space, 1)
    tagged_tokens, sentences = internal_nlp_features(text)
    word_tokens = [token for token, _tag in tagged_tokens if token.isalpha()]
    capitalized = sum(word.istitle() or word.isupper() for word in word_tokens)
    long_sentence_count = internal_sentence_count(sentences, 3)
    exceeds_cap_ratio = long_sentence_count <= 1 and (
        text.isupper() or not word_tokens or capitalized / len(word_tokens) > 0.5
    )
    has_verb = any(tag in internal_POS_VERB_TAGS for _token, tag in tagged_tokens)
    if (
        alpha_ratio >= 0.5
        and not text.isnumeric()
        and (long_sentence_count > 1 or not exceeds_cap_ratio)
        and (has_verb or long_sentence_count >= 2)
    ):
        return NarrativeText
    if (
        len(text.split(" ")) <= 12
        and alpha_ratio >= 0.5
        and not text.isnumeric()
        and not text.endswith(",")
        and not (text.isupper() and re.search(r"[^\w\s]$", text))
        and internal_sentence_count(sentences, 5) <= 1
    ):
        return Title
    return UncategorizedText
