from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator
from functools import lru_cache
from importlib import import_module
from typing import Any

BULLET_CHARS = "\x95•‣⁃ㅤ⁌⁍∙○●◘◦☙❥❧⦾⦿-–\uf0b7*·"

# Since unstructured 0.27, "-" and "–" are bullets only when whitespace or the end of the
# text follows them, so a leading minus sign or hyphenated value is not a list item.
AMBIGUOUS_BULLET_CHARS = "-–"

UNAMBIGUOUS_BULLET_CHARS = "".join(
    char for char in BULLET_CHARS if char not in AMBIGUOUS_BULLET_CHARS
)

BULLET = re.compile(
    rf"^\s*(?:[{re.escape(UNAMBIGUOUS_BULLET_CHARS)}]|[{AMBIGUOUS_BULLET_CHARS}](?=\s|$))"
    rf"(?![{re.escape(BULLET_CHARS)}])"
)

NUMBERED = re.compile(r"^\s*\d+(?:\.|\))\s+.+")

EMAIL = re.compile(r"[a-z0-9.\-+_]+@[a-z0-9.\-+_]+\.[a-z]+")

ADDRESS = re.compile(
    r"\b(?:[A-Z][a-z.\-]{1,15} ?){1,5},\s?"
    r"(?:Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida"
    r"|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland"
    r"|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada"
    r"|New Hampshire|New Jersey|New Mexico|New York|North Carolina|North Dakota"
    r"|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|South Carolina|South Dakota"
    r"|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West Virginia|Wisconsin|Wyoming"
    r"|AL|AK|AS|AZ|AR|CA|CO|CT|DE|DC|FM|FL|GA|GU|HI|ID|IL|IN|IA|KS|KY|LA|ME|MH|"
    r"MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|MP|OH|OK|OR|PW|PA|PR|RI|SC|SD|"
    r"TN|TX|UT|VT|VI|VA|WA|WV|WI|WY)(?:, |\s)?\b\d{5}(?:-\d{4})?\b",
    re.IGNORECASE,
)

POS_VERB_TAGS = frozenset({"VB", "VBG", "VBD", "VBN", "VBP", "VBZ"})


def load_nlp() -> Any:
    try:
        model = import_module("en_core_web_sm")
        return model.load(exclude=["attribute_ruler", "lemmatizer", "ner"])
    except (ImportError, OSError) as error:
        raise ImportError(
            "The Unstructured compatibility facade requires spaCy and en_core_web_sm. "
            "Install them with: python -m pip install 'core-pdf-compat[unstructured]' "
            "'https://github.com/explosion/spacy-models/releases/download/"
            "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'"
        ) from error


NLP = load_nlp()

from ._elements import (  # noqa: E402
    Address,
    Element,
    EmailAddress,
    Footer,
    Header,
    ListItem,
    NarrativeText,
    Text,
    Title,
)


@lru_cache(maxsize=4096)
def nlp_features(
    text: str,
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    nlp = NLP
    document = nlp(text[: nlp.max_length])
    return document_features(document)


def document_features(
    document: Any,
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    tokens = tuple((token.text, "_SP" if token.is_space else token.tag_) for token in document)
    sentences = tuple(sentence.text for sentence in document.sents)
    return tokens, sentences


def sentence_count(sentences: tuple[str, ...], minimum_words: int) -> int:
    if minimum_words <= 0:
        return len(sentences)
    count = 0
    for sentence in sentences:
        words = 0
        in_word = False
        for character in sentence:
            if unicodedata.category(character).startswith("P"):
                continue
            if character.isspace():
                in_word = False
            elif not in_word:
                words += 1
                if words >= minimum_words:
                    count += 1
                    break
                in_word = True
    return count


def classify_element(
    text: str,
    bbox: tuple[float, float, float, float],
    page_height: float,
) -> type[Element]:
    simple_class = simple_element_class(text, bbox, page_height)
    if simple_class is not None:
        return simple_class
    return text_element_class(text, *nlp_features(text))


def classify_elements(
    regions: Iterable[tuple[str, tuple[float, float, float, float]]],
    page_height: float,
) -> Iterator[type[Element]]:
    classified = [(text, simple_element_class(text, bbox, page_height)) for text, bbox in regions]
    nlp = NLP
    documents = nlp.pipe(
        (text[: nlp.max_length] for text, element_class in classified if element_class is None),
        batch_size=32,
    )
    for text, element_class in classified:
        if element_class is not None:
            yield element_class
        else:
            yield text_element_class(text, *document_features(next(documents)))


def simple_element_class(
    text: str,
    bbox: tuple[float, float, float, float],
    page_height: float,
) -> type[Element] | None:
    height_percentage = 1.0 - (bbox[1] + bbox[3]) / (2.0 * page_height) if page_height else 0.5
    if height_percentage < 0.07:
        return Header
    if height_percentage > 0.93:
        return Footer
    if BULLET.match(text) or NUMBERED.match(text):
        return ListItem
    # unstructured 0.27 requires the whole text to be one address.
    if EMAIL.fullmatch(text.strip()):
        return EmailAddress
    if ADDRESS.fullmatch(text.strip()):
        return Address
    if not has_classifiable_text(text):
        return Text
    return None


def has_classifiable_text(text: str) -> bool:
    alphabetic = sum(map(str.isalpha, text))
    non_space = len(text) - sum(map(str.isspace, text))
    return alphabetic / max(non_space, 1) >= 0.5 and not text.isnumeric()


def text_element_class(
    text: str,
    tagged_tokens: tuple[tuple[str, str], ...],
    sentences: tuple[str, ...],
) -> type[Element]:
    if not has_classifiable_text(text):
        return Text
    long_sentence_count = sentence_count(sentences, 3)
    if long_sentence_count > 1:
        return NarrativeText
    word_tokens = [token for token, _tag in tagged_tokens if token.isalpha()]
    capitalized = sum(word.istitle() or word.isupper() for word in word_tokens)
    exceeds_cap_ratio = text.isupper() or not word_tokens or capitalized / len(word_tokens) > 0.5
    has_verb = any(tag in POS_VERB_TAGS for _token, tag in tagged_tokens)
    if not exceeds_cap_ratio and has_verb:
        return NarrativeText
    if (
        len(text.split(" ")) <= 12
        and not text.endswith(",")
        and not (text.isupper() and re.search(r"[^\w\s]$", text))
    ):
        return Title
    return Text
