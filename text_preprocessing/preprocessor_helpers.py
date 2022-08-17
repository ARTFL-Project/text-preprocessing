import re
import sys
import unicodedata
from html import unescape as unescape_html
from typing import (
    Any,
    Callable,
    DefaultDict,
    Deque,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    overload,
)
from xml.sax.saxutils import unescape as unescape_xml

import mmh3
from unidecode import unidecode


PUNCTUATION_MAP = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P"))
PUNCTUATION_CLASS = set([chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")])
TRIM_LAST_SLASH = re.compile(r"/\Z")
NUMBERS = re.compile(r"\d")
TAGS = re.compile(r"<[^>]+>")


def entities_to_string(text):
    """Convert entities to text"""
    text = unescape_html(text)
    text = unescape_xml(text)
    return text


def normalize(
    token,
    stemmer,
    lemmatizer,
    convert_entities,
    strip_numbers,
    strip_punctuation,
    stopwords,
    lowercase,
    min_word_length,
    hash_tokens,
    ascii,
):
    token_text = token.text.strip()
    surface_form = token_text[:]
    if lemmatizer == "spacy":
        token_text = token.lemma_
    elif lemmatizer:  # we have a dict for conversion
        token_text = lemmatizer.get(token_text, token_text)
    if convert_entities is True:
        token_text = entities_to_string(token_text)
    if lowercase is True:
        token_text = token_text.lower()
    if any(
        (
            token_text in stopwords,
            strip_numbers is True and NUMBERS.search(token_text),
            len(token_text) < min_word_length,
        )
    ):
        return "", surface_form
    if strip_punctuation is True:
        token_text = token_text.translate(PUNCTUATION_MAP)
    elif token_text in PUNCTUATION_CLASS:
        return "", surface_form
    if stemmer is not False:
        token_text = stemmer.stemWord(token_text)  # type: ignore
    if ascii is True:
        token_text = unidecode(token_text)
    if hash_tokens is True:
        token_text = str(mmh3.hash(token_text))
    return token_text, surface_form
