"""Shared fixtures and helpers for the text-preprocessing test suite."""

import os
import pytest

from text_preprocessing import Tokens
from text_preprocessing import Token as PreprocessorToken

# ---------------------------------------------------------------------------
# Fixture file paths
# ---------------------------------------------------------------------------

PLAIN = os.path.join(os.path.dirname(__file__), "fixtures", "plain")
HAMLET = os.path.join(PLAIN, "hamlet.txt")
MOBY_DICK = os.path.join(PLAIN, "moby_dick.txt")
MONTAIGNE = os.path.join(PLAIN, "montaigne.txt")
GERMINAL = os.path.join(PLAIN, "germinal.txt")


# ---------------------------------------------------------------------------
# File-based fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stopwords_file(tmp_path):
    """Simple stopwords file: the, and, of, a."""
    f = tmp_path / "stopwords.txt"
    f.write_text("the\nand\nof\na\n", encoding="utf-8")
    return str(f)


@pytest.fixture
def lemma_file(tmp_path):
    """Tab-separated word→lemma file (lowercase keys — lemmatizer is case-sensitive)."""
    f = tmp_path / "lemmas.txt"
    f.write_text("running\trun\nflies\tfly\nwent\tgo\n", encoding="utf-8")
    return str(f)


# ---------------------------------------------------------------------------
# Token construction helpers
# ---------------------------------------------------------------------------


def make_token(text: str, start: int = 0, end: int = None) -> PreprocessorToken:
    """Create a PreprocessorToken with byte-position metadata."""
    if end is None:
        end = start + len(text)
    return PreprocessorToken(text, ext={"token": text, "start_byte": start, "end_byte": end})


def make_tokens(words: list[str], metadata: dict = None) -> Tokens:
    """Build a Tokens container from a list of words, auto-assigning byte positions."""
    byte = 0
    token_list = []
    for w in words:
        token_list.append(make_token(w, byte, byte + len(w)))
        byte += len(w) + 1  # +1 for implicit space separator
    if metadata is None:
        metadata = {
            "filename": "test.txt",
            "start_byte": 0,
            "end_byte": max(byte - 1, 0),
        }
    return Tokens(token_list, metadata)
