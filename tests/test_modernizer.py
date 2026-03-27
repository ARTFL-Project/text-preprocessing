"""
Tests for the Modernizer class.

The Modernizer converts archaic or obsolete word forms to their modern
equivalents.  It wraps a large static dictionary for each supported language
and exposes a simple callable / dict-like interface.
"""

import pytest

from text_preprocessing.modernizer import Modernizer


# ===========================================================================
# English Modernizer
# ===========================================================================

class TestEnglishModernizer:

    @pytest.fixture(autouse=True)
    def modernizer(self):
        self.m = Modernizer("english")

    # --- call interface ---

    def test_known_word_is_modernised(self):
        assert self.m("caviare") == "caviar"

    def test_another_known_word(self):
        assert self.m("loth") == "loath"

    def test_unknown_word_returned_unchanged(self):
        assert self.m("xyzunknown") == "xyzunknown"

    def test_empty_string_returned_unchanged(self):
        assert self.m("") == ""

    # --- dict-like interface ---

    def test_getitem_known_word(self):
        assert self.m["caviare"] == "caviar"

    def test_get_method_known_word(self):
        assert self.m.get("caviare") == "caviar"

    def test_get_method_unknown_word(self):
        assert self.m.get("xyzunknown") == "xyzunknown"

    # --- words found in Hamlet fixture ---

    @pytest.mark.parametrize("archaic,modern", [
        ("caviare", "caviar"),
        ("loth", "loath"),
        ("sayst", "sayest"),
        ("passeth", "passes"),
        ("heyday", "heyday"),  # hey-day hyphen form isn't a single token; check heyday passes through
    ])
    def test_hamlet_vocab(self, archaic, modern):
        assert self.m(archaic) == modern


# ===========================================================================
# French Modernizer
# ===========================================================================

class TestFrenchModernizer:

    @pytest.fixture(autouse=True)
    def modernizer(self):
        self.m = Modernizer("french")

    # --- call interface ---

    def test_known_word_is_modernised(self):
        assert self.m("luy") == "lui"

    def test_another_known_word(self):
        assert self.m("mesme") == "même"

    def test_unknown_word_returned_unchanged(self):
        assert self.m("xyzunknown") == "xyzunknown"

    def test_empty_string_returned_unchanged(self):
        assert self.m("") == ""

    # --- dict-like interface ---

    def test_getitem_known_word(self):
        assert self.m["luy"] == "lui"

    def test_get_method_known_word(self):
        assert self.m.get("mesme") == "même"

    # --- words found in Montaigne fixture ---

    @pytest.mark.parametrize("archaic,modern", [
        ("luy", "lui"),
        ("mesme", "même"),
        ("vray", "vrai"),
        ("iuger", "juger"),
        ("coustume", "coutume"),
        ("souuent", "souvent"),
        ("defaut", "défaut"),
    ])
    def test_montaigne_vocab(self, archaic, modern):
        assert self.m(archaic) == modern


# ===========================================================================
# Language switching
# ===========================================================================

class TestModernizerLanguageSwitching:

    def test_reinitialise_with_different_language(self):
        # Modernizer uses a classmethod __init__, so each instantiation
        # replaces the class-level dict.  Test that language switching works.
        m_en = Modernizer("english")
        assert m_en("caviare") == "caviar"

        m_fr = Modernizer("french")
        assert m_fr("luy") == "lui"

    def test_french_word_not_in_english_dict(self):
        m = Modernizer("english")
        # "luy" is not in the English dict — should come back unchanged.
        assert m("luy") == "luy"

    def test_english_word_not_in_french_dict(self):
        m = Modernizer("french")
        assert m("caviare") == "caviare"
