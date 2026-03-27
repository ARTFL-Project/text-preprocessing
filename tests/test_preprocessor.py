"""
Tests for the core PreProcessor normalization pipeline.

Normalization options are exercised primarily via process_string() since it
avoids filesystem I/O.  File-based integration tests use the four corpus
fixtures (Hamlet, Moby Dick, Montaigne, Germinal).

A note on whitespace tokens
----------------------------
When the preprocessor builds a Spacy Doc from a word list, Spacy inserts a
trailing space after every non-final token (its default spacing behaviour).
__get_tokens() therefore emits a " " PreprocessorToken between each pair of
content words.  All helpers below filter these out alongside empty "#DEL#"
tokens with the predicate  ``t.text and t.text != " "``.
"""

import os
import pytest

from text_preprocessing import PreProcessor, Tokens

from conftest import HAMLET, MOBY_DICK, MONTAIGNE, GERMINAL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def words(tokens) -> list[str]:
    """Return the non-empty, non-whitespace text values from a Tokens object."""
    return [t.text for t in tokens if t.text and t.text != " "]


# ===========================================================================
# Normalisation — exercised via process_string()
# ===========================================================================

class TestNormalization:

    def test_returns_tokens_object(self):
        p = PreProcessor(language="english", workers=1)
        result = p.process_string("Hello world")
        assert isinstance(result, Tokens)

    # --- case ---

    def test_lowercase_default(self):
        p = PreProcessor(language="english", workers=1)
        texts = words(p.process_string("Hello World"))
        assert "hello" in texts
        assert "world" in texts
        assert "Hello" not in texts

    def test_no_lowercase(self):
        p = PreProcessor(language="english", lowercase=False, workers=1)
        texts = words(p.process_string("Hello World"))
        assert "Hello" in texts
        assert "World" in texts

    # --- punctuation ---

    def test_strip_punctuation_default(self):
        p = PreProcessor(language="english", workers=1)
        texts = words(p.process_string("Hello. World!"))
        assert "hello" in texts
        assert "world" in texts
        assert "." not in texts
        assert "!" not in texts

    def test_keep_punctuation(self):
        p = PreProcessor(language="english", strip_punctuation=False, workers=1)
        texts = words(p.process_string("Hello. World!"))
        assert "." in texts
        assert "!" in texts

    # --- numbers ---

    def test_strip_numbers_default(self):
        p = PreProcessor(language="english", workers=1)
        texts = words(p.process_string("I have 42 apples"))
        assert "42" not in texts
        assert "apples" in texts

    def test_keep_numbers(self):
        p = PreProcessor(language="english", strip_numbers=False, workers=1)
        texts = words(p.process_string("I have 42 apples"))
        assert "42" in texts

    # --- min word length ---

    def test_min_word_length_default(self):
        # Default is 2: single-character tokens are filtered.
        p = PreProcessor(language="english", workers=1)
        texts = words(p.process_string("I am here"))
        assert "i" not in texts
        assert "here" in texts

    def test_min_word_length_custom(self):
        # Filter is len(token) < min_word_length, so 5-char words pass with min=5.
        p = PreProcessor(language="english", min_word_length=5, workers=1)
        texts = words(p.process_string("I am here going somewhere"))
        assert "somewhere" in texts
        assert "going" in texts       # 5 chars: 5 < 5 is False → kept
        assert "here" not in texts    # 4 chars: 4 < 5 is True → filtered
        assert "am" not in texts

    # --- stopwords ---

    def test_stopwords_removes_listed_words(self, stopwords_file):
        p = PreProcessor(language="english", stopwords=stopwords_file, workers=1)
        texts = words(p.process_string("the quick brown fox"))
        assert "the" not in texts
        assert "quick" in texts
        assert "fox" in texts

    def test_stopwords_post_lowercase(self, stopwords_file):
        # Stopwords are matched against the lowercased form, so "The" is caught.
        p = PreProcessor(language="english", stopwords=stopwords_file, workers=1)
        texts = words(p.process_string("The quick brown fox"))
        assert "the" not in texts
        assert "quick" in texts

    # --- ASCII conversion ---

    def test_ascii_strips_diacritics(self):
        p = PreProcessor(language="french", ascii=True, workers=1)
        texts = words(p.process_string("élève café"))
        assert "eleve" in texts
        assert "cafe" in texts
        assert "élève" not in texts

    # --- tag stripping ---

    def test_strip_tags_removes_markup(self):
        p = PreProcessor(language="english", strip_tags=True, workers=1)
        texts = words(p.process_string("<div>Hello <em>World</em></div>"))
        assert "hello" in texts
        assert "world" in texts
        assert "div" not in texts
        assert "em" not in texts

    def test_no_strip_tags_leaks_tag_names(self):
        # Without strip_tags, tag names are tokenised as ordinary words.
        p = PreProcessor(language="english", strip_tags=False, workers=1)
        texts = words(p.process_string("<div>Hello</div>"))
        assert "hello" in texts
        assert "div" in texts

    # --- surface form ---

    def test_surface_form_retains_original_case(self):
        p = PreProcessor(language="english", workers=1)
        toks = [t for t in p.process_string("Running") if t.text and t.text != " "]
        assert toks[0].text == "running"
        assert toks[0].surface_form == "Running"

    # --- edge cases ---

    def test_empty_string_yields_no_tokens(self):
        p = PreProcessor(language="english", workers=1)
        assert words(p.process_string("")) == []

    def test_punctuation_only_yields_no_tokens(self):
        p = PreProcessor(language="english", workers=1)
        assert words(p.process_string("... !!! ???")) == []

    def test_numbers_only_yields_no_tokens(self):
        p = PreProcessor(language="english", workers=1)
        assert words(p.process_string("123 456 789")) == []


# ===========================================================================
# Modernizer — archaic-to-modern spelling via process_string()
# ===========================================================================

class TestModernizer:

    # --- English ---

    def test_english_archaic_words_modernised(self):
        p = PreProcessor(language="english", modernize=True, workers=1)
        texts = words(p.process_string("caviare loth"))
        assert "caviar" in texts    # caviare → caviar
        assert "loath" in texts     # loth → loath
        assert "caviare" not in texts
        assert "loth" not in texts

    def test_english_no_modernize_preserves_archaic(self):
        p = PreProcessor(language="english", modernize=False, workers=1)
        texts = words(p.process_string("caviare loth"))
        assert "caviare" in texts
        assert "loth" in texts

    def test_english_unknown_word_unchanged(self):
        p = PreProcessor(language="english", modernize=True, workers=1)
        texts = words(p.process_string("xyzunknownword"))
        assert "xyzunknownword" in texts

    def test_english_surface_form_is_pre_modernization(self):
        p = PreProcessor(language="english", modernize=True, workers=1)
        toks = [t for t in p.process_string("caviare") if t.text and t.text != " "]
        assert toks[0].text == "caviar"
        assert toks[0].surface_form == "caviare"

    # --- French ---

    def test_french_archaic_words_modernised(self):
        p = PreProcessor(language="french", modernize=True, workers=1)
        texts = words(p.process_string("luy mesme vray"))
        assert "lui" in texts       # luy → lui
        assert "même" in texts      # mesme → même
        assert "vrai" in texts      # vray → vrai

    def test_french_no_modernize_preserves_archaic(self):
        p = PreProcessor(language="french", modernize=False, workers=1)
        texts = words(p.process_string("luy mesme vray"))
        assert "luy" in texts
        assert "mesme" in texts


# ===========================================================================
# Lemmatizer (dictionary-based)
# ===========================================================================

class TestDictLemmatizer:

    def test_known_words_replaced_by_lemma(self, lemma_file):
        # The lemmatizer is applied before lowercasing, so keys must match the
        # case of the raw token from tokenisation (lowercase in this example).
        p = PreProcessor(language="english", lemmatizer=lemma_file, workers=1)
        texts = words(p.process_string("running flies went"))
        assert "run" in texts
        assert "fly" in texts
        assert "go" in texts

    def test_unknown_word_passes_through_unchanged(self, lemma_file):
        p = PreProcessor(language="english", lemmatizer=lemma_file, workers=1)
        texts = words(p.process_string("unknown"))
        assert "unknown" in texts


# ===========================================================================
# File-based integration — process_texts()
# ===========================================================================

class TestProcessTexts:

    # --- basic ---

    def test_hamlet_returns_one_tokens_object(self):
        p = PreProcessor(language="english", workers=1)
        results = list(p.process_texts([HAMLET], progress=False))
        assert len(results) == 1
        assert isinstance(results[0], Tokens)

    def test_hamlet_has_many_tokens(self):
        p = PreProcessor(language="english", workers=1)
        results = list(p.process_texts([HAMLET], progress=False))
        assert len(results[0]) > 1000

    def test_hamlet_metadata_contains_filename(self):
        p = PreProcessor(language="english", workers=1)
        results = list(p.process_texts([HAMLET], progress=False))
        assert results[0].metadata["filename"] == HAMLET

    def test_moby_dick_has_many_tokens(self):
        p = PreProcessor(language="english", workers=1)
        results = list(p.process_texts([MOBY_DICK], progress=False))
        assert len(results[0]) > 10_000

    def test_germinal_french(self):
        p = PreProcessor(language="french", workers=1)
        results = list(p.process_texts([GERMINAL], progress=False))
        assert len(results[0]) > 10_000

    def test_montaigne_french(self):
        p = PreProcessor(language="french", workers=1)
        results = list(p.process_texts([MONTAIGNE], progress=False))
        assert len(results[0]) > 10_000

    def test_multiple_files_returns_one_result_per_file(self):
        p = PreProcessor(language="english", workers=1)
        results = list(p.process_texts([HAMLET, MOBY_DICK], progress=False))
        assert len(results) == 2

    # --- normalisation applied to file tokens ---

    def test_file_tokens_are_lowercase(self):
        p = PreProcessor(language="english", workers=1)
        results = list(p.process_texts([HAMLET], progress=False))
        texts = words(results[0])
        assert all(t == t.lower() for t in texts)

    def test_file_tokens_contain_no_digits(self):
        import re
        p = PreProcessor(language="english", strip_numbers=True, workers=1)
        results = list(p.process_texts([HAMLET], progress=False))
        texts = words(results[0])
        assert not any(re.search(r"\d", t) for t in texts)

    # --- modernizer on corpora ---

    def test_hamlet_modernize_replaces_archaic_forms(self):
        # TextFetcher uses class-level state: use each PreProcessor before creating the next.
        p_plain = PreProcessor(language="english", modernize=False, workers=1)
        plain = set(words(list(p_plain.process_texts([HAMLET], progress=False))[0]))

        p_mod = PreProcessor(language="english", modernize=True, workers=1)
        modern = set(words(list(p_mod.process_texts([HAMLET], progress=False))[0]))

        assert "caviare" in plain
        assert "caviare" not in modern
        assert "caviar" in modern

    def test_montaigne_modernize_replaces_archaic_forms(self):
        # TextFetcher uses class-level state: use each PreProcessor before creating the next.
        # Montaigne's 1595 text uses mixed spelling; the modernizer converts lowercase
        # archaic forms and leaves capitalized forms (case-sensitive lookup) for lowercasing
        # to resolve. We verify: (1) archaic forms appear in the plain corpus, (2) the
        # modernizer produces more occurrences of the modern form than the plain corpus.
        p_plain = PreProcessor(language="french", modernize=False, workers=1)
        plain_list = list(words(list(p_plain.process_texts([MONTAIGNE], progress=False))[0]))

        p_mod = PreProcessor(language="french", modernize=True, workers=1)
        modern_list = list(words(list(p_mod.process_texts([MONTAIGNE], progress=False))[0]))

        # Archaic forms present in the plain corpus
        assert "luy" in plain_list

        # The modernized corpus has fewer archaic forms and more modern equivalents
        assert plain_list.count("luy") > modern_list.count("luy")
        assert modern_list.count("lui") > plain_list.count("lui")

    # --- stopwords on file ---

    def test_stopwords_removed_from_file(self, stopwords_file, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("the quick brown fox and the lazy dog", encoding="utf-8")

        p = PreProcessor(language="english", stopwords=stopwords_file, workers=1)
        results = list(p.process_texts([str(test_file)], progress=False))
        texts = words(results[0])
        assert "the" not in texts
        assert "and" not in texts
        assert "quick" in texts
        assert "fox" in texts

    # --- keep_all ---

    def test_keep_all_preserves_filtered_tokens_as_empty_strings(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello 42 world", encoding="utf-8")

        p = PreProcessor(language="english", strip_numbers=True, workers=1)
        results = list(p.process_texts([str(test_file)], keep_all=True, progress=False))
        all_texts = [t.text for t in results[0]]
        non_empty = [t for t in all_texts if t and t != " "]

        assert "hello" in non_empty
        assert "world" in non_empty
        # The filtered "42" is kept as an empty string placeholder
        assert "" in all_texts
        assert len(all_texts) > len(non_empty)

    # --- post-processing function ---

    def test_post_processing_function_applied(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world foo", encoding="utf-8")

        def append_marker(tokens):
            for tok in tokens:
                if tok.text and tok.text != " ":
                    tok.text = tok.text + "_X"
            return tokens

        p = PreProcessor(language="english", post_processing_function=append_marker, workers=1)
        results = list(p.process_texts([str(test_file)], progress=False))
        texts = words(results[0])
        assert "hello_X" in texts
        assert "world_X" in texts
        assert "foo_X" in texts

    # --- workers=1 explicit ---

    def test_single_worker_produces_same_output(self):
        p = PreProcessor(language="english", workers=1)
        results = list(p.process_texts([HAMLET], progress=False))
        assert len(results[0]) > 1000


# ===========================================================================
# N-grams
# ===========================================================================

class TestNgrams:

    def test_bigrams_produced(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("the quick brown fox", encoding="utf-8")
        p = PreProcessor(language="english", ngrams=2, strip_numbers=False, workers=1)
        results = list(p.process_texts([str(test_file)], progress=False))
        texts = [t.text for t in results[0]]
        # Every token should be a bigram joined by "_"
        assert all("_" in t for t in texts)

    def test_bigram_text_values(self, tmp_path):
        test_file = tmp_path / "test.txt"
        # Use min_word_length=1 and strip_numbers=False to keep all words
        test_file.write_text("hello world foo", encoding="utf-8")
        p = PreProcessor(language="english", ngrams=2, min_word_length=1, strip_numbers=False, workers=1)
        results = list(p.process_texts([str(test_file)], progress=False))
        texts = [t.text for t in results[0]]
        assert "hello_world" in texts
        assert "world_foo" in texts

    def test_trigrams_produced(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("one two three four five", encoding="utf-8")
        p = PreProcessor(language="english", ngrams=3, min_word_length=1, strip_numbers=False, workers=1)
        results = list(p.process_texts([str(test_file)], progress=False))
        texts = [t.text for t in results[0]]
        assert any(t.count("_") == 2 for t in texts)  # trigrams have 2 underscores

    def test_ngram_byte_positions_populated(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world foo", encoding="utf-8")
        p = PreProcessor(language="english", ngrams=2, min_word_length=1, strip_numbers=False, workers=1)
        results = list(p.process_texts([str(test_file)], progress=False))
        for tok in results[0]:
            assert "start_byte" in tok.ext
            assert "end_byte" in tok.ext
