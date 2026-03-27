"""
Tests for the PhiloLogic database input path (is_philo_db=True).

Fixtures are the Balzac corpus (5 LZ4 files) from a real PhiloLogic 4 database,
located at tests/fixtures/philo/data/words_and_philo_ids/.  The accompanying
toms.db lives at tests/fixtures/philo/data/toms.db, which is where the code
resolves it (two path components up from the LZ4 file, treating the file as a
directory node).

Directory layout expected by the code
--------------------------------------
  tests/fixtures/philo/data/
    toms.db
    words_and_philo_ids/
      1.lz4 … 5.lz4
    TEXT/           (dir only; real source files not required for these tests)
"""

import os
import shutil
import pytest

from text_preprocessing import PreProcessor, Tokens

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PHILO_DATA = os.path.join(os.path.dirname(__file__), "fixtures", "philo", "data")
PHILO_IDS = os.path.join(PHILO_DATA, "words_and_philo_ids")
LZ4_FILES = [os.path.join(PHILO_IDS, f"{i}.lz4") for i in range(1, 6)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def content_tokens(tokens):
    """Return non-empty, non-whitespace tokens."""
    return [t for t in tokens if t.text and t.text != " "]


# ===========================================================================
# doc-level processing
# ===========================================================================

class TestPhiloDoc:

    def test_one_result_per_file(self):
        p = PreProcessor(language="french", is_philo_db=True, workers=1)
        results = list(p.process_texts(LZ4_FILES, progress=False))
        assert len(results) == 5

    def test_result_is_tokens_instance(self):
        p = PreProcessor(language="french", is_philo_db=True, workers=1)
        results = list(p.process_texts(LZ4_FILES[:1], progress=False))
        assert isinstance(results[0], Tokens)

    def test_doc_has_many_tokens(self):
        p = PreProcessor(language="french", is_philo_db=True, workers=1)
        results = list(p.process_texts(LZ4_FILES[:1], progress=False))
        assert len(content_tokens(results[0])) > 1000

    # --- metadata from toms.db ---

    def test_metadata_has_author(self):
        p = PreProcessor(language="french", is_philo_db=True, workers=1)
        results = list(p.process_texts(LZ4_FILES[:1], progress=False))
        assert results[0].metadata["author"] == "Balzac, Honoré de"

    def test_metadata_has_title(self):
        p = PreProcessor(language="french", is_philo_db=True, workers=1)
        results = list(p.process_texts(LZ4_FILES[:1], progress=False))
        assert results[0].metadata["title"] == "La Fausse Maîtresse"

    def test_metadata_has_year(self):
        p = PreProcessor(language="french", is_philo_db=True, workers=1)
        results = list(p.process_texts(LZ4_FILES[:1], progress=False))
        assert results[0].metadata["year"] == 1842

    def test_metadata_has_philo_id(self):
        p = PreProcessor(language="french", is_philo_db=True, workers=1)
        results = list(p.process_texts(LZ4_FILES[:1], progress=False))
        assert "philo_id" in results[0].metadata

    def test_metadata_has_byte_range(self):
        p = PreProcessor(language="french", is_philo_db=True, workers=1)
        results = list(p.process_texts(LZ4_FILES[:1], progress=False))
        assert "start_byte" in results[0].metadata
        assert "end_byte" in results[0].metadata

    def test_metadata_philo_type_is_doc(self):
        p = PreProcessor(language="french", is_philo_db=True, workers=1)
        results = list(p.process_texts(LZ4_FILES[:1], progress=False))
        assert results[0].metadata["philo_type"] == "doc"

    # --- token ext ---

    def test_token_ext_has_byte_positions(self):
        p = PreProcessor(language="french", is_philo_db=True, workers=1)
        results = list(p.process_texts(LZ4_FILES[:1], progress=False))
        tok = content_tokens(results[0])[0]
        assert "start_byte" in tok.ext
        assert "end_byte" in tok.ext
        assert isinstance(tok.ext["start_byte"], int)

    def test_token_ext_has_position(self):
        p = PreProcessor(language="french", is_philo_db=True, workers=1)
        results = list(p.process_texts(LZ4_FILES[:1], progress=False))
        tok = content_tokens(results[0])[0]
        assert "position" in tok.ext

    def test_token_ext_has_philo_type(self):
        p = PreProcessor(language="french", is_philo_db=True, workers=1)
        results = list(p.process_texts(LZ4_FILES[:1], progress=False))
        tok = content_tokens(results[0])[0]
        assert tok.ext["philo_type"] == "word"

    def test_surface_form_is_original_token(self):
        p = PreProcessor(language="french", is_philo_db=True, workers=1)
        results = list(p.process_texts(LZ4_FILES[:1], progress=False))
        tok = content_tokens(results[0])[0]
        assert tok.surface_form == tok.ext["token"]

    # --- normalisation still applied ---

    def test_tokens_are_lowercased(self):
        p = PreProcessor(language="french", is_philo_db=True, workers=1)
        results = list(p.process_texts(LZ4_FILES[:1], progress=False))
        texts = [t.text for t in content_tokens(results[0])]
        assert all(t == t.lower() for t in texts)

    def test_numbers_stripped(self):
        import re
        p = PreProcessor(language="french", is_philo_db=True, strip_numbers=True, workers=1)
        results = list(p.process_texts(LZ4_FILES[:1], progress=False))
        texts = [t.text for t in content_tokens(results[0])]
        assert not any(re.search(r"\d", t) for t in texts)

    # --- fallback when toms.db is absent ---

    def test_no_toms_db_falls_back_to_filename(self, tmp_path):
        # Copy just the LZ4 file; omit toms.db so fetch_metadata falls back.
        ids_dir = tmp_path / "data" / "words_and_philo_ids"
        ids_dir.mkdir(parents=True)
        shutil.copy(LZ4_FILES[0], ids_dir / "1.lz4")

        p = PreProcessor(language="french", is_philo_db=True, workers=1)
        results = list(p.process_texts([str(ids_dir / "1.lz4")], progress=False))
        assert len(results) == 1
        assert results[0].metadata == {"filename": "1.lz4"}


# ===========================================================================
# paragraph-level processing
# ===========================================================================

class TestPhiloPara:

    def test_para_yields_multiple_objects_per_file(self):
        p = PreProcessor(language="french", is_philo_db=True,
                         text_object_type="para", workers=1)
        results = list(p.process_texts(LZ4_FILES[:1], progress=False))
        assert len(results) > 10

    def test_para_metadata_has_philo_type(self):
        p = PreProcessor(language="french", is_philo_db=True,
                         text_object_type="para", workers=1)
        results = list(p.process_texts(LZ4_FILES[:1], progress=False))
        assert results[0].metadata["philo_type"] == "para"

    def test_para_metadata_inherits_doc_fields(self):
        p = PreProcessor(language="french", is_philo_db=True,
                         text_object_type="para", workers=1)
        results = list(p.process_texts(LZ4_FILES[:1], progress=False))
        # Paragraph metadata inherits author and title from the parent doc.
        assert results[0].metadata["author"] == "Balzac, Honoré de"
        assert results[0].metadata["title"] == "La Fausse Maîtresse"

    def test_para_byte_positions_are_monotonically_increasing(self):
        p = PreProcessor(language="french", is_philo_db=True,
                         text_object_type="para", workers=1)
        results = list(p.process_texts(LZ4_FILES[:1], progress=False))
        starts = [int(r.metadata["start_byte"]) for r in results]
        assert starts == sorted(starts)

    def test_para_word_count_matches_token_count(self):
        p = PreProcessor(language="french", is_philo_db=True,
                         text_object_type="para", workers=1)
        results = list(p.process_texts(LZ4_FILES[:1], progress=False))
        # word_count in metadata should roughly match actual content tokens.
        # (They won't be identical because normalisation filters some tokens.)
        r = results[0]
        declared = int(r.metadata["word_count"])
        actual = len(content_tokens(r))
        assert actual <= declared


# ===========================================================================
# n-grams with PhiloLogic data (byte positions present — should not KeyError)
# ===========================================================================

class TestPhiloNgrams:

    def test_bigrams_work_with_philo_data(self):
        p = PreProcessor(language="french", is_philo_db=True, ngrams=2, workers=1)
        results = list(p.process_texts(LZ4_FILES[:1], progress=False))
        texts = [t.text for t in results[0]]
        assert len(texts) > 0
        assert all("_" in t for t in texts)

    def test_bigram_tokens_have_byte_positions(self):
        p = PreProcessor(language="french", is_philo_db=True, ngrams=2, workers=1)
        results = list(p.process_texts(LZ4_FILES[:1], progress=False))
        for tok in results[0]:
            assert "start_byte" in tok.ext
            assert "end_byte" in tok.ext

    def test_bigram_start_before_end(self):
        p = PreProcessor(language="french", is_philo_db=True, ngrams=2, workers=1)
        results = list(p.process_texts(LZ4_FILES[:1], progress=False))
        for tok in results[0]:
            assert tok.ext["start_byte"] <= tok.ext["end_byte"]


# ===========================================================================
# All five files together
# ===========================================================================

class TestPhiloAllFiles:

    def test_all_five_files_processed(self):
        p = PreProcessor(language="french", is_philo_db=True, workers=1)
        results = list(p.process_texts(LZ4_FILES, progress=False))
        assert len(results) == 5

    def test_each_file_has_distinct_title(self):
        p = PreProcessor(language="french", is_philo_db=True, workers=1)
        results = list(p.process_texts(LZ4_FILES, progress=False))
        titles = [r.metadata["title"] for r in results]
        assert len(set(titles)) == 5  # five distinct Balzac works
