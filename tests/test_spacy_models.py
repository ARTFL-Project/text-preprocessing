"""
Tests for PreProcessor features that require a Spacy language model:
  - POS-based token filtering (pos_to_keep)
  - Named-entity-based token filtering (ents_to_keep)
  - Spacy lemmatizer (lemmatizer="spacy")

Models used
-----------
  en_core_web_sm   English (tagger, NER, lemmatizer)
  fr_core_news_sm  French  (morphologizer, NER, lemmatizer)

Install with:
  python -m spacy download en_core_web_sm
  python -m spacy download fr_core_news_sm

Design notes
------------
* Loading a Spacy model takes a few seconds.  Each test creates its own
  PreProcessor because TextFetcher stores configuration as class-level state;
  mixing instances within the same test would corrupt that state.

* The English rule-based lemmatizer requires POS annotations, which come from
  the tagger.  load_language_model() only keeps the tagger when pos_to_keep is
  non-empty.  Therefore, English lemmatizer="spacy" tests always combine it
  with a broad pos_to_keep so the tagger stays active.

* ents_to_keep semantics: entities whose type is *not* in the list are
  filtered; tokens with no entity type always pass through.

* process_string() always calls self.nlp() directly, so it honours the full
  Spacy pipeline regardless of do_nlp.  process_texts() routes through
  PreparedDoc when do_nlp=True, which also runs the full pipeline.
"""

import pytest
from text_preprocessing import PreProcessor, Tokens
from conftest import HAMLET, MONTAIGNE

# Convenience: Spacy's broad open-class POS tags.
OPEN_POS = ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]

EN_MODEL = "en_core_web_sm"
FR_MODEL = "fr_core_news_sm"


def words(tokens) -> list[str]:
    return [t.text for t in tokens if t.text and t.text != " "]


def tokens_with_pos(tokens) -> list:
    return [t for t in tokens if t.text and t.text != " "]


# ===========================================================================
# POS-based filtering
# ===========================================================================

class TestPosToKeep:

    # --- English ---

    def test_en_pos_filter_keeps_only_specified_tags(self):
        p = PreProcessor(language="english", language_model=EN_MODEL,
                         pos_to_keep=["NOUN", "PROPN"], workers=1)
        toks = tokens_with_pos(p.process_string(
            "The quick brown fox jumps over the lazy dog."
        ))
        kept_pos = {t.pos_ for t in toks}
        # Only NOUN/PROPN should remain
        assert kept_pos <= {"NOUN", "PROPN"}
        # Content nouns are present
        assert "fox" in [t.text for t in toks]
        assert "dog" in [t.text for t in toks]

    def test_en_pos_filter_removes_non_matching_tokens(self):
        p = PreProcessor(language="english", language_model=EN_MODEL,
                         pos_to_keep=["NOUN"], workers=1)
        toks = tokens_with_pos(p.process_string(
            "The quick brown fox jumps over the lazy dog."
        ))
        texts = [t.text for t in toks]
        # Adjectives, determiners, verbs should be absent
        assert "quick" not in texts   # ADJ
        assert "the" not in texts     # DET
        assert "jumps" not in texts   # VERB

    def test_en_pos_token_attribute_is_set(self):
        p = PreProcessor(language="english", language_model=EN_MODEL,
                         pos_to_keep=["NOUN", "PROPN"], workers=1)
        toks = tokens_with_pos(p.process_string(
            "Albert Einstein discovered relativity."
        ))
        # Every kept token should have a non-empty pos_ attribute
        assert all(t.pos_ != "" for t in toks)

    # --- French ---

    def test_fr_pos_filter_keeps_only_nouns_and_verbs(self):
        p = PreProcessor(language="french", language_model=FR_MODEL,
                         pos_to_keep=["NOUN", "VERB"], workers=1)
        toks = tokens_with_pos(p.process_string(
            "Les chiens noirs courent dans le jardin."
        ))
        kept_pos = {t.pos_ for t in toks}
        assert kept_pos <= {"NOUN", "VERB"}

    def test_fr_pos_filter_removes_determiners(self):
        p = PreProcessor(language="french", language_model=FR_MODEL,
                         pos_to_keep=["NOUN", "VERB"], workers=1)
        toks = tokens_with_pos(p.process_string(
            "Les chiens courent dans le jardin."
        ))
        texts = [t.text for t in toks]
        assert "les" not in texts   # DET
        assert "le" not in texts    # DET
        assert "dans" not in texts  # ADP
        assert "chiens" in texts    # NOUN
        assert "courent" in texts   # VERB

    def test_fr_pos_token_attribute_is_set(self):
        p = PreProcessor(language="french", language_model=FR_MODEL,
                         pos_to_keep=["NOUN", "VERB"], workers=1)
        toks = tokens_with_pos(p.process_string(
            "Les chiens courent dans le jardin."
        ))
        assert all(t.pos_ != "" for t in toks)


# ===========================================================================
# Named-entity filtering
# ===========================================================================

class TestEntsToKeep:

    def test_en_ner_entity_of_kept_type_is_present(self):
        # "Marie Curie" → PERSON, "Paris" / "Sorbonne" → GPE
        p = PreProcessor(language="english", language_model=EN_MODEL,
                         ents_to_keep=["PERSON", "GPE"], workers=1)
        toks = tokens_with_pos(p.process_string(
            "Marie Curie worked at the Sorbonne in Paris."
        ))
        texts = [t.text for t in toks]
        assert "marie" in texts
        assert "curie" in texts

    def test_en_ner_entity_not_in_list_is_filtered(self):
        # "the Nobel Prize in Physics" → WORK_OF_ART, which is not in ents_to_keep
        p = PreProcessor(language="english", language_model=EN_MODEL,
                         ents_to_keep=["PERSON", "GPE"], workers=1)
        toks = tokens_with_pos(p.process_string(
            "Marie Curie won the Nobel Prize in Physics."
        ))
        # "Nobel", "Prize", "Physics" are part of a WORK_OF_ART entity → filtered
        texts = [t.text for t in toks]
        assert "nobel" not in texts
        assert "prize" not in texts

    def test_en_ner_non_entity_tokens_pass_through(self):
        # Tokens with no entity type are not filtered by ents_to_keep alone.
        p = PreProcessor(language="english", language_model=EN_MODEL,
                         ents_to_keep=["PERSON"], workers=1)
        toks = tokens_with_pos(p.process_string(
            "She worked hard and succeeded."
        ))
        texts = [t.text for t in toks]
        assert "worked" in texts
        assert "hard" in texts

    def test_en_ner_entity_type_attribute_is_set_for_kept_entities(self):
        p = PreProcessor(language="english", language_model=EN_MODEL,
                         ents_to_keep=["PERSON", "GPE"], workers=1)
        toks = tokens_with_pos(p.process_string(
            "Marie Curie lived in Paris."
        ))
        person_toks = [t for t in toks if t.ent_type_ == "PERSON"]
        assert len(person_toks) > 0

    def test_fr_ner_person_entity_kept(self):
        p = PreProcessor(language="french", language_model=FR_MODEL,
                         ents_to_keep=["PER", "LOC"], workers=1)
        toks = tokens_with_pos(p.process_string(
            "Victor Hugo a vécu à Paris."
        ))
        texts = [t.text for t in toks]
        # "Victor Hugo" → PER, "Paris" → LOC
        assert "hugo" in texts or "victor" in texts
        assert "paris" in texts


# ===========================================================================
# Spacy lemmatizer
# ===========================================================================

class TestSpacyLemmatizer:

    # --- French (morphologizer-based, works without pos_to_keep) ---

    def test_fr_lemmatizer_inflected_noun(self):
        p = PreProcessor(language="french", language_model=FR_MODEL,
                         lemmatizer="spacy", workers=1)
        toks = tokens_with_pos(p.process_string("les chiens"))
        texts = [t.text for t in toks]
        assert "chien" in texts     # chiens → chien

    def test_fr_lemmatizer_conjugated_verb(self):
        p = PreProcessor(language="french", language_model=FR_MODEL,
                         lemmatizer="spacy", workers=1)
        toks = tokens_with_pos(p.process_string("ils mangent"))
        texts = [t.text for t in toks]
        assert "manger" in texts    # mangent → manger

    def test_fr_lemmatizer_surface_form_preserved(self):
        p = PreProcessor(language="french", language_model=FR_MODEL,
                         lemmatizer="spacy", workers=1)
        toks = tokens_with_pos(p.process_string("les chiens"))
        # surface_form should be the original, text should be the lemma
        noun_tok = next((t for t in toks if t.text == "chien"), None)
        assert noun_tok is not None
        assert noun_tok.surface_form == "chiens"

    # --- English (rule-based, requires tagger; enable via pos_to_keep) ---

    def test_en_lemmatizer_plural_noun(self):
        p = PreProcessor(language="english", language_model=EN_MODEL,
                         lemmatizer="spacy", pos_to_keep=OPEN_POS, workers=1)
        toks = tokens_with_pos(p.process_string("dogs are running in cities"))
        texts = [t.text for t in toks]
        assert "dog" in texts       # dogs → dog
        assert "city" in texts      # cities → city

    def test_en_lemmatizer_verb_form(self):
        p = PreProcessor(language="english", language_model=EN_MODEL,
                         lemmatizer="spacy", pos_to_keep=OPEN_POS, workers=1)
        toks = tokens_with_pos(p.process_string("dogs are running in cities"))
        texts = [t.text for t in toks]
        assert "run" in texts       # running → run

    def test_en_lemmatizer_surface_form_preserved(self):
        p = PreProcessor(language="english", language_model=EN_MODEL,
                         lemmatizer="spacy", pos_to_keep=OPEN_POS, workers=1)
        toks = tokens_with_pos(p.process_string("dogs running"))
        noun_tok = next((t for t in toks if t.text == "dog"), None)
        assert noun_tok is not None
        assert noun_tok.surface_form == "dogs"

    def test_en_lemmatizer_does_not_alter_already_base_form(self):
        p = PreProcessor(language="english", language_model=EN_MODEL,
                         lemmatizer="spacy", pos_to_keep=OPEN_POS, workers=1)
        toks = tokens_with_pos(p.process_string("cat dog run"))
        texts = [t.text for t in toks]
        assert "cat" in texts
        assert "dog" in texts


# ===========================================================================
# Integration: language model + corpus files
# ===========================================================================

class TestSpacyWithCorpus:

    def test_en_pos_filter_on_hamlet(self):
        p = PreProcessor(language="english", language_model=EN_MODEL,
                         pos_to_keep=["NOUN", "PROPN"], workers=1)
        results = list(p.process_texts([HAMLET], progress=False))
        toks = tokens_with_pos(results[0])
        kept_pos = {t.pos_ for t in toks}
        assert kept_pos <= {"NOUN", "PROPN"}
        assert len(toks) > 100

    def test_fr_lemmatizer_on_germinal(self):
        from conftest import GERMINAL
        p = PreProcessor(language="french", language_model=FR_MODEL,
                         lemmatizer="spacy", workers=1)
        results = list(p.process_texts([GERMINAL], progress=False))
        texts = {t.text for t in tokens_with_pos(results[0])}
        # Common French infinitives should appear from conjugated forms
        assert "être" in texts or "avoir" in texts or "aller" in texts

    def test_fr_pos_filter_on_germinal(self):
        from conftest import GERMINAL
        p = PreProcessor(language="french", language_model=FR_MODEL,
                         pos_to_keep=["NOUN"], workers=1)
        results = list(p.process_texts([GERMINAL], progress=False))
        toks = tokens_with_pos(results[0])
        assert all(t.pos_ == "NOUN" for t in toks)
        assert len(toks) > 100
