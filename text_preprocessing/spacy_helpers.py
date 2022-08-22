import re
from collections import deque
from itertools import combinations
from typing import Any, Deque, Dict, List, Callable

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
from Stemmer import Stemmer

from preprocessor_helpers import normalize
from modernizer import Modernizer


# Updated as of 3/22/2021
SPACY_LANGUAGE_MODEL_MAP: Dict[str, List[str]] = {
    "danish": ["da_core_news_sm", "da_core_news_md", "da_core_news_lg"],
    "german": ["de_core_news_sm", "de_core_news_md", "de_core_news_lg", "de_dep_news_trf"],
    "greek": ["el_core_news_sm", "el_core_news_md", "el_core_news_lg"],
    "english": ["en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf"],
    "spanish": ["es_core_news_sm", "es_core_news_md", "es_core_news_lg", "es_dep_news_trf"],
    "french": ["fr_core_news_sm", "fr_core_news_md", "fr_core_news_lg", "fr_dep_news_trf"],
    "italian": ["it_core_news_sm", "it_core_news_md", "it_core_news_lg"],
    "japanese": ["ja_core_news_sm", "ja_core_news_md", "ja_core_news_lg"],
    "lithuanian": ["lt_core_news_sm", "lt_core_news_md", "lt_core_news_lg"],
    "norwegian bokmål": ["nb_core_news_sm", "nb_core_news_md", "nb_core_news_lg"],
    "dutch": ["nl_core_news_sm", "nl_core_news_md", "nl_core_news_lg"],
    "polish": ["pl_core_news_sm", "pl_core_news_md", "pl_core_news_lg"],
    "portuguese": ["pt_core_news_sm", "pt_core_news_md", "pt_core_news_lg"],
    "romanian": ["ro_core_news_sm", "ro_core_news_md", "ro_core_news_lg"],
    "russian": ["ru_core_news_sm", "ru_core_news_md", "ru_core_news_lg"],
    "multi-language": ["xx_ent_wiki_sm", "xx_sent_ud_sm"],
    "chinese": ["zh_core_web_sm", "zh_core_web_md", "zh_core_web_lg", "zh_core_web_trf"],
}


TRIM_LAST_SLASH = re.compile(r"/\Z")
TAGS = re.compile(r"<[^>]+>")
WORD_CHARS = re.compile(r"\w+")

Token.set_extension("surface_form", default="")
Token.set_extension("ext", default={})


def check_for_updates(language):
    """Check for spacy language model updates"""
    import requests

    response = requests.get("https://raw.githubusercontent.com/explosion/spaCy/master/website/meta/languages.json")
    if response.status_code == 404:
        print("Unable to fetch language information from Spacy GitHub")
        return []
    try:
        languages = response.json()
        models = {lang["name"].lower(): lang["models"] for lang in languages["languages"] if "models" in lang}
        model: List[str] = models[language]
    except KeyError:
        return []
    return model


def tokens_to_doc(tokens, vocab, metadata):
    doc = Doc(vocab=vocab, words=[t.text for t in tokens], user_data=metadata)
    for index, token in enumerate(tokens):
        doc[index]._.ext = token.ext
    return doc


class PassThroughTokenizer:
    """Used to bypass Spacy tokenizer"""

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, tokens):
        try:
            return spacy.tokens.Doc(self.vocab, words=[t.text for t in tokens])  # type: ignore
        except AttributeError:
            return spacy.tokens.Doc(self.vocab, words=tokens)  # type: ignore


class PlainTextTokenizer:
    """Tokenizer for plain text"""

    def __init__(self, vocab, **tokenizer_config):
        self.vocab = vocab
        self.strip_tags = tokenizer_config["strip_tags"]
        self.token_regex = tokenizer_config["token_regex"]
        if tokenizer_config["modernize"]:
            self.modernizer: Callable = Modernizer(tokenizer_config["language"])
            self.modernize = True
        else:
            self.modernize = False

    def __call__(self, text_path) -> Doc:
        with open(text_path, encoding="utf-8") as text_file:
            text = text_file.read()
        if self.strip_tags is True:
            text = self.remove_tags(text)
        words = []
        for match in self.token_regex.finditer(text):
            if self.modernize is True:
                words.append(self.modernizer(match[0]))
            else:
                words.append(match[0])
        return Doc(self.vocab, words=words)

    def remove_tags(self, text: str):
        end_header_index: int = text.rfind("</teiHeader>")
        if end_header_index != -1:
            end_header_index += 12
            text = text[end_header_index:]
        text = TAGS.sub("", text)
        return text


@Language.factory(
    "normalizer",
    default_config={
        "convert_entities": True,
        "lowercase": True,
        "strip_punctuation": True,
        "strip_numbers": True,
        "stemmer": False,
        "ascii": False,
        "hash_tokens": False,
        "min_word_length": 1,
        "stopwords": None,
        "language": "english",
        "lemmatizer": False,
    },
)
def init_normalizer(
    nlp,
    name,
    convert_entities,
    lowercase,
    strip_punctuation,
    strip_numbers,
    stemmer,
    ascii,
    hash_tokens,
    min_word_length,
    stopwords,
    language,
    lemmatizer,
):
    if stopwords is None:
        stopwords = []
    else:
        stopwords = set(stopwords)
    if stemmer is True:
        stemmer = Stemmer(language)
        stemmer.maxCacheSize = 50000

    def normalizer(doc) -> Doc:
        """Normalize tokens in a doc"""
        tokens = []
        lemmas = []
        ents = []
        pos = []
        surface_forms = []
        if doc[0].pos_ == 0:
            pos = None
        if doc[0].ent_type == 0:
            ents = None
        for token in doc:
            token_text, surface_form = normalize(
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
            )
            surface_forms.append(surface_form)
            tokens.append(token_text)
            lemmas.append(token.lemma_)
            if pos is not None:
                pos.append(token.pos_)
            if ents is not None:
                if token.ent_type_:
                    ents.append(f"{token.ent_iob_}-{token.ent_type_}")
                else:
                    ents.append(token.ent_iob_)
        norm_doc = Doc(vocab=doc.vocab, words=tokens, lemmas=lemmas, pos=pos, ents=ents)
        for index, surface_form in enumerate(surface_forms):
            norm_doc[index]._.surface_form = surface_form
        return norm_doc

    return normalizer


@Language.factory(
    "filter_tokens",
    default_config={
        "pos_to_keep": [],
        "ents_to_keep": [],
    },
)
def init_filter_tokens(nlp, name, pos_to_keep, ents_to_keep):
    if pos_to_keep:
        filter_pos = True
    else:
        filter_pos = False
    if ents_to_keep:
        filter_ents = True
    else:
        filter_ents = False

    def filter_tokens(doc) -> Doc:
        """Normalize tokens in a doc"""
        words = []
        lemmas = []
        ents = []
        pos = []
        for token in doc:
            keep_ent = False
            if filter_ents is True:
                if token.ent_type_ in ents_to_keep:
                    keep_ent = True
            if keep_ent is False and filter_pos is True:
                if token.pos_ not in pos_to_keep:
                    continue
            words.append(token.text)
            lemmas.append(token.lemma_)
            if token.ent_type_:
                ents.append(f"{token.ent_iob_}-{token.ent_type_}")
            else:
                ents.append(token.ent_iob_)
            pos.append(token.pos_)
        filtered_doc = Doc(nlp.vocab, words=words, lemmas=lemmas, pos=pos, ents=ents)
        return filtered_doc

    return filter_tokens


@Language.factory(
    "ngram_generator",
    default_config={"ngram_window": 1, "ngram_size": 2, "ngram_word_order": True},
)
def init_generate_ngrams(nlp, name, ngram_window, ngram_size, ngram_word_order):
    def generate_ngrams(doc) -> Doc:
        ngrams: List[str] = []
        ngram: Deque[Token] = deque()
        ngram_text: str
        extras = []
        for token in doc:
            ngram.append(token)
            if len(ngram) == ngram_window:
                for local_ngram in combinations(ngram, ngram_size):
                    ext: Dict[str, Any] = local_ngram[0]._.ext
                    if ngram_word_order is True:
                        ngram_text = "_".join(t.text for t in local_ngram)
                        ext["end_byte"] = local_ngram[-1]._.ext["end_byte"]
                    else:
                        ngram_text = "_".join(t.text for t in sorted(local_ngram, key=lambda x: x.text))
                        ngram_sorted_position = sorted(local_ngram, key=lambda x: x._.ext["start_byte"])
                        ext["start_byte"] = ngram_sorted_position[0]._.ext["start_byte"]
                        ext["end_byte"] = ngram_sorted_position[-1]._.ext["end_byte"]
                    ngrams.append(ngram_text)
                    extras.append(ext)
                ngram.popleft()
        ngram_doc = Doc(nlp.vocab, words=ngrams)
        for index, ext in enumerate(extras):
            ngram_doc[index]._.ext = extras
        return ngram_doc

    return generate_ngrams


def load_language_model(
    language,
    tokenizer_config: Dict[str, Any],
    normalizer_config,
    filter_config: Dict[str, Any],
    ngram_config: Dict[str, Any],
    is_philo_db: bool,
):
    """Load language model based on name"""
    nlp = None
    if any(
        (
            filter_config["pos_to_keep"] is not None,
            filter_config["ents_to_keep"] is not None,
            normalizer_config["lemmatizer"] == "spacy",
        )
    ):
        language = language.lower()
        try:
            possible_models = SPACY_LANGUAGE_MODEL_MAP[language]
        except KeyError:
            try:
                possible_models = check_for_updates(language)
            except KeyError:
                print(f"Spacy does not support the {language} language.")
                exit(-1)
        for model in possible_models:
            try:
                if filter_config is None or not filter_config["ents_to_keep"]:
                    nlp = spacy.load(model, exclude=["parser", "ner", "textcat"])
                else:
                    nlp = spacy.load(model)
                if is_philo_db:
                    nlp.tokenizer = PassThroughTokenizer(nlp.vocab)
                else:
                    nlp.tokenizer = PlainTextTokenizer(nlp.vocab, **tokenizer_config)
            except OSError:
                pass
            if nlp is not None:
                break
        if nlp is None:
            print(f"No Spacy model installed for the {language} language. Stopping...")
            exit(-1)
        if any(
            (
                filter_config["pos_to_keep"] is not None,
                filter_config["ents_to_keep"] is not None,
            )
        ):
            nlp.add_pipe("filter_tokens", config=filter_config)
    else:
        nlp = spacy.blank("en")
        nlp.tokenizer = PlainTextTokenizer(nlp.vocab, **tokenizer_config)
    nlp.add_pipe("normalizer", config={"language": language, **normalizer_config})
    if ngram_config is not None:
        nlp.add_pipe("ngram_generator", config=ngram_config)
    print(nlp.pipe_names, tokenizer_config)
    return nlp


if __name__ == "__main__":
    nlp = load_language_model(
        "french",
        {
            "convert_entities": True,
            "lowercase": True,
            "strip_punctuation": True,
            "strip_numbers": True,
            "stemmer": True,
            "ascii": True,
            "hash_tokens": False,
            "min_word_length": 1,
            "stopwords": None,
        },
        filter_config={"pos_to_keep": ["NOUN", "ADJ"], "ents_to_keep": ["PER", "LOC"]},
    )
    s = """Comme pour « l’incident » survenu sur l’aérodrome de Saky, Kiev n’a pas revendiqué d’attaque sur Djankoï, un conseiller présidentiel, Mykhaïlo Podoliak, se contentant de confirmer l’explosion. Un responsable ukrainien a cependant affirmé au New York Times, sous couvert d’anonymat, qu’une unité militaire d’élite ukrainienne opérant derrière les lignes ennemies était à l’origine de l’attaque. Les responsables ukrainiens ont aussi prévenu mardi que la Crimée ne serait pas épargnée par les ravages de la guerre."""
    doc = nlp(s)
    for token in doc:
        print(token._.surface_form, token.text)
