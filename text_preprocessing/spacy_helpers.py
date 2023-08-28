"""Helper functions for Spacy"""

import os
import re
import sys
import unicodedata
from html import unescape as unescape_html
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import unescape as unescape_xml

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token
from Stemmer import Stemmer
from unidecode import unidecode

from .modernizer import Modernizer

# Updated as of 8/23/2022
SPACY_LANGUAGE_MODEL_MAP: Dict[str, List[str]] = {
    "catalan": ["ca_core_news_sm", "ca_core_news_md", "ca_core_news_lg", "ca_core_news_trf"],
    "chinese": ["zh_core_web_sm", "zh_core_web_md", "zh_core_web_lg", "zh_core_web_trf"],
    "croation": ["hr_core_news_sm", "hr_core_news_md", "hr_core_news_lg"],
    "danish": ["da_core_news_sm", "da_core_news_md", "da_core_news_lg"],
    "dutch": ["nl_core_news_sm", "nl_core_news_md", "nl_core_news_lg"],
    "english": ["en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf"],
    "finnish": ["fi_core_news_sm", "fi_core_news_md", "fi_core_news_lg"],
    "german": ["de_core_news_sm", "de_core_news_md", "de_core_news_lg", "de_dep_news_trf"],
    "greek": ["el_core_news_sm", "el_core_news_md", "el_core_news_lg"],
    "french": ["fr_core_news_sm", "fr_core_news_md", "fr_core_news_lg", "fr_dep_news_trf"],
    "italian": ["it_core_news_sm", "it_core_news_md", "it_core_news_lg"],
    "japanese": ["ja_core_news_sm", "ja_core_news_md", "ja_core_news_lg"],
    "korean": ["ko_core_news_sm", "ko_core_news_md", "ko_core_news_lg"],
    "lithuanian": ["lt_core_news_sm", "lt_core_news_md", "lt_core_news_lg"],
    "macedonian": ["mk_core_news_sm", "mk_core_news_md", "mk_core_news_lg"],
    "norwegian": ["nb_core_news_sm", "nb_core_news_md", "nb_core_news_lg"],
    "polish": ["pl_core_news_sm", "pl_core_news_md", "pl_core_news_lg"],
    "portuguese": ["pt_core_news_sm", "pt_core_news_md", "pt_core_news_lg"],
    "romanian": ["ro_core_news_sm", "ro_core_news_md", "ro_core_news_lg"],
    "russian": ["ru_core_news_sm", "ru_core_news_md", "ru_core_news_lg"],
    "spanish": ["es_core_news_sm", "es_core_news_md", "es_core_news_lg", "es_dep_news_trf"],
    "swedish": ["sv_core_news_sm", "sv_core_news_md", "sv_core_news_lg"],
    "ukrainian": ["uk_core_news_sm", "uk_core_news_md", "uk_core_news_lg"],
    "multi-language": ["xx_ent_wiki_sm", "xx_sent_ud_sm"],
}


PUNCTUATION_MAP = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P"))
PUNCTUATION_CLASS = set([chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")])
NUMBERS = re.compile(r"\d")


def check_for_updates(language) -> List[str]:
    """Check for spacy language model updates"""
    import requests

    response = requests.get("https://raw.githubusercontent.com/explosion/spaCy/master/website/meta/languages.json")
    if response.status_code == 404:
        print("Unable to fetch language information from Spacy GitHub")
        return []
    try:
        languages = response.json()
        models = {lang["name"].lower(): lang["models"] for lang in languages["languages"] if "models" in lang}
        model: List[str] = models[language][::-1]
        print(model)
    except KeyError:
        return []
    return model


@Language.factory(
    "postprocessor",
    default_config={
        "stemmer": False,
        "lemmatizer": False,
        "stopwords": False,
        "strip_punctuation": True,
        "strip_numbers": True,
        "lowercase": True,
        "min_word_length": 2,
        "ascii": False,
        "convert_entities": False,
        "pos_to_keep": False,
        "ents_to_keep": False,
    },
)
def post_process_component(
    nlp: Language,
    name: str,
    language: str,
    stemmer: bool,
    lemmatizer: str,
    stopwords: str | bool,
    strip_punctuation: bool,
    strip_numbers: bool,
    lowercase: bool,
    min_word_length: int,
    ascii: bool,
    convert_entities: bool,
    pos_to_keep: bool | List[str],
    ents_to_keep: bool | List[str],
):
    """Create a preprocessor pipeline component"""
    return PreProcessingPipe(
        nlp,
        language,
        stemmer,
        lemmatizer,
        stopwords,
        strip_punctuation,
        strip_numbers,
        lowercase,
        min_word_length,
        ascii,
        convert_entities,
        pos_to_keep,
        ents_to_keep,
    )


class PreProcessingPipe:
    """Preprocessing pipeline component"""

    def __init__(
        self,
        nlp,
        language,
        stemmer,
        lemmatizer,
        stopwords,
        strip_punctuation,
        strip_numbers,
        lowercase,
        min_word_length,
        ascii,
        convert_entities,
        pos_to_keep,
        ents_to_keep,
    ):
        self.nlp = nlp
        self.stopwords = self.__get_stopwords(stopwords)
        self.pos_to_keep = pos_to_keep
        self.ents_to_keep = ents_to_keep

        self.convert_entities = convert_entities
        self.ascii = ascii
        self.lowercase = lowercase
        self.strip_punctuation = strip_punctuation
        self.strip_numbers = strip_numbers
        self.min_word_length = min_word_length

        if stemmer is True:
            self.stemmer = Stemmer(language)
            self.stemmer.maxCacheSize = 50000  # type: ignore
        else:
            self.stemmer = False

        if lemmatizer != "spacy":
            self.lemmatizer = self.__get_lemmatizer(lemmatizer)
            self.lemmatizer_path = lemmatizer
            self.spacy_lemmatizer = False
        else:
            self.spacy_lemmatizer = True

    def __get_stopwords(self, file_path: str | bool) -> set[str]:
        if not file_path or file_path == "False":
            return set()
        elif os.path.isfile(file_path) is False:
            print("Stopwords file", file_path, "not found. Exiting...")
            exit()
        stopwords = set()
        with open(file_path, encoding="utf-8") as stopword_file:
            for line in stopword_file:
                stopwords.add(line.strip())
        return stopwords

    def __get_lemmatizer(self, file_path: Optional[str]) -> dict[str, str]:
        if file_path is None or os.path.isfile(file_path) is False:
            return {}
        lemmas: dict = {}
        with open(file_path, encoding="utf-8") as input_file:
            for line in input_file:
                word, lemma = line.strip().split("\t")
                lemmas[word] = lemma
        return lemmas

    def __call__(self, doc: Doc) -> Doc:
        """Process a doc"""
        words = []
        pos = []
        ents = []
        for token in doc:
            if self.__filter_token(token) is True:
                normalized_text = "#DEL#"
            else:
                normalized_text = self.__normalize_token(token)
            if not normalized_text:
                normalized_text = "#DEL#"
            words.append(normalized_text)
            pos.append(token.pos_)
            if self.ents_to_keep and token.ent_iob_ != "O":
                ents.append(f"{token.ent_iob_}-{token.ent_type_}")
            else:
                ents.append("")
        new_doc = Doc(self.nlp.vocab, words=words, pos=pos, ents=ents)
        new_doc._.metadata = doc._.metadata
        assert len(new_doc) == len(doc)
        for index, token in enumerate(doc):
            new_doc[index]._.ext = token._.ext
        return new_doc

    def __filter_token(self, token: Token) -> bool:
        """Filter tokens based on pos and ents"""
        if token.text in self.stopwords:
            return True
        if self.ents_to_keep and token.ent_type_ != "":
            if token.ent_type_ not in self.ents_to_keep:
                return True
            return False
        if self.pos_to_keep and token.pos_ not in self.pos_to_keep:
            return True
        return False

    def __normalize_token(self, orig_token: Token) -> str:
        """Normalize a token"""
        token = orig_token.text.strip()
        if self.convert_entities is True:
            token = unescape_html(token)
            token = unescape_xml(token)
        if self.spacy_lemmatizer is True:
            token = orig_token.lemma_
        elif self.lemmatizer:
            token = self.lemmatizer.get(token, token)
        if self.lowercase is True:
            token = token.lower()
        if token in self.stopwords:
            return "#DEL#"
        if self.strip_punctuation is True:
            token = token.translate(PUNCTUATION_MAP)
        elif token in PUNCTUATION_CLASS:
            return token
        if self.strip_numbers is True and NUMBERS.search(token):
            return "#DEL#"
        if self.stemmer is not False:
            token = self.stemmer.stemWord(token)  # type: ignore
        if len(token) < self.min_word_length:
            return "#DEL#"
        if self.ascii is True:
            token = unidecode(token)
        return token


def load_language_model(language, normalize_options: dict[str, Any], text_object_type: str) -> Language:
    """Load language model based on name"""
    nlp = None
    language = language.lower()
    try:
        possible_models = SPACY_LANGUAGE_MODEL_MAP[language][::-1]
    except KeyError:
        try:
            possible_models = check_for_updates(language)
        except KeyError:
            print(f"Spacy does not support the {language} language.")
            exit(-1)
    if any(
        (
            normalize_options["lemmatizer"] == "spacy",
            normalize_options["pos_to_keep"],
            normalize_options["ents_to_keep"],
        )
    ):
        diabled_pipelines = ["tokenizer", "textcat"]
        if not normalize_options["pos_to_keep"]:
            diabled_pipelines.append("tagger")
        if not normalize_options["ents_to_keep"]:
            diabled_pipelines.append("ner")
        model_loaded = ""
        # spacy.prefer_gpu()
        for model in possible_models:
            try:
                nlp = spacy.load(model, exclude=diabled_pipelines)
                print("Using Spacy model", model)
            except OSError:
                pass
            if nlp is not None:
                model_loaded = model
                break
        if nlp is None:
            print(f"No Spacy model installed for the {language} language. Stopping...")
            exit(-1)
        nlp.add_pipe("postprocessor", config=normalize_options, last=True)
        if normalize_options["ents_to_keep"] and "ner" not in nlp.pipe_names:
            print(f"There is no NER pipeline for model {model_loaded}. Exiting...")
            exit(-1)
        # if text_object_type == "doc":
        #     nlp.batch_size = 16
        return nlp
    nlp = spacy.blank("en")
    nlp.add_pipe("postprocessor", config=normalize_options, last=True)
    return nlp
