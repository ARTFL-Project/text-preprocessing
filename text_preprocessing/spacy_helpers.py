import re
from typing import Any, Dict, List

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token


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


TRIM_LAST_SLASH = re.compile(r"/\Z")
TAGS = re.compile(r"<[^>]+>")
WORD_CHARS = re.compile(r"\w+")

Token.set_extension("surface_form", default="")
Token.set_extension("ext", default={})
Doc.set_extension("metadata", default={})


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
        model: List[str] = models[language]
    except KeyError:
        return []
    return model


class PassThroughTokenizer:
    """Used to bypass Spacy tokenizer"""

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, tokens):
        doc = Doc(self.vocab, words=tokens.split())
        return doc


def load_language_model(
    language,
    filter_config: Dict[str, Any],
) -> Language:
    """Load language model based on name"""
    nlp = None
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
            if filter_config["pos_to_keep"] is not None or filter_config["ents_to_keep"] is not None:
                nlp = spacy.load(model, exclude=["parser", "ner", "textcat"])
            else:
                nlp = spacy.load(model)
            nlp.tokenizer = PassThroughTokenizer(nlp.vocab)
        except OSError:
            pass
        if nlp is not None:
            break
    if nlp is None:
        print(f"No Spacy model installed for the {language} language. Stopping...")
        exit(-1)
    return nlp


if __name__ == "__main__":
    nlp = load_language_model(
        "french",
        {"pos_to_keep": ["NOUN", "ADJ"], "ents_to_keep": ["PER", "LOC"]},
    )
    import sys

    doc = nlp(sys.argv[1])
    for token in doc:
        print(token._.surface_form, token.text)
