#!/usr/bin/env python3

import re
import unicodedata
import sys

import spacy
from .modernize import modernize

PUNCTUATION_MAP = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
TRIM_LAST_SLASH = re.compile(r'/\Z')
NUMBERS = re.compile(r'\d')
TAGS = re.compile(r"<[^>]+>")
WORD_CHARS = re.compile(r"\w+")


class PreProcessor:

    def __init__(self, token_regex=r"\w+", language="french", stemmer=True, lemmatizer=None, modernize=True, tokenize=True, ngrams=None,
                stopwords=None, strip_punctuation=True, strip_numbers=True, strip_tags=False, pos_to_keep=[]):
        self.modernize = modernize
        self.language = language
        if stemmer is True:
            self.stemmer = Stemmer(self.language)
            self.stemmer.maxCacheSize = 50000
        else:
            self.stemmer = False
        self.ngrams = None
        self.tokenize = tokenize
        self.stopwords = self.__get_stopwords(stopwords)
        self.lemmatizer = self.__get_lemmatizer(lemmatizer)
        self.strip_punctuation = strip_punctuation
        self.strip_numbers = strip_numbers
        self.token_regex = re.compile(token_regex)
        self.strip_tags = strip_tags
        self.pos_to_keep = set(pos_to_keep)
        if self.pos_to_keep:
            try:
                self.pos_tagger = spacy.load(language[:2])  # assuming first two letters define language model
            except:
                print("Spacy does not support {} POS tagging".format(language))
                self.pos_to_keep = []

    def __get_stopwords(self, file_path):
        if file_path is None:
            return []
        stopwords = set([])
        with open(file_path) as stopword_file:
            for line in stopword_file:
                stopwords.add(line.strip())
        return stopwords

    def __get_lemmatizer(self, file_path):
        if file_path is None:
            return None
        else:
            lemmas = {}
        with open(file_path) as input_file:
            for line in input_file:
                word, lemma = line.strip().split("\t")
                lemmas[word] = lemma
        return lemmas

    def process(self, text):
        final_tokens = []
        if self.strip_tags:
            end_header_index = text.rfind("</teiHeader>") + 12
            text = text[end_header_index:]
            text = TAGS.sub("", text)
        if self.tokenize is False and isinstance(text, str):
            self.tokenize = True
        if self.tokenize is True:
            tokens = self.token_regex.findall(text)
        if self.pos_to_keep:
            if self.modernize:
                tokens_w_pos = self.pos_tagger(" ".join([modernize(t, self.language) for t in tokens]))
            else:
                tokens_w_pos = self.pos_tagger(" ".join(tokens))
            tokens = [token.text for token in tokens_w_pos if token.pos_ in self.pos_to_keep]
        for token in tokens:
            if self.modernize and not self.pos_to_keep:
                token = modernize(token, self.language)
            if token in self.stopwords:
                continue
            if self.strip_punctuation:
                token = token.translate(PUNCTUATION_MAP)
            if self.strip_numbers:
                if NUMBERS.search(token):
                    continue
            if self.lemmatizer is not None:
                token = self.lemmatizer.get(token, token)
            if self.stemmer is not False:
                token = self.stemmer.stemWord(token)
            final_tokens.append(token)
        return final_tokens

def main():
    pass

if __name__ == "__main__":
    main()