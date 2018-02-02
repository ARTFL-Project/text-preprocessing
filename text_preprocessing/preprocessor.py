#!/usr/bin/env python3

import re
import unicodedata
import sys
from collections import namedtuple

import spacy
from .modernize import modernize
from Stemmer import Stemmer
from unidecode import unidecode

PUNCTUATION_MAP = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
TRIM_LAST_SLASH = re.compile(r'/\Z')
NUMBERS = re.compile(r'\d')
TAGS = re.compile(r"<[^>]+>")
WORD_CHARS = re.compile(r"\w+")

tokenObject = namedtuple("tokenObject", "text, pos")
tokenObject.__new__.__defaults__ = ("", "")


class PreProcessor:
    """ Text Preprocessing class"""

    def __init__(self, word_regex=r"\w+", sentence_regex=r"[.!?]+", language="french", stemmer=False, lemmatizer=None, modernize=False,
                 ngrams=None, stopwords=None, strip_punctuation=True, strip_numbers=True, strip_tags=False, lowercase=True, min_word_length=2,
                 ascii=True, pos_to_keep=[]):
        self.modernize = modernize
        self.language = language
        if stemmer is True:
            self.stemmer = Stemmer(self.language)
            self.stemmer.maxCacheSize = 50000
        else:
            self.stemmer = False
        self.ngrams = ngrams
        self.stopwords = self.__get_stopwords(stopwords)
        self.lemmatizer = self.__get_lemmatizer(lemmatizer)
        self.strip_punctuation = strip_punctuation
        self.strip_numbers = strip_numbers
        self.strip_tags = strip_tags
        self.lowercase = lowercase
        self.min_word_length = min_word_length
        self.ascii = ascii
        self.pos_to_keep = set(pos_to_keep)
        try:
            self.nlp = spacy.load(language[:2])  # assuming first two letters define language model
        except:
            self.nlp = False
            print("Spacy does not support {} POS tagging".format(language))
        self.token_regex = re.compile(r"{}|{}".format(word_regex, sentence_regex))
        self.word_tokenizer = re.compile(word_regex)
        self.sentence_tokenizer = re.compile(sentence_regex)


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

    def __generate_ngrams(self, tokens):
        ngrams = []
        ngram = []
        for token, _ in tokens:
            ngram.append(token)
            if len(ngram) == self.ngrams:
                ngrams.append("_".join(ngram))
                ngram = []
        return ngrams

    def __pos_tagger(self, tokens):
        filtered_tokens = []
        sentence = []
        for token in tokens:
            if self.sentence_tokenizer.search(token):
                if self.pos_to_keep:
                    filtered_tokens.extend([tokenObject(t.text, t.pos_) for t in self.nlp(" ".join(sentence)) if t.pos_ in self.pos_to_keep])
                else:
                    filtered_tokens.extend([tokenObject(t.text, t.pos_) for t in self.nlp(" ".join(sentence))])
                sentence = []
                continue
            if self.modernize:
                sentence.append(modernize(token, self.language))
            else:
                sentence.append(token)
        if sentence:
            if self.pos_to_keep:
                filtered_tokens.extend([tokenObject(t.text, t.pos_) for t in self.nlp(" ".join(sentence)) if t.pos_ in self.pos_to_keep])
            else:
                filtered_tokens.extend([tokenObject(t.text, t.pos_) for t in self.nlp(" ".join(sentence))])
        return filtered_tokens

    def __normalize(self, token):
        if self.lowercase is True:
            token = token.lower()
        if self.modernize:
            token = modernize(token, self.language)
        if token in self.stopwords:
            return ""
        if self.strip_punctuation:
            token = token.translate(PUNCTUATION_MAP)
        if self.strip_numbers:
            if NUMBERS.search(token):
                return ""
        if self.lemmatizer is not None:
            token = self.lemmatizer.get(token, token)
        if self.stemmer is not False:
            token = self.stemmer.stemWord(token)
        if len(token) < self.min_word_length:
            return ""
        if self.ascii:
            token = unidecode(token)
        return token

    def __normalize_sentence(self, sentence, with_pos):
        if self.pos_to_keep or with_pos:
            sentence = self.__pos_tagger(sentence)
        else:
            sentence = [tokenObject(t) for t in sentence]
        normalized_sentence = []
        for inner_token in sentence:
            normalized_token = self.__normalize(inner_token.text)
            if normalized_token:
                normalized_sentence.append(tokenObject(normalized_token, inner_token.pos))
        return normalized_sentence

    def process(self, text, return_type="words", with_pos=False):
        """Process text"""
        if not isinstance(text, str):
            print("Error: The text you provided is not a string so it cannot be processed.")
            exit()
        if self.strip_tags:
            end_header_index = text.rfind("</teiHeader>") + 12
            text = text[end_header_index:]
            text = TAGS.sub("", text)
        tokens = self.token_regex.findall(text)
        sentences = []
        sentence = []
        for token in tokens:
            if self.sentence_tokenizer.search(token):
                normalized_sentence = self.__normalize_sentence(sentence, with_pos)
                if normalized_sentence:
                    sentences.append(normalized_sentence)
                sentence = []
            else:
                sentence.append(token)
        if sentence:
            normalized_sentence = self.__normalize_sentence(sentence, with_pos)
            if normalized_sentence:
                sentences.append(normalized_sentence)
        if self.ngrams is not None:
            for i in range(len(sentences)):
                sentences[i] = self.__generate_ngrams(sentences[i])
        elif with_pos is False:
            new_sentences = []
            for sentence in sentences:
                new_sentences.append([w.text for w in sentence])
            sentences = new_sentences
        if return_type == "words":
            return [w for sentence in sentences for w in sentence]
        elif return_type == "sentences":
            return sentences
        else:
            print("Error: only token_types possible are 'sentences' and 'words' (which can be ngrams)")
            exit()
