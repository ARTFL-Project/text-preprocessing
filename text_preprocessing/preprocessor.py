#!/usr/bin/env python3

import re
import sys
import unicodedata
from collections import namedtuple

import spacy
from unidecode import unidecode

from Stemmer import Stemmer

from .modernize import modernize

PUNCTUATION_MAP = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
TRIM_LAST_SLASH = re.compile(r'/\Z')
NUMBERS = re.compile(r'\d')
TAGS = re.compile(r"<[^>]+>")
WORD_CHARS = re.compile(r"\w+")

tokenObject = namedtuple("tokenObject", "text, pos_")
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
            self.nlp = spacy.load(language[:2], disable=["parser", "ner", "textcat"])  # assuming first two letters define language model
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
        if self.pos_to_keep:
            filtered_tokens = [tokenObject(t.text, t.pos_) for t in self.nlp(" ".join(tokens)) if t.pos_ in self.pos_to_keep]
        else:
            filtered_tokens = self.nlp(" ".join(tokens))
        return filtered_tokens

    def __normalize(self, token):
        if self.lowercase:
            token = token.lower()
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

    def __normalize_doc(self, doc, return_type, with_pos):
        normalized_doc = []
        for inner_token in doc:
            normalized_token = self.__normalize(inner_token.text)
            if normalized_token:
                normalized_doc.append(tokenObject(normalized_token, inner_token.pos_))
        return self.__format(normalized_doc, return_type, with_pos)

    def __format(self, doc, return_type, with_pos):
        if self.ngrams is not None:
            for i in range(len(doc)):
                doc[i] = self.__generate_ngrams(doc[i])
            return doc
        if return_type == "words":
            if with_pos is True:
                return [(w.text, w.pos_) for w in doc]
            else:
                return [w.text for w in doc]
        elif return_type == "sentences":
            sentence = []
            list_of_sentences = []
            for word in doc:
                if self.sentence_tokenizer.search(word):
                    list_of_sentences.append(sentence)
                    sentence = []
                sentence.append(word)
            if sentence:
                list_of_sentences.append(sentence)
            return list_of_sentences
        else:
            print("Error: only return_types possible are 'sentences' and 'words' (which can be ngrams)")
            exit()

    def process_texts(self, texts, return_type="words", with_pos=False, batch_size=100, threads=-1):
        """Process all documents. Returns an iterator of documents"""
        if with_pos is True or self.pos_to_keep:
            texts = (" ".join(self.tokenize_text(text)) for text in texts)
            for doc in self.nlp.pipe(texts, batch_size=batch_size, n_threads=threads):
                if self.pos_to_keep:
                    doc = [tokenObject(t.text, t.pos_) for t in doc if t.pos_ in self.pos_to_keep]
                yield self.__normalize_doc(doc, return_type, with_pos)
        else:
            texts = (self.tokenize_text(text) for text in texts)
            for doc in texts:
                doc = [tokenObject(t) for t in doc]
                yield self.__normalize_doc(doc, return_type, with_pos)

    def process_text(self, text, return_type="words", with_pos=False):
        """Process one document. Return the transformed document"""
        return self.process_texts([text], return_type=return_type, with_pos=with_pos)

    def remove_tags(self, text):
        """Strip XML tags"""
        end_header_index = text.rfind("</teiHeader>") + 12
        text = text[end_header_index:]
        text = TAGS.sub("", text)
        return text

    def tokenize_text(self, text):
        """Tokenize text"""
        if not isinstance(text, str):
            print("Error: The text you provided is not a string so it cannot be processed.")
            exit()
        if self.strip_tags:
            text = self.remove_tags(text)
        for match in self.token_regex.finditer(text):
            if self.modernize:
                yield modernize(match[0], self.language)
            yield match[0]
