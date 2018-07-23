#!/usr/bin/env python3
"""Text Preprocessor"""

import json
import re
import sys
import unicodedata
from collections import namedtuple

import spacy
from multiprocess import Pool, cpu_count
from spacy.pipeline import Tagger
from unidecode import unidecode

from Stemmer import Stemmer

from .modernize import modernize

PUNCTUATION_MAP = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
TRIM_LAST_SLASH = re.compile(r'/\Z')
NUMBERS = re.compile(r'\d')
TAGS = re.compile(r"<[^>]+>")
WORD_CHARS = re.compile(r"\w+")

PHILO_TEXT_OBJECT_TYPE = {'doc': 1, 'div1': 2, 'div2': 3, 'div3': 4, 'para': 5, 'sent': 6, 'word': 7}

tokenObject = namedtuple("tokenObject", "text, pos_, ext")
tokenObject.__new__.__defaults__ = ("", "", {})


class PreProcessor:
    """ Text Preprocessing class"""

    def __init__(self, word_regex=r"\w+", sentence_regex=r"[.!?]+", language="french", stemmer=False, lemmatizer=None, modernize=False,
                 ngrams=None, stopwords=None, strip_punctuation=True, strip_numbers=True, strip_tags=False, lowercase=True, min_word_length=2,
                 ascii=True, with_pos=False, pos_to_keep=[], is_philo_db=False, text_object_type="doc"):
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
        self.with_pos = with_pos
        self.is_philo_db = is_philo_db
        self.text_object_type = text_object_type
        if self.pos_to_keep or self.with_pos is True:
            try:
                self.nlp = spacy.load(language[:2], disable=["parser", "ner", "textcat"])  # assuming first two letters define language model
            except:
                self.nlp = False
                print("Spacy does not support {} POS tagging".format(language))
        else:
            self.nlp = False
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
        if file_path is None or file_path == "":
            return {}
        elif file_path == "spacy":
            return "spacy"
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
        if self.pos_to_keep:
            filtered_tokens = [tokenObject(t.text, t.pos_) for t in self.nlp(" ".join(tokens)) if t.pos_ in self.pos_to_keep]
        else:
            filtered_tokens = self.nlp(" ".join(tokens))
        return filtered_tokens

    def normalize(self, token):  # This function can be used standalone
        if self.lowercase:
            token = token.lower()
        if token in self.stopwords:
            return ""
        if self.strip_punctuation:
            token = token.translate(PUNCTUATION_MAP)
        if self.strip_numbers:
            if NUMBERS.search(token):
                return ""
        if self.stemmer is not False:
            token = self.stemmer.stemWord(token)
        if len(token) < self.min_word_length:
            return ""
        if self.ascii:
            token = unidecode(token)
        return token

    def __normalize_doc(self, doc, return_type):
        normalized_doc = []
        for inner_token in doc:
            normalized_token = self.normalize(inner_token.text)
            if normalized_token:
                normalized_doc.append(tokenObject(normalized_token, inner_token.pos_, inner_token.ext))
        return self.__format(normalized_doc, return_type)

    def __format(self, doc, return_type):
        if self.ngrams is not None:
            for i in range(len(doc)):
                doc[i] = self.__generate_ngrams(doc[i])
            return doc
        if return_type == "words":
            return [w for w in doc]
        elif return_type == "sentences":
            sentence = []
            list_of_sentences = []
            for word in doc:
                if self.sentence_tokenizer.search(word.text):
                    list_of_sentences.append(sentence)
                    sentence = []
                sentence.append(word)
            if sentence:
                list_of_sentences.append(sentence)
            return list_of_sentences
        else:
            print("Error: only return_types possible are 'sentences' and 'words' (which can be ngrams)")
            exit()

    def process_texts(self, texts, return_type="words", progress=True):
        """Process all documents. Returns an iterator of documents"""
        count = 0
        if progress is True:
            print("\nProcessing texts...", end="", flush=True)
        if self.with_pos is True or self.pos_to_keep:
            if self.is_philo_db is True:
                texts = (t for text in texts for t in self.process_philo_texts(text, return_type))
            else:
                texts = (self.tokenize_text(text) for text in texts)
            for text in texts:
                doc = self.process_text(text)
                if progress is True:
                    count += 1
                    print("\rProcessing texts... {} done".format(count), end="", flush=True)
                yield self.__normalize_doc(doc, return_type)
        else:
            pool = Pool(cpu_count())
            def __local_process(text):
                if self.is_philo_db is True:
                    return self.process_philo_texts(text, return_type)
                else:
                    doc = []
                    for token in text:
                        if self.modernize:
                            token = modernize(token, self.language)
                        if self.lemmatizer:
                            token = self.lemmatizer.get(token.lower(), token)
                        doc.append(tokenObject(token))
                    return [doc]
            for processed_doc in pool.imap_unordered(__local_process, texts):
                for processed_text_object in processed_doc:
                    yield processed_text_object
                if progress is True:
                    count += 1
                    print("\rProcessing texts... {} done".format(count), end="", flush=True)

    def process_text(self, text):
        """Process one document. Return the transformed document"""
        # We bypass Spacy's tokenizer which is slow and call the POS tagger directly from the language model
        doc = self.nlp.tagger(spacy.tokens.Doc(self.nlp.vocab, [w.text for w in text]))
        if self.pos_to_keep:
            if self.lemmatizer and self.lemmatizer != "spacy":
                doc = [tokenObject(self.lemmatizer.get(t.text.lower(), t.text), t.pos_, old_t.ext) for t, old_t in zip(doc, text) if t.pos_ in self.pos_to_keep]
            else:
                doc = [tokenObject(t.lemma_, t.pos_, old_t.ext) for t, old_t in zip(doc, text) if t.pos_ in self.pos_to_keep]
        else:
            doc = [tokenObject(t.lemma_, t.pos_, old_t.ext) for t, old_t in zip(doc, text)]
        return doc

    def remove_tags(self, text):
        """Strip XML tags"""
        end_header_index = text.rfind("</teiHeader>") + 12
        text = text[end_header_index:]
        text = TAGS.sub("", text)
        return text

    def tokenize_text(self, text):
        """Tokenize text"""
        if self.strip_tags:
            text = self.remove_tags(text)
        for match in self.token_regex.finditer(text):
            if self.modernize:
                yield tokenObject(modernize(match[0], self.language))
            yield tokenObject(match[0])

    def process_philo_texts(self, text, return_type):
        docs = []
        current_object_id = None
        current_text_object = []
        with open(text) as philo_db_text:
            for line in philo_db_text:
                word_obj = json.loads(line.strip())
                object_id = "_".join(word_obj["position"].split()[:PHILO_TEXT_OBJECT_TYPE[self.text_object_type]])
                if current_object_id is None:
                    current_object_id = object_id
                if object_id != current_object_id:
                    if current_text_object:
                        if self.with_pos is True or self.pos_to_keep:
                            docs.append(current_text_object)
                        else:
                            docs.append(self.__normalize_doc(current_text_object, return_type))
                        current_text_object = []
                    current_object_id = object_id
                if self.modernize:
                    current_text_object.append(tokenObject(modernize(word_obj["token"], self.language), '', word_obj))
                else:
                    current_text_object.append(tokenObject(word_obj["token"], '', word_obj))
        if current_text_object:
            if self.with_pos is True or self.pos_to_keep:
                docs.append(current_text_object)
            else:
                docs.append(self.__normalize_doc(current_text_object, return_type))
        return docs