#!/usr/bin/env python3
"""Text Preprocessor"""

import json
import os
import random
import re
import string
import sys
import unicodedata
import sqlite3

import msgpack
import spacy
from multiprocess import Pool, cpu_count
from unidecode import unidecode

from Stemmer import Stemmer

from .modernize import modernize

PUNCTUATION_MAP = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P"))
TRIM_LAST_SLASH = re.compile(r"/\Z")
NUMBERS = re.compile(r"\d")
TAGS = re.compile(r"<[^>]+>")
WORD_CHARS = re.compile(r"\w+")

PHILO_TEXT_OBJECT_TYPE = {"doc": 1, "div1": 2, "div2": 3, "div3": 4, "para": 5, "sent": 6, "word": 7}


class Token(str):
    """Token Object class inheriting from string"""

    def __new__(cls, value, pos_="", ext=""):
        return str.__new__(cls, value)

    def __init__(self, text, pos_="", ext=""):
        self.text = text or ""
        self.pos_ = pos_ or ""
        self.ext = ext or {}

    def __hash__(self):
        return hash(self.text)

    def __eq__(self, other):
        if isinstance(other, Token):
            return self.text == other.text
        return self.text == other

    def __str__(self):
        return self.text

    def __call__(self):
        return self

    def __repr__(self):
        return self.text

    def __add__(self, other):
        return self.text + other


class Tokens:
    """Tokens Object class"""

    def __init__(self, tokens, metadata):
        self.tokens = tokens
        self.metadata = metadata

    def __iter__(self):
        for token in self.tokens:
            yield token

    def __getitem__(self, item):
        return self.tokens[item]

    def __len__(self):
        return len(self.tokens)


class Lemmatizer:
    """Lemmatizer wrapper"""

    def __init__(self, input_file=None):
        self.input = input_file
        self.loaded = False
        self.lemmatizer = {}

    def get(self, word, replacement):
        """Get lemma"""
        if self.loaded is False:
            self.__load_input()
        try:
            return self.lemmatizer[word]
        except KeyError:
            return replacement

    def __load_input(self):
        with open(self.input, "rb") as pack_input:
            self.lemmatizer = msgpack.load(pack_input, encoding="utf8")
        self.loaded = True


class PreProcessor:
    """ Text Preprocessing class"""

    def __init__(
        self,
        word_regex=r"\w+",
        sentence_regex=r"[.!?]+",
        language="french",
        stemmer=False,
        lemmatizer=None,
        modernize=False,
        ngrams=None,
        stopwords=None,
        strip_punctuation=True,
        strip_numbers=True,
        strip_tags=False,
        lowercase=True,
        min_word_length=2,
        ascii=True,
        with_pos=False,
        pos_to_keep=[],
        is_philo_db=False,
        text_object_type="doc",
        return_type="words",
    ):
        self.modernize = modernize
        self.language = language
        self.stemmer = stemmer
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
        self.return_type = return_type
        if self.pos_to_keep or self.with_pos is True or self.lemmatizer == "spacy":
            # spacy.prefer_gpu()
            try:
                self.nlp = spacy.load(
                    language[:2], disable=["parser", "ner", "textcat"]
                )  # assuming first two letters define language model
            except Exception as e:
                print(e)
                self.nlp = False
                print("Spacy does not support {} POS tagging".format(language))
        else:
            self.nlp = False
        self.token_regex = re.compile(r"{}|{}".format(word_regex, sentence_regex))
        self.word_tokenizer = re.compile(word_regex)
        self.sentence_tokenizer = re.compile(sentence_regex)

    def __get_stopwords(self, file_path):
        if file_path is None or os.path.isfile(file_path) is False:
            return []
        stopwords = set()
        with open(file_path) as stopword_file:
            for line in stopword_file:
                stopwords.add(line.strip())
        return stopwords

    def __get_lemmatizer(self, file_path):
        if file_path == "spacy":
            return "spacy"
        elif file_path is None or os.path.isfile(file_path) is False:
            return {}
        else:
            lemmas = {}
        with open(file_path) as input_file:
            for line in input_file:
                word, lemma = line.strip().split("\t")
                lemmas[word] = lemma
        output_file = f"/tmp/{''.join([random.choice(string.ascii_lowercase) for _ in range(20)])}.pack"
        with open(output_file, "wb") as output:
            output.write(msgpack.dumps(lemmas))
        return Lemmatizer(output_file)

    def __generate_ngrams(self, tokens):
        ngrams = []
        ngram = []
        for token in tokens:
            ngram.append(token)
            if len(ngram) == self.ngrams:
                ngram_text = " ".join(t.text for t in ngram)
                ext = ngram[0].ext
                ext["end_byte"] = ngram[-1].ext["end_byte"]
                ngrams.append(Token(ngram_text, None, ext))
                ngram = []
        return ngrams

    def __pos_tagger(self, tokens):
        filtered_tokens = []
        if self.pos_to_keep:
            filtered_tokens = [Token(t.text, t.pos_) for t in self.nlp(" ".join(tokens)) if t.pos_ in self.pos_to_keep]
        else:
            filtered_tokens = self.nlp(" ".join(tokens))
        return filtered_tokens

    def normalize(self, token, stemmer):  # This function can be used standalone
        """Normalize a single string token"""
        if self.lowercase:
            token = token.lower()
        if token in self.stopwords:
            return ""
        if self.strip_punctuation:
            token = token.translate(PUNCTUATION_MAP)
        if self.strip_numbers:
            if NUMBERS.search(token):
                return ""
        if stemmer is not None:
            token = stemmer.stemWord(token)
        if len(token) < self.min_word_length:
            return ""
        if self.ascii:
            token = unidecode(token)
        return token

    def __normalize_doc(self, doc):
        """Normalize single documents"""
        normalized_doc = []
        if self.stemmer is True:
            stemmer = Stemmer(self.language)
            stemmer.maxCacheSize = 50000
        else:
            stemmer = None
        for inner_token in doc:
            normalized_token = self.normalize(inner_token.text, stemmer)
            if normalized_token:
                normalized_doc.append(Token(normalized_token, inner_token.pos_, inner_token.ext))
        return normalized_doc

    def __format(self, doc, metadata):
        """Format output"""
        doc = self.__normalize_doc(doc)
        if self.return_type == "words":
            if self.ngrams is not None:
                doc = self.__generate_ngrams(doc)
            return Tokens(doc, metadata)
        elif self.return_type == "sentences":
            sentence = []
            list_of_sentences = []
            for word in doc:
                if self.sentence_tokenizer.search(word.text):
                    if self.ngrams is not None:
                        sentence = self.__generate_ngrams(sentence)
                    list_of_sentences.append(sentence)
                    sentence = []
                    continue
                sentence.append(word)
            if sentence:
                if self.ngrams is not None:
                    sentence = self.__generate_ngrams(sentence)
                list_of_sentences.append(Tokens(sentence, metadata))
            return list_of_sentences
        else:
            print("Error: only return_types possible are 'sentences' and 'words' (which can be ngrams)")
            exit()

    def process_texts(self, texts, progress=True):
        """Process all documents. Returns an iterator of documents"""
        count = 0
        if progress is True:
            print("\nProcessing texts...", end="", flush=True)
        if self.with_pos is True or self.pos_to_keep or self.lemmatizer == "spacy":
            for text in texts:
                if self.is_philo_db is True:
                    text_objects, metadata = self.process_philo_texts(text)
                    for text_object, object_metadata in zip(text_objects, metadata):
                        yield self.__format(text_object, object_metadata)
                else:
                    doc, metadata = self.process_text(text)
                    yield self.__format(doc, metadata)
                if progress is True:
                    count += 1
                    print("\rProcessing texts... {} done".format(count), end="", flush=True)
        else:
            with Pool(cpu_count()) as pool:
                for processed_doc in pool.imap_unordered(self.__local_process, texts):
                    if progress is True:
                        count += 1
                        print("\rProcessing texts... {} done".format(count), end="", flush=True)
                    for sub_doc in processed_doc:
                        yield sub_doc
        print()

    def __local_process(self, text):
        if self.is_philo_db is True:
            text_objects, metadata = self.process_philo_texts(text)
            return [
                self.__format(processed_text, obj_metadata)
                for processed_text, obj_metadata in zip(text_objects, metadata)
            ]
        doc, metadata = self.process_text(text)
        return [self.__format(doc, metadata)]

    def pos_tag_text(self, text):
        """POS tag document. Return tagged document"""
        # We bypass Spacy's tokenizer which is slow and call the POS tagger directly from the language model
        doc = self.nlp.tagger(spacy.tokens.Doc(self.nlp.vocab, [w.text for w in text]))
        if self.pos_to_keep:
            if self.lemmatizer and self.lemmatizer != "spacy":
                doc = [
                    Token(self.lemmatizer.get(t.text.lower(), t.text), t.pos_, old_t.ext)
                    for t, old_t in zip(doc, text)
                    if t.pos_ in self.pos_to_keep
                ]
            else:
                doc = [Token(t.lemma_, t.pos_, old_t.ext) for t, old_t in zip(doc, text) if t.pos_ in self.pos_to_keep]
        else:
            doc = [Token(t.lemma_, t.pos_, old_t.ext) for t, old_t in zip(doc, text)]
        return doc

    def remove_tags(self, text):
        """Strip XML tags"""
        end_header_index = text.rfind("</teiHeader>") + 12
        text = text[end_header_index:]
        text = TAGS.sub("", text)
        return text

    def tokenize_text(self, text):
        """Tokenize text"""
        with open(text) as input_text:
            doc = input_text.read()
        if self.strip_tags:
            doc = self.remove_tags(doc)
        for match in self.token_regex.finditer(doc):
            if self.modernize:
                yield Token(modernize(match[0], self.language))
            yield Token(match[0])

    def process_text(self, text):
        """Process one document. Return the transformed document"""
        tokens = self.tokenize_text(text)
        metadata = {"filename": text}
        if self.with_pos is True or self.pos_to_keep or self.lemmatizer == "spacy":
            return self.pos_tag_text(tokens), metadata
        elif self.lemmatizer and self.lemmatizer != "spacy":
            tokens = [Token(self.lemmatizer.get(word, word), "", word.ext) for word in tokens]
        else:
            tokens = list(tokens)  # We need to convert generator to list for pickling in multiprocessing
        return tokens, metadata

    def process_philo_texts(self, text):
        """Process files produced by PhiloLogic parser"""
        docs = []
        current_object_id = None
        current_text_object = []
        db_path = os.path.abspath(os.path.join(text, os.pardir, os.pardir, "toms.db"))
        if not os.path.exists(db_path):
            print(
                "Philologic files must be inside a standard PhiloLogic database directory to extract metadata\nExiting..."
            )
            exit()
        with sqlite3.connect(db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = db.cursor()
            metadata = []
            with open(text) as philo_db_text:
                for line in philo_db_text:
                    word_obj = json.loads(line.strip())
                    object_id = " ".join(word_obj["position"].split()[: PHILO_TEXT_OBJECT_TYPE[self.text_object_type]])
                    if current_object_id is None:
                        current_object_id = object_id
                    if object_id != current_object_id:
                        if current_text_object:
                            obj_metadata = recursive_search(cursor, object_id, self.text_object_type)
                            metadata.append(obj_metadata)
                            if self.with_pos is True or self.pos_to_keep or self.lemmatizer == "spacy":
                                current_text_object = self.pos_tag_text(current_text_object)
                                docs.append(current_text_object)
                            else:
                                if self.lemmatizer:
                                    current_text_object = [
                                        Token(self.lemmatizer.get(word, word), "", word.ext)
                                        for word in current_text_object
                                    ]
                                docs.append(current_text_object)
                            current_text_object = []
                        current_object_id = object_id
                    if self.modernize:
                        word_obj["token"] = modernize(word_obj["token"], self.language)
                    current_text_object.append(Token(word_obj["token"], "", word_obj))
            if current_text_object:
                obj_metadata = recursive_search(cursor, current_object_id, self.text_object_type)
                metadata.append(obj_metadata)
                if self.with_pos is True or self.pos_to_keep or self.lemmatizer == "spacy":
                    current_text_object = self.pos_tag_text(current_text_object)
                    docs.append(current_text_object)
                else:
                    if self.lemmatizer:
                        current_text_object = [
                            Token(self.lemmatizer.get(word, word), "", word.ext) for word in current_text_object
                        ]
                    docs.append(current_text_object)
        return docs, metadata


def recursive_search(cursor, object_id, object_type):
    """Recursive look-up of PhiloLogic objects"""
    # TODO: cache redundent lookups
    object_id = object_id.split()
    object_level = PHILO_TEXT_OBJECT_TYPE[object_type]
    obj_metadata = {}
    while object_id:
        current_id = f"{' '.join(object_id[:object_level])} {' '.join('0' for _ in range(7 - object_level))}"
        cursor.execute("SELECT * from toms where philo_id = ?", (current_id,))
        result = cursor.fetchone()
        if result is not None:
            for field in result.keys():
                if field not in obj_metadata:
                    if result[field] or object_level == 1:  # make sure we get all metadata stored at the last level
                        obj_metadata[field] = result[field] or ""
        object_id.pop()
        object_level -= 1
    return obj_metadata
