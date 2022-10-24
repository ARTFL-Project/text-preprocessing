#!/usr/bin/env python3
"""Text Preprocessor"""

import json
import os
import re
import sqlite3
import sys
import unicodedata
from collections import defaultdict, deque
from html import unescape as unescape_html
from itertools import combinations
from typing import (
    Any,
    Callable,
    DefaultDict,
    Deque,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    overload,
)
from xml.sax.saxutils import unescape as unescape_xml

import lz4.frame
import mmh3
import orjson
from multiprocess.pool import Pool
from Stemmer import Stemmer
from unidecode import unidecode

from .modernizer import Modernizer
from .spacy_helpers import load_language_model

PUNCTUATION_MAP = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P"))
PUNCTUATION_CLASS = set([chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")])
TRIM_LAST_SLASH = re.compile(r"/\Z")
NUMBERS = re.compile(r"\d")
TAGS = re.compile(r"<[^>]+>")
WORD_CHARS = re.compile(r"\w+")

PHILO_TEXT_OBJECT_TYPE: Dict[str, int] = {
    "doc": 1,
    "div1": 2,
    "div2": 3,
    "div3": 4,
    "para": 5,
    "sent": 6,
    "word": 7,
}

PHILO_OBJECT_LEVEL: Dict[int, str] = {1: "doc", 2: "div1", 3: "div2", 4: "div3", 5: "para", 6: "sent", 7: "word"}


class Token(str):
    """Token Object class inheriting from string

    Args:
        text: a string value
        surface_form: surface form to be changed. Defaults to text if none given
        pos_: a string value describing part-of-speech
        ext: a dictionary containing additional metadata

    Attributes:
        text: a string value
        surface_form: surface form to be changed. Defaults to text if none given
        pos_: a string value describing part-of-speech
        ext: a dictionary containing additional metadata

    """

    ext: Dict[str, Any]

    def __new__(cls, value, surface_form="", pos_="", ent="", ext={}):
        return str.__new__(cls, value)

    def __init__(
        self,
        text: str,
        surface_form: str = "",
        pos_: str = "",
        ent: str = "",
        ext: Optional[Dict[str, Any]] = None,
    ):
        self.text = text or ""
        self.surface_form = surface_form or text
        self.ext = ext or {}
        self.ext["pos"] = pos_
        self.pos_ = pos_
        self.ent = ent

    def __hash__(self):
        return hash(self.text)

    def __eq__(self, other) -> bool:
        if isinstance(other, Token):
            return self.text == other.text
        return self.text == other

    def __str__(self) -> str:
        return self.text

    def __call__(self):
        return self

    def __repr__(self) -> str:
        return f"text={repr(self.text)}, surface_form={repr(self.surface_form)}, pos={self.pos_}, ext={repr(self.ext)}"

    def __add__(self, other) -> str:
        return self.text + other


class Tokens:
    """Tokens object contains a list of tokens as well as metadata

    Args:
        tokens: a list of Token objects
        metadata: a dict containing metadata

    Attributes:
        tokens: a list of Token ojects
        metadata: a dict containing metadata
        length: length of Tokens.tokens

    """

    def __init__(self, tokens: Iterable[Token], metadata: Dict[str, Any]):
        self.tokens: Deque[Token] = deque(tokens)
        self.metadata: Dict[str, Any] = metadata
        self.length: int = len(self.tokens)
        self.iter_index = 0

    def __iter__(self) -> Iterator[Token]:
        for token in self.tokens:
            yield token

    def __next__(self):
        self.iter_index += 1
        if self.iter_index < self.length:
            return self.tokens[self.iter_index]
        else:
            raise IndexError

    @overload
    def __getitem__(self, index: int) -> Token:
        ...

    @overload
    def __getitem__(self, index: slice) -> Iterable[Token]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Token, Iterable[Token]]:
        if isinstance(index, int):
            return self.tokens[index]
        elif isinstance(index, slice):
            return Tokens(list(self.tokens)[index], self.metadata)
        else:
            print(f"{repr(index)} of type {type(index)} is not an index or slice")
            raise TypeError

    def __len__(self) -> int:
        return self.length

    def __bool__(self) -> bool:
        if self.length == 0:
            return False
        return True

    def __repr__(self):
        return repr([repr(t) for t in self.tokens])

    def __str__(self):
        return repr([str(t) for t in self.tokens])

    def split_tokens(self, n: int) -> Iterator["Tokens"]:
        """Divide Tokens in to smaller Tokens of n length

        Args:
            n: split Tokens obj into a list of Tokens of length n

        Returns:
            A Iterator of Tokens

        """
        max_index: int = self.length - 1
        for i in range(0, len(self), n):
            end: int = i + n
            if end > max_index:
                metadata: Dict[str, Any] = {
                    **self.metadata,
                    "start_byte": self[i].ext["start_byte"],
                    "end_byte": self[max_index].ext["end_byte"],
                }
                yield Tokens(self[i:max_index], metadata)
            else:
                metadata = {
                    **self.metadata,
                    "start_byte": self[i].ext["start_byte"],
                    "end_byte": self[end - 1].ext["end_byte"],
                }
                yield Tokens(self[i:end], metadata)

    def extend(self, tokens) -> None:
        """Extend size of Tokens"""
        self.tokens.extend(tokens)
        if not self.metadata:
            self.metadata = tokens.metadata
        self.metadata["end_byte"] = tokens.metadata["end_byte"]

    def pop(self) -> Optional[Token]:
        """Remove last token from self.tokens"""
        if self.tokens:
            token = self.tokens.pop()
            try:
                self.metadata["end_byte"] = self.tokens[-1].ext["end_byte"]
                self.length -= 1
                return token
            except IndexError:
                self.length = 0
            return token
        return None

    def popleft(self) -> Optional[Token]:
        """Remove first token from self.tokens"""
        if self.tokens:
            token = self.tokens.popleft()
            try:
                self.metadata["start_byte"] = self.tokens[0].ext["start_byte"]
                self.length -= 1
            except IndexError:
                self.length = 0
            return token
        return None

    def append(self, token: Token):
        """Append Token"""
        if not self.tokens:
            self.metadata["start_byte"] = token.ext["start_byte"]
        self.tokens.append(token)
        self.metadata["end_byte"] = token.ext["end_byte"]
        self.length += 1

    def appendleft(self, token: Token):
        """Append Token to the left of tokens"""
        if not self.tokens:
            self.metadata["end_byte"] = token.ext["end_byte"]
        self.tokens.appendleft(token)
        self.metadata["start_byte"] = token.ext["start_byte"]
        self.length += 1

    def purge(self):
        """Remove empty tokens"""
        self.tokens = deque(token for token in self.tokens if token)
        self.length = len(self.tokens)
        if self.length:
            self.metadata["start_byte"] = self.tokens[0].ext["start_byte"]
            self.metadata["end_byte"] = self.tokens[-1].ext["end_byte"]
        else:
            self.metadata["start_byte"] = 0
            self.metadata["end_byte"] = 0

    def save(self, path):
        """Save Tokens to disk"""
        tokens_to_serialize = {"tokens": [], "metadata": self.metadata}
        for token in self:
            tokens_to_serialize["tokens"].append((token.text, token.surface_form, token.pos_, token.ext))
        with open(path, "w") as output:
            json.dump(tokens_to_serialize, output)

    def load(self, path):
        """Load tokens from disk"""
        with open(path, "r") as input_file:
            tokens = json.load(input_file)
        self.metadata = tokens["metadata"]
        self.tokens = deque(Token(t[0], t[1], t[2], t[3]) for t in tokens["tokens"])


def chunks(l, n):
    """Yield n number of sequential chunks from l."""
    l = list(l)
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        result = l[si : si + (d + 1 if i < r else d)]
        if result:
            yield result


class PreProcessor:
    """Text Preprocessing class"""

    nlp: Callable = lambda x: x  # workaround for mypy
    word_regex: str = r"\w+"
    sentence_regex: str = r"[.!?]+"
    language: str = "french"
    lemmatizer: Dict[str, str] = {}
    lemmatizer_path: str = ""
    modernize: Callable = lambda x: x  # workaround for mypy
    ngrams: int = 0
    ngram_gap: int = 0
    stopwords: Set[str] = set()
    min_word_length: int = 2
    pos_to_keep: List[str] = []
    text_object_type: str = "doc"
    return_type: str = "words"
    workers: Optional[int] = None
    post_func: Optional[Callable] = None
    token_regex: re.Pattern = re.compile(r"")
    word_tokenizer = re.compile(word_regex)
    sentence_tokenizer = re.compile(sentence_regex)
    options: Dict[str, bool] = {}

    @classmethod
    def __init__(
        cls,
        word_regex: str = r"\w+",
        sentence_regex: str = r"[.!?]+",
        language: str = "french",
        stemmer: bool = False,
        lemmatizer: Optional[str] = None,
        modernize: bool = False,
        ngrams: Optional[int] = None,
        ngram_gap: int = 0,
        ngram_word_order: bool = True,
        stopwords: Optional[str] = None,
        strip_punctuation: bool = True,
        strip_numbers: bool = True,
        strip_tags: bool = False,
        lowercase: bool = True,
        min_word_length: int = 2,
        ascii: bool = False,
        convert_entities: bool = False,
        with_pos: bool = False,
        pos_to_keep: Optional[List[str]] = None,
        ents_to_keep: Optional[List[str]] = None,
        is_philo_db: bool = False,
        text_object_type: str = "doc",
        return_type: str = "words",
        hash_tokens: bool = False,
        workers: Optional[int] = None,
        post_processing_function: Optional[Callable] = None,
        **_,  # this is meant to make the constructor accept invalid keywords
    ):
        cls.language = language
        cls.options = {
            "modernize": modernize,
            "stemmer": stemmer,
            "ngram_word_order": ngram_word_order,
            "strip_punctuation": strip_punctuation,
            "strip_numbers": strip_numbers,
            "strip_tags": strip_tags,
            "lowercase": lowercase,
            "ascii": ascii,
            "convert_entities": convert_entities,
            "with_pos": with_pos,
            "is_philo_db": is_philo_db,
            "hash_tokens": hash_tokens,
            "spacy_lemmatizer": False,
        }
        if modernize is True:
            cls.modernize = Modernizer(language)
        if stemmer is True:
            cls.stemmer = Stemmer(cls.language)
            cls.stemmer.maxCacheSize = 50000  # type: ignore
        else:
            cls.stemmer = False
        cls.filter_config = {
            "pos_to_keep": pos_to_keep or None,
            "ents_to_keep": ents_to_keep or None,
        }
        if lemmatizer != "spacy":
            cls.lemmatizer = cls.__get_lemmatizer(lemmatizer)
        else:
            cls.options["spacy_lemmatizer"] = True
        cls.min_word_length = min_word_length
        if pos_to_keep is not None:
            cls.pos_to_keep = pos_to_keep
        if cls.options["with_pos"] is True or cls.pos_to_keep or cls.options["spacy_lemmatizer"] is True:
            cls.options["nlp"] = True
            cls.nlp = load_language_model(cls.language, cls.filter_config)
        else:
            cls.options["nlp"] = False
        cls.stopwords = cls.__get_stopwords(stopwords)
        cls.ngrams = ngrams or 0
        if cls.ngrams:
            cls.ngram_window = cls.ngrams + ngram_gap
            cls.ngram_word_order = ngram_word_order
        cls.text_object_type = text_object_type
        cls.return_type = return_type
        cls.token_regex = re.compile(rf"({word_regex})|([^{word_regex}])")
        cls.word_tokenizer = re.compile(word_regex)
        cls.sentence_tokenizer = re.compile(sentence_regex)
        if workers is None:
            cpu_count = os.cpu_count() or 2
            cls.workers = cpu_count - 1
        else:
            cls.workers = workers
        cls.post_func = post_processing_function

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        preproc_repr = ""
        preproc_repr += "### Preprocessing options ###\n"
        preproc_repr += f"Language: {self.language}\n"
        for key, value in self.options.items():
            preproc_repr += f"{key}: {value}\n"
        if self.options["spacy_lemmatizer"] is False:
            preproc_repr += f"Lemmatizer path: {self.lemmatizer_path}\n"
        preproc_repr += f"word_regex: {self.word_regex}\n"
        preproc_repr += f"ngrams: {self.ngrams}\n"
        for key, value in self.filter_config.items():
            preproc_repr += f"{key}: {value}\n"
        preproc_repr += f"text_object_type: {self.text_object_type}\n"
        return preproc_repr

    @classmethod
    def process_texts(
        cls,
        texts: Iterable[str],
        keep_all: bool = False,
        progress: bool = True,
        progress_prefix="Processing texts...",
    ) -> Iterable[Tokens]:
        """Process all documents. Returns an iterator of documents"""
        count: int = 0
        doc_count: int = 0
        cls.keep_all = keep_all
        if progress is True:
            print(f"{progress_prefix}", end="", flush=True)

        with Pool(cls.workers) as pool:
            for processed_docs in pool.imap_unordered(cls.__local_process, texts):
                doc_count += 1
                for processed_doc in processed_docs:
                    if progress is True:
                        count += 1
                        print(
                            f"\r{progress_prefix} {doc_count} done, {count} text objects extracted",
                            end="",
                            flush=True,
                        )
                    yield processed_doc
        if progress is True:
            print()

    @classmethod
    def __local_process(cls, text):
        if cls.options["is_philo_db"] is True:
            text_objects, metadata = cls.process_philo_text(text)
            if cls.post_func is None:
                if cls.options["nlp"] is True:
                    return [
                        cls.format(cls.__run_nlp(processed_text), obj_metadata)
                        for processed_text, obj_metadata in zip(text_objects, metadata)
                    ]
                return [
                    cls.format(processed_text, obj_metadata)
                    for processed_text, obj_metadata in zip(text_objects, metadata)
                ]
            else:
                if cls.options["nlp"] is True:
                    return [
                        cls.post_func(cls.format(cls.__run_nlp(processed_text), obj_metadata))
                        for processed_text, obj_metadata in zip(text_objects, metadata)
                    ]
                return [
                    cls.post_func(cls.format(processed_text, obj_metadata))
                    for processed_text, obj_metadata in zip(text_objects, metadata)
                ]
        doc, metadata = cls.process_text(text)
        if cls.post_func is None:
            return [cls.format(doc, metadata)]
        return [cls.post_func(cls.format(doc, metadata))]

    @classmethod
    def process_text(cls, text: str):
        """Process one document. Return the transformed document"""
        with open(text, encoding="utf-8") as input_text:
            doc: str = input_text.read()
        tokens = cls.tokenize_text(doc)
        metadata: Dict[str, Any] = {"filename": text}
        if cls.options["nlp"] is True:
            return cls.nlp(tokens), metadata
        elif cls.lemmatizer and cls.options["spacy_lemmatizer"] is False:
            tokens = [Token(cls.lemmatizer.get(word, word), word.surface_form, ext=word.ext) for word in tokens]
        else:
            tokens = list(tokens)  # We need to convert generator to list for pickling in multiprocessing
        return tokens, metadata

    @classmethod
    def process_string(cls, text, keep_all=True):
        """Take a string and return a list of preprocessed tokens"""
        tokens = list(cls.tokenize_text(text))
        cls.keep_all = keep_all
        if cls.options["with_pos"] is True or cls.pos_to_keep or cls.options["spacy_lemmatizer"] is True:
            tokens = cls.__run_nlp(tokens)
        elif cls.lemmatizer and cls.options["spacy_lemmatizer"] is False:
            tokens = [Token(cls.lemmatizer.get(word, word), word.surface_form, ext=word.ext) for word in tokens]
        return cls.__normalize_doc(tokens)

    @classmethod
    def process_philo_text(cls, text: str, fetch_metadata: bool = True):
        """Process files produced by PhiloLogic parser"""
        docs: List = []
        current_object_id: str = ""
        current_text_object: List = []
        text_path: str = os.path.abspath(os.path.join(text, os.pardir, os.pardir, "TEXT"))
        db_path: str = os.path.abspath(os.path.join(text, os.pardir, os.pardir, "toms.db"))
        if os.path.exists(db_path) is False:
            fetch_metadata = False
        else:
            db: sqlite3.Connection = sqlite3.connect(db_path)
            db.row_factory = sqlite3.Row
            cursor: sqlite3.Cursor = db.cursor()
        metadata_cache: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)
        metadata: List = []
        if text.endswith(".lz4"):
            open_file = lz4.frame.open
        else:
            open_file = open
        with open_file(text) as philo_db_text:
            for line in philo_db_text:
                word_obj: Dict[str, Any] = orjson.loads(line.strip())
                object_id = " ".join(word_obj["position"].split()[: PHILO_TEXT_OBJECT_TYPE[cls.text_object_type]])
                if current_object_id == "":
                    current_object_id = object_id
                if object_id != current_object_id:
                    if current_text_object:
                        if fetch_metadata is True:
                            obj_metadata, metadata_cache = recursive_search(
                                cursor,
                                current_object_id,
                                cls.text_object_type,
                                metadata_cache,
                                text_path,
                                text,
                            )
                            metadata.append(obj_metadata)
                        else:
                            metadata.append(os.path.basename(text))
                        docs.append(current_text_object)
                        current_text_object = []
                    current_object_id = object_id
                if cls.options["modernize"]:
                    word_obj["token"] = cls.modernize(word_obj["token"])
                    current_text_object.append(
                        Token(
                            cls.modernize(word_obj["token"]),
                            word_obj["token"],
                            "",
                            "",
                            word_obj,
                        )
                    )
                else:
                    current_text_object.append(Token(word_obj["token"], word_obj["token"], "", "", word_obj))
        if current_text_object:
            if fetch_metadata is True:
                obj_metadata, _ = recursive_search(
                    cursor,
                    current_object_id,
                    cls.text_object_type,
                    metadata_cache,
                    text_path,
                    text,
                )
                metadata.append(obj_metadata)
            docs.append(current_text_object)
        if fetch_metadata is False:
            metadata = [{"filename": os.path.basename(text)}]  # Return empty shell matching the number of docs returned
        return docs, metadata

    @classmethod
    def __get_stopwords(cls, file_path: Optional[str]) -> Set[str]:
        if file_path is None:
            return set()
        elif os.path.isfile(file_path) is False:
            print("Stopwords file", file_path, "not found. Exiting...")
            exit()
        stopwords = set()
        with open(file_path, encoding="utf-8") as stopword_file:
            for line in stopword_file:
                stopwords.add(line.strip())
        return stopwords

    @classmethod
    def __get_lemmatizer(cls, file_path: Optional[str]) -> Dict[str, str]:
        if file_path is None or os.path.isfile(file_path) is False:
            return {}
        lemmas: dict = {}
        with open(file_path, encoding="utf-8") as input_file:
            for line in input_file:
                word, lemma = line.strip().split("\t")
                lemmas[word] = lemma
        return lemmas

    @classmethod
    def __run_nlp(cls, tokens: Iterable[Token]) -> List[Token]:
        text_data = " ".join(tokens)
        try:
            doc = cls.nlp(text_data)
        except ValueError:  # text is longer than 1,000,000 characters (default spacy limit)
            nlp = load_language_model(cls.language, cls.filter_config)
            nlp.max_length = len(text_data)
            doc = nlp(text_data)
        processed_doc: List[Token] = []
        for token, old_token in zip(doc, tokens):
            keep_token = True
            if cls.filter_config["ents_to_keep"] is not None and token.ent_type_ != "":
                if token.ent_type_ not in cls.filter_config["ents_to_keep"]:
                    keep_token = False
            if cls.filter_config["pos_to_keep"] is not None and token.pos_ not in cls.filter_config["pos_to_keep"]:
                keep_token = False
            if keep_token is False and cls.keep_all is True:
                processed_doc.append(Token("", old_token.surface_form, token.pos_, token.ent_type_, old_token.ext))
                continue
            # print(keep_token)
            if cls.options["spacy_lemmatizer"] is True:
                new_token = Token(token.lemma_, old_token.surface_form, token.pos_, token.ent_type_, old_token.ext)
            elif cls.lemmatizer:
                new_token = Token(
                    cls.lemmatizer.get(token.text.lower(), token.text.lower()),
                    old_token.surface_form,
                    token.pos_,
                    token.ent_type_,
                    old_token.ext,
                )
            else:
                new_token = Token(token.text, old_token.surface_form, token.pos_, token.ent_type_, old_token.ext)
            processed_doc.append(new_token)
        return processed_doc

    @classmethod
    def __generate_ngrams(cls, tokens: Iterable[Token]) -> List[Token]:
        ngrams: List[Token] = []
        ngram: Deque[Token] = deque()
        ngram_text: str
        for token in tokens:
            ngram.append(token)
            if len(ngram) == cls.ngram_window:
                for local_ngram in combinations(ngram, cls.ngrams):
                    ext: Dict[str, Any] = local_ngram[0].ext
                    if cls.ngram_word_order is True:
                        ngram_text = "_".join(t.text for t in local_ngram)
                        ext["end_byte"] = local_ngram[-1].ext["end_byte"]
                    else:
                        ngram_text = "_".join(t.text for t in sorted(local_ngram, key=lambda x: x.text))
                        ngram_sorted_position = sorted(local_ngram, key=lambda x: x.ext["start_byte"])
                        ext["start_byte"] = ngram_sorted_position[0].ext["start_byte"]
                        ext["end_byte"] = ngram_sorted_position[-1].ext["end_byte"]
                    ngrams.append(Token(ngram_text, ngram_text, "", "", ext))
                ngram.popleft()
        return ngrams

    @classmethod
    def normalize(cls, orig_token: Token) -> str:  # This function can be used standalone
        """Normalize a single string token"""
        token = orig_token.text
        token = token.strip()
        if cls.options["convert_entities"] is True:
            token = entities_to_string(token)
        if cls.options["lowercase"] is True:
            token = token.lower()
        if token in cls.stopwords or orig_token.surface_form in cls.stopwords:
            return ""
        if cls.options["strip_punctuation"] is True:
            token = token.translate(PUNCTUATION_MAP)
        elif token in PUNCTUATION_CLASS:
            return token
        if cls.options["strip_numbers"] is True and NUMBERS.search(token):
            return ""
        if cls.stemmer is not False:
            token = cls.stemmer.stemWord(token)  # type: ignore
        if len(token) < cls.min_word_length:
            return ""
        if cls.options["ascii"] is True:
            token = unidecode(token)
        if cls.options["hash_tokens"] is True:
            token = str(mmh3.hash(token))
        return token  # type: ignore

    @classmethod
    def __normalize_doc(cls, doc: List[Token]) -> List[Token]:
        """Normalize single documents"""
        normalized_doc: List[Token] = []
        for inner_token in doc:
            normalized_token = cls.normalize(inner_token)
            if normalized_token or cls.keep_all is True:
                normalized_doc.append(
                    Token(
                        normalized_token,
                        inner_token.surface_form,
                        inner_token.pos_,
                        inner_token.ent,
                        inner_token.ext,
                    )
                )
        return normalized_doc

    @classmethod
    def format(cls, doc: List[Token], metadata: Dict[str, Any]) -> Tokens:
        """Format output"""
        doc = cls.__normalize_doc(doc)
        if cls.ngrams:
            doc = cls.__generate_ngrams(doc)
        return Tokens(doc, metadata)

    @classmethod
    def remove_tags(cls, text: str) -> str:
        """Strip XML tags"""
        end_header_index: int = text.rfind("</teiHeader>")
        if end_header_index != -1:
            end_header_index += 12
            text = text[end_header_index:]
        text = TAGS.sub("", text)
        return text

    @classmethod
    def tokenize_text(cls, doc: str) -> Iterable[Token]:
        """Tokenize text"""
        if cls.options["strip_tags"]:
            doc = cls.remove_tags(doc)
        for match in cls.token_regex.finditer(doc):
            if cls.options["modernize"]:
                yield Token(cls.modernize(match[0]), match[0])
            else:
                yield Token(match[0], match[0])


def recursive_search(
    cursor: sqlite3.Cursor,
    position: str,
    object_type: str,
    metadata_cache: DefaultDict[str, Dict[str, Any]],
    text_path: str,
    text: str,
) -> Tuple[Dict[str, Any], DefaultDict[str, Dict[str, Any]]]:
    """Recursive look-up of PhiloLogic objects"""
    object_id = position.split()
    object_level = PHILO_TEXT_OBJECT_TYPE[object_type]
    obj_metadata: Dict[str, Any] = {"parsed_filename": text}
    while object_id:
        current_id = f"{' '.join(object_id[:object_level])} {' '.join('0' for _ in range(7 - object_level))}"
        if current_id in metadata_cache:
            result = metadata_cache[current_id]
        else:
            cursor.execute("SELECT * from toms WHERE philo_id = ?", (current_id,))
            result = cursor.fetchone()
        if result is not None:
            for field in result.keys():
                if field not in obj_metadata:
                    if result[field] or object_level == 1:  # make sure we get all metadata stored at the last level
                        if field == "filename":
                            obj_metadata[field] = os.path.join(text_path, result[field])
                        else:
                            if field not in obj_metadata or obj_metadata[field]:
                                obj_metadata[field] = result[field]
                            if obj_metadata[field] is None:
                                obj_metadata[field] = ""
                        metadata_cache[current_id][field] = obj_metadata[field]
            current_philo_type = PHILO_OBJECT_LEVEL[object_level]
            philo_object_id = f"philo_{current_philo_type}_id"
            if philo_object_id not in obj_metadata or not obj_metadata[philo_object_id]:
                philo_object_id_value = " ".join(object_id[:object_level])
                obj_metadata[philo_object_id] = philo_object_id_value
                metadata_cache[current_id][philo_object_id] = philo_object_id_value
        object_id.pop()
        object_level -= 1
    return obj_metadata, metadata_cache


def entities_to_string(text):
    """Convert entities to text"""
    text = unescape_html(text)
    text = unescape_xml(text)
    return text


def main():
    """Performance testing"""
    import timeit
    from math import floor

    word_num = 0
    for file in os.scandir(sys.argv[1]):
        word_num += len(open(file.path, encoding="utf-8").readlines())
    preproc = PreProcessor()
    start_time = timeit.default_timer()
    _ = [f for f in preproc.process_texts((i.path for i in os.scandir(sys.argv[1])))]
    elapsed = timeit.default_timer() - start_time
    print("\n", elapsed, floor(word_num / elapsed), "words processed per second")


if __name__ == "__main__":
    main()
