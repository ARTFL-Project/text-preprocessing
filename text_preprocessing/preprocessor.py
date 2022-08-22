#!/usr/bin/env python3
"""Text Preprocessor"""

import json
import os
import re
import sqlite3
import sys
from collections import defaultdict, deque
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

import lz4.frame
import rapidjson
from multiprocess.pool import Pool
from spacy.tokens import Doc

from spacy_helpers import load_language_model, tokens_to_doc

from .modernizer import Modernizer

TRIM_LAST_SLASH = re.compile(r"/\Z")
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

    def __new__(cls, value, surface_form="", pos="", ent="", ext=""):
        return str.__new__(cls, value)

    def __init__(
        self,
        text: str,
        surface_form: str = "",
        pos: str = "",
        ent: str = "",
        ext: Optional[Dict[str, Any]] = None,
    ):
        self.text = text or ""
        self.surface_form = surface_form or text
        self.ext = ext or {}
        self.ext["pos"] = pos
        self.pos_ = pos
        self.ent_type_ = ent

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

    metadata: Dict[str, Any]

    def __init__(self, tokens: Union[Doc, Iterable[Token]], metadata: Dict[str, Any]):
        if isinstance(tokens, Doc):
            self.tokens, self.metadata = self.__spacy_doc_to_tokens(tokens)
        else:
            self.tokens: Deque[Token] = deque(tokens)
            self.metadata = metadata
        self.length: int = len(self.tokens)
        self.iter_index = 0

    def __spacy_doc_to_tokens(self, doc: Doc) -> Tuple[Deque[Token], Dict[str, Any]]:
        tokens = deque()
        for token in doc:
            if token.text != "EMPTY_STRING":
                local_token = Token(
                    token.text, surface_form=token._.surface_form, pos=token.pos_, ent=token.ent_type_, ext=token._.ext
                )
                tokens.append(local_token)
        return tokens, doc.user_data

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
    modernize: bool = False
    modernizer: Callable = lambda x: x  # workaround for mypy
    ngrams: int = 0
    ngram_gap: int = 0
    stopwords: Set[str] = set()
    min_word_length: int = 2
    pos_to_keep: Set[str] = set()
    text_object_type: str = "doc"
    return_type: str = "words"
    workers: Optional[int] = None
    post_func: Optional[Callable] = None
    token_regex: re.Pattern = re.compile(r"")
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
        **extra_options,  # this is meant to make the constructor accept invalid keywords
    ):
        cls.language = language
        cls.is_philo_db = is_philo_db
        cls.normalizer_config = {
            "convert_entities": convert_entities,
            "lowercase": lowercase,
            "strip_punctuation": strip_punctuation,
            "strip_numbers": strip_numbers,
            "stemmer": stemmer,
            "ascii": ascii,
            "hash_tokens": hash_tokens,
            "min_word_length": min_word_length,
            "stopwords": cls.__get_stopwords(stopwords),
            "language": language,
            "lemmatizer": cls.__get_lemmatizer(lemmatizer),
        }
        cls.filter_config = {
            "pos_to_keep": pos_to_keep,
            "ents_to_keep": ents_to_keep,
        }
        if modernize is True:
            cls.modernize = modernize
            cls.modernizer = Modernizer(language)
        cls.tokenizer_config = {
            "language": language,
            "modernize": modernize,
            "strip_tags": strip_tags,
            "token_regex": re.compile(rf"({word_regex})|([^{word_regex}])"),
        }
        cls.ngram_config = {"ngram_window": ngrams or 0 + ngram_gap, "ngram_word_order": ngram_word_order}
        cls.nlp = load_language_model(
            cls.language,
            cls.tokenizer_config,
            cls.normalizer_config,
            cls.filter_config,
            cls.ngram_config,
            cls.is_philo_db,
        )
        cls.text_object_type = text_object_type
        cls.return_type = return_type
        cls.sentence_tokenizer = re.compile(sentence_regex)
        if workers is None:
            cpu_count = os.cpu_count() or 2
            cls.workers = cpu_count - 1
        else:
            cls.workers = workers
        cls.post_func = post_processing_function

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
        if cls.is_philo_db is True:
            text_objects, metadata = cls.process_philo_text(text)
            for text_tokens, text_metadata in zip(text_objects, metadata):
                doc = cls.nlp(tokens_to_doc(text_tokens, cls.nlp.vocab, text_metadata))
            if cls.post_func is None:
                return [
                    cls.format(cls.nlp(tokens_to_doc(text_tokens, cls.nlp.vocab, text_metadata)))
                    for text_tokens, text_metadata in zip(text_objects, metadata)
                ]
            else:
                return [
                    cls.post_func(cls.format(cls.nlp(tokens_to_doc(text_tokens, cls.nlp.vocab, text_metadata))))
                    for text_tokens, text_metadata in zip(text_objects, metadata)
                ]
        try:
            doc = cls.nlp(text)
        except ValueError:  # text is longer than 1,000,000 characters (default spacy limit)
            nlp = load_language_model(
                cls.language,
                cls.tokenizer_config,
                cls.normalizer_config,
                filter_config=cls.filter_config,
                ngram_config=cls.ngram_config,
            )
            with open(text, encoding="utf-8") as text_file:
                text_data = text_file.read()
            nlp.max_length = len(text_data)
            doc = nlp(text)
            doc.user_data = {"filename": text}
        if cls.post_func is None:
            return [cls.format(doc)]
        return [cls.post_func(cls.format(doc))]

    @classmethod
    def process_string(cls, text, keep_all=True):
        """Take a string and return a list of preprocessed tokens"""
        doc = cls.nlp(text)
        return Tokens(doc)

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
                word_obj: Dict[str, Any] = rapidjson.loads(line.strip())
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
                if cls.modernize:
                    word_obj["token"] = cls.modernizer(word_obj["token"])
                    current_text_object.append(
                        Token(
                            cls.modernizer(word_obj["token"]),
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
            return []
        elif os.path.isfile(file_path) is False:
            print("Stopwords file", file_path, "not found. Exiting...")
            exit()
        stopwords = []
        with open(file_path) as stopword_file:
            for line in stopword_file:
                stopwords.append(line.strip())
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
    def format(cls, doc: Doc) -> Tokens:
        """Format output"""
        return Tokens(doc, doc.user_data)


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


def main():
    """Performance testing"""
    import timeit
    from math import floor

    word_num = 0
    for file in os.scandir(sys.argv[1]):
        word_num += len(open(file.path).readlines())
    preproc = PreProcessor()
    start_time = timeit.default_timer()
    _ = [f for f in preproc.process_texts((i.path for i in os.scandir(sys.argv[1])))]
    elapsed = timeit.default_timer() - start_time
    print("\n", elapsed, floor(word_num / elapsed), "words processed per second")


if __name__ == "__main__":
    main()
