#!/usr/bin/env python3
"""Text Preprocessor"""

import json
import os
import sqlite3
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, DefaultDict, Deque, Iterable, Iterator, Union, overload

import lz4.frame
import orjson
import regex as re
from multiprocess.pool import Pool
from spacy.language import Language
from spacy.tokens import Doc, Token

from .modernizer import Modernizer
from .spacy_helpers import load_language_model

Doc.set_extension("metadata", default={})
Doc.set_extension("char_num", default=0)
Token.set_extension("ext", default={})

TAGS = re.compile(r"<[^>]+>")

PHILO_TEXT_OBJECT_TYPE: dict[str, int] = {
    "doc": 1,
    "div1": 2,
    "div2": 3,
    "div3": 4,
    "para": 5,
    "sent": 6,
    "word": 7,
}

PHILO_OBJECT_LEVEL: dict[int, str] = {1: "doc", 2: "div1", 3: "div2", 4: "div3", 5: "para", 6: "sent", 7: "word"}


@dataclass(slots=True)
class PreparedDoc:
    """Prepared doc for conversion to Spacy Doc object"""

    words: list[str]
    sent_starts: list[bool]
    metadata: dict[str, Any]
    exts: list[dict[str, Any]]
    char_num: int


class PreprocessorToken(str):
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

    ext: dict[str, Any]

    def __new__(cls, value, pos_="", ent="", ext={}):
        return str.__new__(cls, value)

    def __init__(
        self,
        text: str,
        pos_: str = "",
        ent: str = "",
        ext: dict[str, Any] | None = None,
    ):
        self.text = text or ""
        self.ext = ext or {}
        if self.ext is not None:
            self.surface_form = ext["token"]
        else:
            self.surface_form = text
        self.ext["pos"] = pos_
        self.pos_ = pos_
        self.ent = ent

    def __hash__(self):
        return hash(self.text)

    def __eq__(self, other) -> bool:
        if isinstance(other, PreprocessorToken):
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

    def __init__(self, doc: Doc | Iterable[PreprocessorToken], metadata=None, keep_all=False):
        self.keep_all = keep_all
        if isinstance(doc, Doc):
            self.tokens: Deque[PreprocessorToken] = Deque(self.__get_tokens(doc))
        else:
            self.tokens = Deque(doc)
        if metadata is None:
            self.metadata: dict[str, Any] = doc._.metadata  # type: ignore
        else:
            self.metadata = metadata
        self.length: int = len(self.tokens)
        self.iter_index = 0

    def __get_tokens(self, doc: Doc):
        """Return a generator of PreprocessorToken objects"""
        for token in doc:
            if token.text != "#DEL#":
                yield PreprocessorToken(token.text, token.pos_, token.ent_type_, token._.ext)
            elif self.keep_all is True:
                yield PreprocessorToken("", token.pos_, token.ent_type_, token._.ext)
                if token.whitespace_:
                    yield PreprocessorToken(token.whitespace_, "", "", {"token": token.whitespace_})

    def __iter__(self) -> Iterator[PreprocessorToken]:
        for token in self.tokens:
            yield token

    def __next__(self):
        self.iter_index += 1
        if self.iter_index < self.length:
            return self.tokens[self.iter_index]
        else:
            raise IndexError

    @overload
    def __getitem__(self, index: int) -> PreprocessorToken:
        ...

    @overload
    def __getitem__(self, index: slice) -> Iterable[PreprocessorToken]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[PreprocessorToken, Iterable[PreprocessorToken]]:
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
                metadata: dict[str, Any] = {
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

    def pop(self) -> PreprocessorToken | None:
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

    def popleft(self) -> PreprocessorToken | None:
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

    def append(self, token: PreprocessorToken):
        """Append Token"""
        if not self.tokens:
            self.metadata["start_byte"] = token.ext["start_byte"]
        self.tokens.append(token)
        self.metadata["end_byte"] = token.ext["end_byte"]
        self.length += 1

    def appendleft(self, token: PreprocessorToken):
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
        with open(path, "w", encoding="utf8") as output:
            json.dump(tokens_to_serialize, output)

    def load(self, path):
        """Load tokens from disk"""
        with open(path, "r", encoding="utf8") as input_file:
            tokens = json.load(input_file)
        self.metadata = tokens["metadata"]
        self.tokens = deque(PreprocessorToken(t[0], t[1], t[2], t[3]) for t in tokens["tokens"])


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

    def __init__(
        self,
        word_regex: str = r"[\p{L}\p{M}\p{N}]+|'",
        sentence_boundaries: list[str] = [".", "!", "?"],
        language: str = "french",
        modernize: bool = False,
        strip_tags: bool = False,
        is_philo_db: bool = False,
        text_object_type: str = "doc",
        workers: int | None = None,
        stemmer: bool = False,
        lemmatizer: str | bool = False,
        ngrams: int | bool = False,
        ngram_gap: int = 0,
        ngram_word_order: bool = True,
        stopwords: str | bool = False,
        strip_punctuation: bool = True,
        strip_numbers: bool = True,
        lowercase: bool = True,
        min_word_length: int = 2,
        ascii: bool = False,
        convert_entities: bool = False,
        pos_to_keep: list[str] | bool = False,
        ents_to_keep: list[str] | bool = False,
        post_processing_function: Callable | None = None,
        **_,  # this is meant to make the constructor accept invalid keywords
    ):
        self.normalize_options = {
            "language": language,
            "stemmer": stemmer,
            "stopwords": stopwords,
            "strip_punctuation": strip_punctuation,
            "strip_numbers": strip_numbers,
            "lowercase": lowercase,
            "ascii": ascii,
            "convert_entities": convert_entities,
            "min_word_length": min_word_length,
            "lemmatizer": lemmatizer,
            "pos_to_keep": pos_to_keep or [],
            "ents_to_keep": ents_to_keep or [],
        }
        self.language = language
        self.nlp = load_language_model(self.language, self.normalize_options)
        if workers is None:
            cpu_count = os.cpu_count() or 2
            self.workers = cpu_count - 1
        else:
            self.workers = workers
        ngrams = ngrams or 0
        if ngrams:
            self.ngram_config = {
                "ngram_size": ngrams,
                "ngram_window": ngrams + ngram_gap,
                "ngram_word_order": ngram_word_order,
            }
        else:
            self.ngram_config = None
        self.post_func = post_processing_function
        self.text_fetcher = TextFetcher(
            self.nlp,
            word_regex=word_regex,
            sentence_boundaries=sentence_boundaries,
            language=language,
            modernize=modernize,
            strip_tags=strip_tags,
            is_philo_db=is_philo_db,
            text_object_type=text_object_type,
            ngram_config=self.ngram_config,
            workers=workers,
        )
        if self.normalize_options["pos_to_keep"] or self.normalize_options["ents_to_keep"]:
            self.do_nlp = True
        else:
            self.do_nlp = False

    def process_texts(
        self,
        texts: Iterable[str],
        keep_all=False,
        progress: bool = True,
        progress_prefix="Processing texts...",
    ) -> Iterable[Tokens]:
        """Process all documents. Returns an iterator of documents"""
        count = 0
        fetched_texts = self.text_fetcher(
            texts, do_nlp=self.do_nlp, keep_all=keep_all, progress=progress, post_func=self.post_func
        )
        for tokens, doc_count in fetched_texts:
            count += 1
            if progress is True:
                print(
                    f"\r{progress_prefix} {doc_count} done: {count} text objects extracted... ",
                    end="",
                    flush=True,
                )
            if isinstance(tokens, PreparedDoc):
                spacy_doc = make_spacy_doc(self.nlp, tokens)
                if spacy_doc._.char_num > 10000:
                    split_doc = self.__split_spacy_docs(spacy_doc)
                    rebuilt_doc = Doc.from_docs(list(self.nlp.pipe(split_doc)))
                    rebuilt_doc._.metadata = spacy_doc._.metadata
                    tokens = Tokens(rebuilt_doc, keep_all=keep_all)
                else:
                    tokens = Tokens(self.nlp(spacy_doc), keep_all=keep_all)
                if self.ngram_config is not None:
                    tokens = generate_ngrams(**self.ngram_config, tokens=tokens)
                if self.post_func is not None:
                    processed_doc = self.post_func(tokens)
                    yield processed_doc
            else:
                yield tokens

    def process_string(self, text: str, keep_all: bool = True) -> Tokens:
        """Take a string and return a list of preprocessed tokens"""
        doc = self.text_fetcher.process_string(text)
        processed_doc = self.nlp(doc)
        return Tokens(processed_doc, keep_all=keep_all)

    def __split_spacy_docs(self, doc: Doc) -> list[Doc]:
        """Split spacy doc into smaller docs of 10 sentences"""
        sentence_group: list[Doc] = []
        docs: list[Doc] = []
        for sent in doc.sents:
            if len(sentence_group) == 10:
                docs.append(Doc.from_docs(sentence_group))
                sentence_group = []
            else:
                sent_starts = []
                words = []
                for token in sent:
                    sent_starts.append(token.is_sent_start)
                    words.append(token.text)
                sent_doc = Doc(self.nlp.vocab, words, sent_starts=sent_starts)
                for pos, token in enumerate(sent):
                    sent_doc[pos]._.ext = token._.ext
                sentence_group.append(sent_doc)
        if sentence_group:
            docs.append(Doc.from_docs(sentence_group))
        return docs


class TextFetcher:
    """Text fetcher"""

    word_regex: str = r"[\p{L}\p{M}\p{N}]+|'"
    sentence_boundaries: list[str] = [".", "!", "?"]
    language: str = "french"
    modernize: bool | Callable = lambda x: x
    text_object_type: str = "doc"
    workers: int | None = None
    token_regex: re.Pattern = re.compile(r"")
    strip_tags: bool = False
    model: Language | Callable = lambda x: x
    ngram_config: dict[str, Any] | None = None

    @classmethod
    def __init__(
        cls,
        model: Language,
        word_regex=r"[\p{L}\p{M}\p{N}]+|'",
        sentence_boundaries=[".", "!", "?"],
        language="french",
        modernize=False,
        strip_tags=False,
        is_philo_db=False,
        text_object_type="doc",
        ngram_config=None,
        workers=None,
        **_,  # this is meant to make the constructor accept invalid keywords
    ):
        cls.language = language
        cls.model = model
        if modernize is True:
            cls.modernize = Modernizer(language)
        else:
            cls.modernize = False
        cls.strip_tags = strip_tags

        cls.is_philo_db = is_philo_db

        cls.text_object_type = text_object_type
        cls.token_regex = re.compile(rf"({word_regex})|([{''.join(sentence_boundaries)}])")
        cls.sentence_boundaries = sentence_boundaries
        if workers is None:
            cpu_count = os.cpu_count() or 2
            cls.workers = cpu_count - 1
        else:
            cls.workers = workers
        cls.ngram_config = ngram_config

    @classmethod
    def __call__(
        cls,
        texts: Iterable[str],
        do_nlp: bool = False,
        keep_all=False,
        progress: bool = True,
        post_func: Callable | None = None,
    ) -> Iterable[tuple[PreparedDoc | Tokens, int]]:
        """Process all documents. Returns an iterator of documents"""
        doc_count: int = 0
        if progress is True:
            print("Extractings texts...", end="", flush=True)

        with Pool(cls.workers) as pool:
            for processed_docs in pool.imap_unordered(
                cls.__local_process, ((text, do_nlp, keep_all, post_func) for text in texts)
            ):
                doc_count += 1
                for doc in processed_docs:
                    yield doc, doc_count

    @classmethod
    def __local_process(cls, args) -> Iterable[PreparedDoc | Tokens]:
        text, do_nlp, keep_all, post_func = args
        if cls.is_philo_db is True:
            text_objects, sent_starts_list, metadata = cls.process_philo_text(text)
        else:
            text_objects, sent_starts_list, metadata = cls.process_text(text)
        docs = cls.__prepare_docs(text_objects, sent_starts_list, metadata)
        if do_nlp is True:
            return docs
        spacy_docs = (make_spacy_doc(cls.model, doc) for doc in docs)
        tokens_list: list[Tokens] = []
        for spacy_doc in spacy_docs:
            tokens = Tokens(cls.model(spacy_doc), keep_all=keep_all)
            if cls.ngram_config is not None:
                tokens = generate_ngrams(**cls.ngram_config, tokens=tokens)
            if post_func is not None:
                tokens = post_func(tokens)
            tokens_list.append(tokens)
        return tokens_list

    @classmethod
    def __prepare_docs(cls, text_objects, sent_starts_list, metadata) -> list[PreparedDoc]:
        """Prepare doc for creating Spacy doc"""
        list_doc_words: list[PreparedDoc] = []
        for processed_doc, sent_starts, local_metadata in zip(text_objects, sent_starts_list, metadata):
            words = []
            exts = []
            char_num = 0
            for word, ext in processed_doc:
                char_num += len(word)
                words.append(word)
                exts.append(ext)
            list_doc_words.append(PreparedDoc(words, sent_starts, local_metadata, exts, char_num))
        return list_doc_words

    @classmethod
    def process_text(cls, text: str):
        """Process one document. Return the transformed document"""
        with open(text, encoding="utf-8") as input_text:
            doc: str = input_text.read()
        tokens, sent_starts = cls.tokenize_text(doc)
        metadata: dict[str, Any] = {"filename": text}
        return tokens, sent_starts, metadata

    @classmethod
    def process_string(cls, text: str) -> Doc:
        """Process one string. Return the transformed document"""
        tokens, sent_starts = cls.tokenize_text(text)
        doc = Doc(cls.model.vocab, [word for word, _ in tokens], sent_starts=sent_starts)  # type: ignore
        for pos, (_, ext) in enumerate(tokens):
            doc[pos]._.ext = ext
        return doc

    @classmethod
    def tokenize_text(cls, doc: str) -> tuple[list[tuple[str, dict[str, str]]], list[bool]]:
        """Tokenize text"""
        if cls.strip_tags is True:
            doc = remove_tags(doc)
        tokens: list[tuple[str, dict[str, str]]] = []
        sent_starts: list[bool] = []
        new_sent: bool = False
        for match in cls.token_regex.finditer(doc):
            sent_starts.append(new_sent)
            new_sent = match[0] in cls.sentence_boundaries
            if cls.modernize is not False:
                tokens.append((cls.modernize(match[0]), {"token": match[0]}))  # type: ignore
            else:
                tokens.append((match[0], {"token": match[0]}))
        return tokens, sent_starts

    @classmethod
    def process_philo_text(cls, text: str, fetch_metadata: bool = True):
        """Process files produced by PhiloLogic parser"""
        docs: list = []
        current_object_id: str = ""
        current_text_object: list[tuple[str, dict[str, Any]]] = []
        text_path: str = os.path.abspath(os.path.join(text, os.pardir, os.pardir, "TEXT"))
        db_path: str = os.path.abspath(os.path.join(text, os.pardir, os.pardir, "toms.db"))
        if os.path.exists(db_path) is False:
            fetch_metadata = False
        else:
            db: sqlite3.Connection = sqlite3.connect(db_path)
            db.row_factory = sqlite3.Row
            cursor: sqlite3.Cursor = db.cursor()
        metadata_cache: DefaultDict[str, dict[str, Any]] = defaultdict(dict)
        metadata: list = []
        if text.endswith(".lz4"):
            open_file = lz4.frame.open
        else:
            open_file = open
        sent_id: str | None = None
        sent_starts_list: list[list[bool]] = []
        sent_starts: list[bool] = []
        with open_file(text) as philo_db_text:
            for line in philo_db_text:
                word_obj: dict[str, Any] = orjson.loads(line.strip())
                philo_id = word_obj["position"].split()
                object_id = " ".join(philo_id[: PHILO_TEXT_OBJECT_TYPE[cls.text_object_type]])
                current_sent_id = " ".join(philo_id[: PHILO_TEXT_OBJECT_TYPE["sent"]])
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
                        sent_starts_list.append(sent_starts)
                        sent_starts = []
                        current_text_object = []
                    current_object_id = object_id
                if current_sent_id == sent_id:
                    sent_starts.append(False)
                else:
                    sent_starts.append(True)
                if cls.modernize is not False:
                    current_text_object.append((cls.modernize(word_obj["token"]), word_obj))  # type: ignore
                else:
                    current_text_object.append((word_obj["token"], word_obj))
                sent_id = " ".join(philo_id[: PHILO_TEXT_OBJECT_TYPE["sent"]])
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
            sent_starts_list.append(sent_starts)
            sent_starts = []
        if fetch_metadata is False:
            metadata = [{"filename": os.path.basename(text)}]  # Return empty shell matching the number of docs returned
        return docs, sent_starts_list, metadata


def generate_ngrams(ngram_size: int, ngram_window: int, ngram_word_order: bool, tokens: Tokens) -> Tokens:
    """Generate ngrams from tokens"""
    ngrams: list[PreprocessorToken] = []
    ngram: Deque[PreprocessorToken] = deque()
    ngram_text: str
    for token in tokens:
        ngram.append(token)
        if len(ngram) == ngram_window:
            for local_ngram in combinations(ngram, ngram_size):
                ext: dict[str, Any] = local_ngram[0].ext
                if ngram_word_order is True:
                    ngram_text = "_".join(t.text for t in local_ngram)
                    ext["end_byte"] = local_ngram[-1].ext["end_byte"]
                else:
                    ngram_text = "_".join(t.text for t in sorted(local_ngram, key=lambda x: x.text))
                    ngram_sorted_position = sorted(local_ngram, key=lambda x: x.ext["start_byte"])
                    ext["start_byte"] = ngram_sorted_position[0].ext["start_byte"]
                    ext["end_byte"] = ngram_sorted_position[-1].ext["end_byte"]
                ngrams.append(PreprocessorToken(ngram_text, "", "", ext))
            ngram.popleft()
    return Tokens(ngrams, tokens.metadata)


def remove_tags(text: str) -> str:
    """Strip XML tags"""
    end_header_index: int = text.rfind("</teiHeader>")
    if end_header_index != -1:
        end_header_index += 12
        text = text[end_header_index:]
    text = TAGS.sub("", text)
    return text


def recursive_search(
    cursor: sqlite3.Cursor,
    position: str,
    object_type: str,
    metadata_cache: DefaultDict[str, dict[str, Any]],
    text_path: str,
    text: str,
) -> tuple[dict[str, Any], DefaultDict[str, dict[str, Any]]]:
    """Recursive look-up of PhiloLogic objects"""
    object_id = position.split()
    object_level = PHILO_TEXT_OBJECT_TYPE[object_type]
    obj_metadata: dict[str, Any] = {"parsed_filename": text}
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


def make_spacy_doc(model: Language, prepared_doc: PreparedDoc) -> Doc:
    """Make Spacy doc"""
    doc = Doc(model.vocab, prepared_doc.words, sent_starts=prepared_doc.sent_starts)
    doc._.metadata = prepared_doc.metadata
    doc._.char_num = prepared_doc.char_num
    for pos, ext in enumerate(prepared_doc.exts):
        doc[pos]._.ext = ext
    return doc


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
